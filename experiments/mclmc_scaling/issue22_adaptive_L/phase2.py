import time, os, sys, warnings, json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from jax.flatten_util import ravel_pytree
from scipy.stats import spearmanr
os.chdir("/home/jp/blackjax-devs/tuningfork"); sys.path.insert(0,"/home/jp/blackjax-devs/tuningfork")
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model._registry import MODELS as _M
import blackjax

init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["horseshoe"])
flat0, unravel = ravel_pytree(init_dict)
d = flat0.size; TAU_IDX = 2
ld = lambda xf: ld_raw(unravel(xf))
neglog = lambda xf: -ld(xf)
grad_neg = jax.jit(jax.grad(neglog))
hess = jax.jit(jax.hessian(neglog))
@jax.jit
def hvp(x, v): return jax.jvp(grad_neg, (x,), (v,))[1]

# ---- reference cloud: NUTS, 2 dispersed chains for funnel-depth coverage ----
def nuts_chain(seed, n_warm, n_samp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wa = blackjax.window_adaptation(blackjax.nuts, ld, progress_bar=False)
        (st, params), _ = wa.run(jax.random.key(seed), flat0, num_steps=n_warm)
        nuts = blackjax.nuts(ld, **params)
        _, (pos, dv) = blackjax.util.run_inference_algorithm(
            jax.random.key(seed+100), nuts, n_samp, initial_state=nuts.init(st.position),
            transform=lambda s,i:(s.position, i.is_divergent), progress_bar=False)
    return np.asarray(pos), float(params["step_size"]), int(np.asarray(dv).sum())

t0=time.time()
p1,eps1,dv1 = nuts_chain(1, 400, 600)
p2,eps2,dv2 = nuts_chain(7, 400, 600)
pos = np.vstack([p1,p2]); eps = float(np.mean([eps1,eps2]))
tau = pos[:,TAU_IDX]
print(f"reference cloud: {pos.shape[0]} draws in {time.time()-t0:.1f}s | eps~{eps:.4g} | div={dv1+dv2} | tau range [{tau.min():.2f},{tau.max():.2f}]")

# ---- stratified subsample by tau quantile so each depth bin is populated ----
NBIN=6; rng=np.random.default_rng(0)
edges=np.quantile(tau, np.linspace(0,1,NBIN+1))
sel=[]
for b in range(NBIN):
    idx=np.where((tau>=edges[b])&(tau<=edges[b+1]))[0]
    if len(idx)>0: sel.extend(rng.choice(idx, size=min(30,len(idx)), replace=False))
sel=np.array(sorted(set(sel)))
print(f"subsampled {len(sel)} positions across {NBIN} tau bins")

# ---- per-position geometry ----
def slow_fast(x):
    H=np.asarray(hess(jnp.asarray(x)))
    w=np.linalg.eigvalsh((H+H.T)/2)            # symmetric eigenvalues
    lam_max=float(w[-1])
    pos_w=w[w>1e-6*lam_max]                     # positive curvature directions
    lam_min_pos=float(pos_w.min()) if pos_w.size else float('nan')
    n_nonpos=int((w<=1e-6*lam_max).sum())
    return lam_max, lam_min_pos, n_nonpos, w
rows=[]
t0=time.time()
e_tau=np.zeros(d); e_tau[TAU_IDX]=1.0; e_tau=jnp.asarray(e_tau)
for j,i in enumerate(sel):
    x=jnp.asarray(pos[i])
    lam_max,lam_min,n_np,_=slow_fast(pos[i])
    sk=np.sqrt(lam_max/lam_min) if lam_min==lam_min and lam_min>0 else float('nan')
    # cheap proxies
    pr=jax.random.normal(jax.random.key(j+5),(d,)); pr=pr/jnp.linalg.norm(pr)
    g0=grad_neg(x); proxy_rand=float(jnp.linalg.norm(grad_neg(x+eps*pr)-g0)/eps)
    H_tau=float(jnp.dot(e_tau, hvp(x,e_tau)))   # curvature along tau axis (1 Hvp)
    rows.append(dict(tau=float(pos[i,TAU_IDX]), lam_max=lam_max, lam_min_pos=lam_min,
                     sqrt_kappa=float(sk), n_nonpos=n_np, proxy_rand=proxy_rand, H_tau=H_tau))
print(f"geometry on {len(sel)} positions in {time.time()-t0:.1f}s")

# ---- decision gate ----
R=rows; t_=np.array([r['tau'] for r in R]); sk_=np.array([r['sqrt_kappa'] for r in R])
ok=np.isfinite(sk_)
print("\n=== per tau-depth bin (median sqrt_kappa = implied optimal avg) ===")
bin_med=[]
for b in range(NBIN):
    m=(t_>=edges[b])&(t_<=edges[b+1])&ok
    if m.sum():
        med=float(np.median(sk_[m]))
        bin_med.append(med)
        print(f"  tau in [{edges[b]:+.2f},{edges[b+1]:+.2f}]: n={int(m.sum())} median sqrtK={med:8.1f} "
              f"lam_max~{np.median([r['lam_max'] for r,mm in zip(R,m) if mm]):.2e} "
              f"lam_min~{np.median([r['lam_min_pos'] for r,mm in zip(R,m) if mm]):.2e}")
ratio = max(bin_med)/min(bin_med) if bin_med else float('nan')
print(f"\nGATE-1: sqrt_kappa bin-median range = {min(bin_med):.1f} .. {max(bin_med):.1f} -> ratio {ratio:.1f}x  ({'YES >2-3x' if ratio>3 else 'NO'})")
# gate-2: do cheap proxies track sqrt_kappa?
pr_=np.array([r['proxy_rand'] for r in R]); ht_=np.array([abs(r['H_tau']) for r in R])
lm_=np.array([r['lam_max'] for r in R])
print("\nGATE-2: cheap-proxy correlations (Spearman rho vs target):")
for nm,arr in [("proxy_rand",pr_),("|H_tau|",ht_),("lam_max",lm_)]:
    rho_sk=spearmanr(arr[ok], sk_[ok]).statistic
    rho_tau=spearmanr(arr, np.abs(t_)).statistic
    print(f"  {nm:11s}: rho(.,sqrtK)={rho_sk:+.2f}  rho(.,|tau|)={rho_tau:+.2f}")
rho_tau_sk=spearmanr(np.abs(t_[ok]), sk_[ok]).statistic
print(f"  |tau| (oracle depth): rho(|tau|,sqrtK)={rho_tau_sk:+.2f}")
json.dump({"eps":eps,"n_pos":len(sel),"edges":edges.tolist(),"bin_median_sqrtK":bin_med,
           "gate1_ratio":ratio,"rows":R}, open("/tmp/issue22_scoping/phase2_results.json","w"), indent=2)
print("\nDONE_PHASE2")
