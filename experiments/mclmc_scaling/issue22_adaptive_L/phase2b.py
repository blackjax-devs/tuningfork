import time, os, sys, warnings, json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from scipy.stats import spearmanr
from jax.flatten_util import ravel_pytree
os.chdir("/home/jp/blackjax-devs/tuningfork"); sys.path.insert(0,"/home/jp/blackjax-devs/tuningfork")
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model._registry import MODELS as _M
import blackjax
init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["horseshoe"])
flat0, unravel = ravel_pytree(init_dict); d=flat0.size; TAU_IDX=2
ld=lambda xf: ld_raw(unravel(xf)); neglog=lambda xf:-ld(xf)
hess=jax.jit(jax.hessian(neglog))
def nuts_chain(seed,nw,ns):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wa=blackjax.window_adaptation(blackjax.nuts,ld,progress_bar=False)
        (st,pp),_=wa.run(jax.random.key(seed),flat0,num_steps=nw); nt=blackjax.nuts(ld,**pp)
        _,(pos,dv)=blackjax.util.run_inference_algorithm(jax.random.key(seed+100),nt,ns,
            initial_state=nt.init(st.position),transform=lambda s,i:(s.position,i.is_divergent),progress_bar=False)
    return np.asarray(pos),float(pp["step_size"])
p1,e1=nuts_chain(1,400,600); p2,e2=nuts_chain(7,400,600)
pos=np.vstack([p1,p2]); tau=pos[:,TAU_IDX]; eps=float(np.mean([e1,e2]))
NBIN=6; rng=np.random.default_rng(0); edges=np.quantile(tau,np.linspace(0,1,NBIN+1))
sel=[]
for b in range(NBIN):
    idx=np.where((tau>=edges[b])&(tau<=edges[b+1]))[0]
    if len(idx): sel.extend(rng.choice(idx,size=min(25,len(idx)),replace=False))
sel=np.array(sorted(set(sel)))
print(f"cloud {pos.shape[0]} | eps~{eps:.4g} | tau[{tau.min():.2f},{tau.max():.2f}] | {len(sel)} positions")
specs=[]
for i in sel:
    H=np.asarray(hess(jnp.asarray(pos[i]))); w=np.linalg.eigvalsh((H+H.T)/2)
    specs.append(w)
specs=np.array(specs)   # (n,204) ascending
ti=tau[sel]
print("\nper tau-bin: spectrum tail (no cutoff) — what's the genuine SLOW scale?")
print(f"{'tau bin':>16s} {'n':>3s} {'#neg':>5s} {'lam_min':>10s} {'p05':>9s} {'p25':>9s} {'lam_max':>9s} {'slowL=1/sqrt(p05+)':>18s}")
binstat=[]
for b in range(NBIN):
    m=(ti>=edges[b])&(ti<=edges[b+1])
    if not m.sum(): continue
    sw=specs[m]
    n_neg=np.median((sw<=0).sum(axis=1))
    lmin=np.median(sw[:,0]); lmax=np.median(sw[:,-1])
    # smallest positive per row, then median
    smp=np.median([row[row>1e-10][0] if (row>1e-10).any() else np.nan for row in sw])
    p05=np.median(np.nanpercentile(np.where(sw>1e-10,sw,np.nan),5,axis=1))
    p25=np.median(np.nanpercentile(np.where(sw>1e-10,sw,np.nan),25,axis=1))
    slowL=1/np.sqrt(p05) if p05>0 else np.nan
    binstat.append(dict(bin=b,smp=float(smp),p05=float(p05),lmax=float(lmax),slowL=float(slowL),n_neg=float(n_neg)))
    print(f"[{edges[b]:+.2f},{edges[b+1]:+.2f}] {int(m.sum()):>3d} {n_neg:>5.0f} {lmin:>10.2e} {p05:>9.2e} {p25:>9.2e} {lmax:>9.2e} {slowL:>18.1f}")
# variation of slow length-scale across depth
slowLs=[b['slowL'] for b in binstat]; smps=[b['smp'] for b in binstat]
print(f"\nslow length-scale (1/sqrt(p05+)) across bins: {min(slowLs):.1f}..{max(slowLs):.1f} -> ratio {max(slowLs)/min(slowLs):.2f}x")
print(f"smallest-positive eig across bins: {min(smps):.2e}..{max(smps):.2e} -> ratio {max(smps)/min(smps):.1f}x")
# correlation: does |tau| (depth) predict the slow scale per-draw?
smp_row=np.array([ (row[row>1e-10][0] if (row>1e-10).any() else np.nan) for row in specs])
p05_row=np.array([ np.nanpercentile(np.where(row>1e-10,row,np.nan),5) for row in specs])
ok=np.isfinite(smp_row)
print(f"\nrho(|tau|, smallest_pos_eig) = {spearmanr(np.abs(ti[ok]),smp_row[ok]).statistic:+.2f}")
print(f"rho(|tau|, p05_eig)          = {spearmanr(np.abs(ti),p05_row).statistic:+.2f}")
print(f"rho(tau,   smallest_pos_eig) = {spearmanr(ti[ok],smp_row[ok]).statistic:+.2f}")
json.dump({"eps":eps,"edges":edges.tolist(),"binstat":binstat},open("/tmp/issue22_scoping/phase2b_results.json","w"),indent=2)
print("\nDONE_PHASE2B")
