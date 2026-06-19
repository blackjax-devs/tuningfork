import time, os, sys, warnings
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from jax.flatten_util import ravel_pytree
os.chdir("/home/jp/blackjax-devs/tuningfork"); sys.path.insert(0,"/home/jp/blackjax-devs/tuningfork")
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.model._registry import MODELS as _M
import blackjax

init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["horseshoe"])
flat0, unravel = ravel_pytree(init_dict)
d = flat0.size
ld = lambda xf: ld_raw(unravel(xf))           # logdensity (to maximize)
neglog = lambda xf: -ld(xf)
grad_neg = jax.jit(jax.grad(neglog))
TAU_IDX = 2

# Hvp + power iteration for lambda_max of Hessian(neglog)
@jax.jit
def hvp(x, v):
    return jax.jvp(grad_neg, (x,), (v,))[1]
def lam_max(x, iters=15, key=0):
    v = jax.random.normal(jax.random.key(key), (d,)); v = v/jnp.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = hvp(x, v); lam = jnp.linalg.norm(w); v = w/(lam+1e-30)
    return float(lam)
def proxy(x, eps, key=0):
    # cheap gradient-difference directional curvature at warmup step scale eps
    p = jax.random.normal(jax.random.key(key+999), (d,)); p = p/jnp.linalg.norm(p)
    g0 = grad_neg(x); g1 = grad_neg(x + eps*p)
    return float(jnp.linalg.norm(g1-g0)/eps)

print(f"horseshoe d={d}")
# quick NUTS reference cloud (short) to get realistic positions + a tuned step
t0=time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    wa = blackjax.window_adaptation(blackjax.nuts, ld, progress_bar=False)
    (state, params), _ = wa.run(jax.random.key(1), flat0, num_steps=300)
print(f"NUTS warmup(300) done in {time.time()-t0:.1f}s | step_size={float(params['step_size']):.4g}")
eps = float(params["step_size"])
nuts = blackjax.nuts(ld, **params)
t0=time.time()
_, (pos, info) = blackjax.util.run_inference_algorithm(
    jax.random.key(2), nuts, 200, initial_state=nuts.init(state.position),
    transform=lambda s,i:(s.position, i.is_divergent), progress_bar=False)
pos=np.asarray(pos)
print(f"NUTS sample(200) done in {time.time()-t0:.1f}s | tau range [{pos[:,TAU_IDX].min():.2f},{pos[:,TAU_IDX].max():.2f}] | div={int(np.asarray(info).sum())}")
# test curvature machinery at a few positions
t0=time.time()
for i in [0, 50, 100, 150, 199]:
    x=jnp.asarray(pos[i]); lm=lam_max(x); px=proxy(x,eps)
    print(f"  draw{i}: tau={pos[i,TAU_IDX]:+.2f} lam_max={lm:.3e} ell=1/sqrt={1/np.sqrt(lm):.3e} proxy={px:.3e}")
print(f"5 curvature evals in {time.time()-t0:.1f}s")
print("SMOKE_OK")
