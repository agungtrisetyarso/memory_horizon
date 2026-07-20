import numpy as np, json
from experiments2 import (env_switching, env_drift, dissipative_mw, fixed_share,
    adahedge, discounted_hedge, dynamic_mirror_descent, strongly_adaptive,
    parameter_free, dynamic_regret, clip01)
import itertools

def tune_multiseed(factory, grids, envfn, seeds):
    """Grid-search hyperparams; report mean+/-std over seeds at the grid-best mean."""
    keys=list(grids.keys()); combos=list(itertools.product(*[grids[k] for k in keys]))
    best_mean=np.inf; best=None
    for combo in combos:
        h=dict(zip(keys,combo)); rs=[]
        for s in seeds:
            clean,obs,best_e=envfn(s)
            rs.append(dynamic_regret(factory(obs,**h),clean,best_e))
        m=np.mean(rs)
        if m<best_mean: best_mean=m; best=(m,np.std(rs),h)
    return best

def no_tune_multiseed(factory, envfn, seeds):
    rs=[]
    for s in seeds:
        clean,obs,best_e=envfn(s); rs.append(dynamic_regret(factory(obs),clean,best_e))
    return (np.mean(rs),np.std(rs),{})

T,d=2000,10
etas=[1,2,4,8]
seeds=list(range(15))

def make_env(kind,K,sigma):
    def f(seed):
        rng=np.random.default_rng(1000+seed)
        return env_switching(T,d,K,sigma,rng) if kind=="switching" else env_drift(T,d,sigma,rng)
    return f

results={}
for label,kind,K,sigma in [("Switching (P_T~40)","switching",20,0.5),
                           ("Drifting (sinusoidal)","drift",0,0.5),
                           ("Switching high-noise","switching",20,0.9)]:
    envfn=make_env(kind,K,sigma)
    tbl={}
    tbl["Dissipative MW (ours)"]=tune_multiseed(dissipative_mw,{"eta":etas,"gamma0":[0.02,0.04,0.08,0.15,0.3]},envfn,seeds)
    tbl["Discounted Hedge"]=tune_multiseed(discounted_hedge,{"eta":etas,"gamma":[0.02,0.04,0.08,0.15,0.3]},envfn,seeds)
    tbl["Fixed-Share"]=tune_multiseed(fixed_share,{"eta":etas,"alpha":[0.005,0.01,0.02,0.05,0.1]},envfn,seeds)
    tbl["Dynamic Mirror Descent"]=tune_multiseed(dynamic_mirror_descent,{"eta":etas,"alpha":[0.005,0.01,0.02,0.05,0.1]},envfn,seeds)
    tbl["Strongly Adaptive"]=tune_multiseed(strongly_adaptive,{"eta":etas},envfn,seeds)
    tbl["AdaHedge"]=no_tune_multiseed(adahedge,envfn,seeds)
    tbl["Parameter-Free"]=no_tune_multiseed(parameter_free,envfn,seeds)
    results[label]=tbl
    print(f"\n=== {label} (T={T}, d={d}, {len(seeds)} seeds) ===")
    for k,(m,s,h) in sorted(tbl.items(),key=lambda x:x[1][0]):
        print(f"  {k:26s} {m:7.1f} ± {s:5.1f}   {h}")

json.dump({k:{m:[v[0],v[1],v[2]] for m,v in t.items()} for k,t in results.items()},
          open("results_table.json","w"),indent=2)
print("\nsaved results_table.json")
