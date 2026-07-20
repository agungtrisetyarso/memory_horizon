import numpy as np

def clip01(x): return np.clip(x,0.0,1.0)

# ---------------- environments ----------------
def env_switching(T,d,K,sigma,rng):
    seg=max(1,T//K); clean=np.full((T,d),0.6); best=np.zeros(T,dtype=int)
    for t in range(T):
        b=(t//seg)%d; best[t]=b; clean[t,b]=0.2
    return clean,clip01(clean+sigma*rng.standard_normal((T,d))),best

def env_drift(T,d,sigma,rng):
    clean=np.zeros((T,d)); best=np.zeros(T,dtype=int)
    phase=np.linspace(0,2*np.pi,d,endpoint=False)
    for t in range(T):
        prof=0.5+0.3*np.cos(2*np.pi*t/(T/4.0)+phase)
        clean[t]=prof; best[t]=int(np.argmin(prof))
    return clean,clip01(clean+sigma*rng.standard_normal((T,d))),best

# ---------------- our method ----------------
def dissipative_mw(obs,eta,gamma0):
    T,d=obs.shape; hatE=np.zeros(d); p=np.ones(d)/d
    a=1-np.exp(-gamma0); relax=a; P=np.zeros((T,d))
    for t in range(T):
        hatE=(1-a)*hatE+a*obs[t]
        w=np.exp(-eta*hatE/max(a,1e-9)); w/=w.sum()
        p=p+relax*(w-p); p=clip01(p); s=p.sum(); p=p/s if s>0 else np.ones(d)/d
        P[t]=p
    return P

# ---------------- baselines ----------------
def fixed_share(obs,eta,alpha):
    T,d=obs.shape; w=np.ones(d)/d; P=np.zeros((T,d))
    for t in range(T):
        P[t]=w/w.sum()
        w=w*np.exp(-eta*obs[t]); w/=w.sum()
        w=(1-alpha)*w+alpha/d; w/=w.sum()
    return P

def adahedge(obs):
    # Correct AdaHedge (de Rooij et al. 2014), cumulative mixability gap.
    T,d=obs.shape; L=np.zeros(d); delta=0.0; P=np.zeros((T,d))
    for t in range(T):
        if delta<=0:
            w=np.ones(d)/d; eta=np.inf
        else:
            eta=np.log(d)/delta
            m=L.min(); u=np.exp(-eta*(L-m)); w=u/u.sum()
        P[t]=w; l=obs[t]; hl=w@l
        if not np.isfinite(eta):
            mix=hl-l.min()  # eta->inf : Hedge picks min
        else:
            m=(w*np.exp(-eta*l)).sum()
            mix=hl+ (1.0/eta)*np.log(m)   # per-round mixability gap >=0
        delta+=max(0.0,mix); L+=l
    return P

def discounted_hedge(obs,eta,gamma):
    T,d=obs.shape; L=np.zeros(d); P=np.zeros((T,d)); g=1-np.exp(-gamma)
    for t in range(T):
        w=np.exp(-eta*L); w/=w.sum(); P[t]=w
        L=(1-g)*L+g*obs[t]/max(g,1e-9)
    return P

def dynamic_mirror_descent(obs,eta,alpha):
    T,d=obs.shape; theta=np.zeros(d); P=np.zeros((T,d))
    for t in range(T):
        w=np.exp(theta-theta.max()); w/=w.sum(); P[t]=w
        theta=theta-eta*obs[t]; theta=(1-alpha)*theta
    return P

def strongly_adaptive(obs,eta):
    T,d=obs.shape; P=np.zeros((T,d))
    starts=[0]; s=1
    while s<T: starts.append(s); s*=2
    bw={s:np.ones(d)/d for s in starts}; meta=np.ones(len(starts))
    for t in range(T):
        active=[i for i,s in enumerate(starts) if t>=s]
        mv=meta[active]/meta[active].sum()
        mix=np.zeros(d)
        for wgt,i in zip(mv,active): mix+=wgt*bw[starts[i]]
        mix/=mix.sum(); P[t]=mix; l=obs[t]
        for i in active:
            s=starts[i]; loss=bw[s]@l
            bw[s]=bw[s]*np.exp(-eta*l); bw[s]/=bw[s].sum()
            meta[i]*=np.exp(-eta*loss)
        meta/=meta.sum()
    return P

def parameter_free(obs):
    # Simplex coin-betting (parameter-free). Stable normalized wealth.
    T,d=obs.shape; P=np.zeros((T,d)); theta=np.zeros(d); G=np.ones(d)
    for t in range(T):
        w=np.exp(theta-theta.max()); w/=w.sum(); P[t]=w
        g=(w@obs[t])-obs[t]           # gradient of loss wrt weights
        theta-=g/np.sqrt(G); G+=g*g    # adagrad-style parameter-free-ish
    return P

def dynamic_regret(P,clean,best):
    T=P.shape[0]; return float(((P*clean).sum(1)-clean[np.arange(T),best]).sum())

# ---------------- fair tuning: grid-search each method per environment ----------------
def tune(method_factory, grids, obs, clean, best):
    """Return best dynamic regret over the hyperparameter grid."""
    import itertools
    best_r=np.inf; best_h=None
    keys=list(grids.keys())
    for combo in itertools.product(*[grids[k] for k in keys]):
        h=dict(zip(keys,combo))
        P=method_factory(obs,**h)
        r=dynamic_regret(P,clean,best)
        if r<best_r: best_r=r; best_h=h
    return best_r,best_h

if __name__=="__main__":
    rng=np.random.default_rng(1)
    T,d=2000,10
    etas=[0.5,1,2,4,8]
    for envname,envfn in [("switching",lambda: env_switching(T,d,20,0.5,rng)),
                          ("drift",lambda: env_drift(T,d,0.5,rng))]:
        clean,obs,best=envfn()
        rows={}
        rows["Dissipative MW (ours)"]=tune(dissipative_mw,{"eta":etas,"gamma0":[0.02,0.04,0.08,0.15,0.3]},obs,clean,best)
        rows["Fixed-Share"]=tune(fixed_share,{"eta":etas,"alpha":[0.005,0.01,0.02,0.05,0.1]},obs,clean,best)
        rows["AdaHedge"]=(dynamic_regret(adahedge(obs),clean,best),{})
        rows["Discounted Hedge"]=tune(discounted_hedge,{"eta":etas,"gamma":[0.02,0.04,0.08,0.15,0.3]},obs,clean,best)
        rows["Dynamic Mirror Descent"]=tune(dynamic_mirror_descent,{"eta":etas,"alpha":[0.005,0.01,0.02,0.05,0.1]},obs,clean,best)
        rows["Strongly Adaptive"]=tune(strongly_adaptive,{"eta":etas},obs,clean,best)
        rows["Parameter-Free"]=(dynamic_regret(parameter_free(obs),clean,best),{})
        print(f"\n=== {envname} (T={T}, d={d}, K=20, sigma=0.5) ===")
        for k,(r,h) in sorted(rows.items(),key=lambda x:x[1][0]):
            print(f"  {k:26s} {r:8.1f}   {h}")
