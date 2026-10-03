import numpy as np, json, math
from scipy.special import exp1
def kalman_gain(q2, s2=1.0):
    Pp = (q2 + math.sqrt(q2**2 + 4*q2*s2))/2; return Pp/(Pp+s2)
def dt_mse(a, q2, s2=1.0): return ((1-a)**2*q2 + a**2*s2)/(a*(2-a))
h=0.01; sig=1.0; T=120000
Svals=[0.1,0.3,1,3,10,30,100,300,1000,3000,10000]
out=[]
for S in Svals:
    D=math.sqrt(S*h)*sig
    rng=np.random.default_rng(int(S*10)+3)
    jump=rng.random(T)<h; J=np.where(jump,D*rng.standard_normal(T),0.0); M=np.cumsum(J)
    X=M+sig*rng.standard_normal(T); burn=2000
    # linear optimum
    a=kalman_gain(h*D*D); hh=0.0; el=0.0
    # GPB1 (Nassar-style reduced Bayesian)
    mu=0.0; v=D*D; eg=0.0
    # run-length mixture
    Rmax=80; w=np.array([1.0]); mus=np.array([0.0]); vs=np.array([D*D]); er=0.0
    # genie
    nsince=0; eg2=0.0
    for t in range(T):
        x=X[t]
        hh+=a*(x-hh)
        # GPB1
        comps_m=np.array([mu,mu]); comps_v=np.array([v,v+D*D]); pri=np.array([1-h,h])
        lik=np.exp(-0.5*(x-comps_m)**2/(comps_v+sig**2))/np.sqrt(comps_v+sig**2)
        post=pri*lik; post/=post.sum()
        K=comps_v/(comps_v+sig**2); pm=comps_m+K*(x-comps_m); pv=(1-K)*comps_v
        mu=float(post@pm); v=float(post@(pv+pm**2)-mu**2)
        # run-length mixture
        cm=float(w@mus); cv=float(w@(vs+mus**2)-cm**2)
        w=np.concatenate([[h],(1-h)*w]); mus=np.concatenate([[cm],mus]); vs=np.concatenate([[cv+D*D],vs])
        lik=np.exp(-0.5*(x-mus)**2/(vs+sig**2))/np.sqrt(vs+sig**2)
        w=w*lik; w/=w.sum(); Kk=vs/(vs+sig**2); mus=mus+Kk*(x-mus); vs=(1-Kk)*vs
        if len(w)>Rmax:
            keep=np.argsort(w)[-Rmax:]; w=w[keep]; mus=mus[keep]; vs=vs[keep]; w/=w.sum()
        est=float(w@mus)
        # genie: knows jump times and previous level
        nsince = 1 if jump[t] else nsince+1
        gv=1.0/(1.0/(D*D)+nsince/sig**2) if t>0 else 0
        if t>=burn:
            el+=(M[t]-hh)**2; eg+=(M[t]-mu)**2; er+=(M[t]-est)**2; eg2+=gv
    n=T-burn
    rec=dict(S=S,Delta=D,lin_sim=el/n,lin_theory=dt_mse(a,h*D*D),gpb1=eg/n,runlength=er/n,genie_sim=eg2/n,
             genie_ct=h*sig**2*math.exp(1/S)*exp1(1/S),lin_ct=h*sig**2*math.sqrt(S))
    out.append(rec); print({k:(round(v,5) if isinstance(v,float) else v) for k,v in rec.items()},flush=True)
json.dump(dict(E4=out,h=h,sigma=sig,T=T),open('results_part3.json','w'),indent=1)
