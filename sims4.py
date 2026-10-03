import numpy as np, json, math
def kalman_gain(q2, s2=1.0):
    Pp = (q2 + math.sqrt(q2**2 + 4*q2*s2))/2; return Pp/(Pp+s2)
R={}
# ---------- E5: volatility-switching random walk; bank of leaky integrators ----------
T=200000; blk=5000; qs=[0.01,0.3]; sig=1.0
rng=np.random.default_rng(77)
qt=np.array([qs[(t//blk)%2] for t in range(T)])
M=np.cumsum(qt*rng.standard_normal(T)); X=M+sig*rng.standard_normal(T)
def run_bank(taus, beta=0.5, tauL=50.0, X=X, M=M):
    a=1/np.array(taus); h=np.zeros_like(a); L=np.zeros_like(a); gL=1/tauL; err=0.0; burn=blk
    for t in range(T):
        x=X[t]
        L = (1-gL)*L + gL*(x-h)**2        # leaky trace of each unit's one-step prediction error
        h += a*(x-h)
        w=np.exp(-beta*(L-L.min())/ (1e-12+1)); w/=w.sum()
        est=w@h
        if t>=burn: err+=(M[t]-est)**2
    return err/(T-burn)
def run_single(a, X=X, M=M):
    h=0.0; err=0.0; burn=blk
    for t in range(T):
        h+=a*(X[t]-h)
        if t>=burn: err+=(M[t]-h)**2
    return err/(T-burn)
def run_oracle(X=X,M=M):
    h=0.0; err=0.0; burn=blk
    for t in range(T):
        a=kalman_gain(qt[t]**2); h+=a*(X[t]-h)
        if t>=burn: err+=(M[t]-h)**2
    return err/(T-burn)
oracle=run_oracle()
grid=np.geomspace(0.005,0.8,40)
singles=[run_single(a) for a in grid]; best_single=min(singles); a_bs=grid[int(np.argmin(singles))]
bank8=[2**k for k in range(1,9)]  # 2..256, ratio 2
bank4=[2,8,32,128]               # ratio 4
res_bank8=run_bank(bank8); res_bank4=run_bank(bank4)
sens={f"beta{b}_tauL{tl}":run_bank(bank8,beta=b,tauL=tl) for b in [0.25,1.0] for tl in [20.0,100.0]}
taus_opt=[1/kalman_gain(q**2) for q in qs]
R['E5']=dict(oracle=oracle,best_single=best_single,best_single_tau=1/a_bs,bank8=res_bank8,bank4=res_bank4,
   sensitivity=sens,taus_opt=taus_opt,q=qs,block=blk,T=T)
print('E5',R['E5'],flush=True)
# ---------- E6: learnable time constant by a local two-trace rule ----------
def learn(q2, a0, kappa=0.02, T=300000, seed=0, qseq=None):
    r=np.random.default_rng(seed)
    if qseq is None: J=math.sqrt(q2)*r.standard_normal(T)
    else: J=qseq*r.standard_normal(T)
    Mx=np.cumsum(J); Xx=Mx+r.standard_normal(T)
    th=math.log(a0); h=0.0; s=0.0; G=1e-8; traj=np.empty(T); err=0.0
    for t in range(T-1):
        a=math.exp(th)
        hprev=h; h=hprev+a*(Xx[t]-hprev)
        s=(1-a)*s+(Xx[t]-hprev)            # sensitivity trace dh/da (a second leaky trace)
        e=Xx[t+1]-h                          # observable one-step prediction error
        g=-2*e*s*a                           # d e^2 / d log a
        G=0.999*G+0.001*g*g                  # running gradient power (normalization)
        th-=kappa*g/math.sqrt(G)
        th=min(max(th,math.log(1e-4)),math.log(0.95))
        traj[t]=1/math.exp(th)
        err+=(Mx[t]-h)**2
    traj[-1]=traj[-2]
    return traj, err/T
E6=[]
for q2 in [1e-4,1e-3,1e-2,1e-1]:
    for a0 in [0.5,0.002]:
        traj,_=learn(q2,a0,seed=int(-math.log10(q2))*10+(a0<0.1))
        fin=traj[-60000:]
        E6.append(dict(q2=q2,tau0=1/a0,tau_star=1/kalman_gain(q2),tau_learned_geomean=float(np.exp(np.mean(np.log(fin)))),
                       tau_learned_p10=float(np.percentile(fin,10)),tau_learned_p90=float(np.percentile(fin,90)),
                       traj=traj[::1000].tolist()))
        print(E6[-1]['q2'],E6[-1]['tau0'],E6[-1]['tau_star'],E6[-1]['tau_learned_geomean'],E6[-1]['tau_learned_p10'],E6[-1]['tau_learned_p90'],flush=True)
# learnable single integrator in the switching environment
_,err_learn=learn(None,0.1,qseq=qt,seed=77,T=T)
R['E6']=E6; R['E6_switching_learned_mse']=err_learn
print('learned in switching env',err_learn)
json.dump(R,open('results_part4.json','w'),indent=1)
