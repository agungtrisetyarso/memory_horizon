import numpy as np, json, math
from scipy.special import exp1
def kalman_gain(q2, s2=1.0):
    Pp = (q2 + math.sqrt(q2**2 + 4*q2*s2))/2; return Pp/(Pp+s2)
def dt_mse(a, q2, s2=1.0): return ((1-a)**2*q2 + a**2*s2)/(a*(2-a))
R={}
# ---------------- E3: quadratic variation vs path length -------------
def ema_grid(J, X, agrid):
    a=agrid; h=np.zeros_like(a); m=0.0; acc=np.zeros_like(a); T=len(J); burn=T//20
    M=np.cumsum(J)
    for t in range(T):
        h += a*(M[t]+X[t]-h)
        if t>=burn: acc += (M[t]-h)**2
    return acc/(T-burn)
agrid=np.geomspace(0.003,0.9,90)
rng=np.random.default_rng(31)
envs=[]; T=150000
kinds=['fixed','exponential','pareto','brownian']
for i in range(24):
    kind=kinds[i%4]
    J=np.zeros(T)
    if kind=='brownian':
        q=10**rng.uniform(-2.2,-0.8); J=q*rng.standard_normal(T)
    else:
        K=int(10**rng.uniform(1.3,3.0)); idx=rng.choice(T,K,replace=False)
        scale=10**rng.uniform(-0.3,0.6)
        if kind=='fixed': sz=np.full(K,scale)
        elif kind=='exponential': sz=rng.exponential(scale,K)
        else: sz=scale*(rng.pareto(2.5,K)+1)*0.6
        J[idx]=sz*rng.choice([-1,1],K)
    X=rng.standard_normal(T)
    mse=ema_grid(J,X,agrid)
    k=np.argmin(mse); 
    if 0<k<len(agrid)-1:  # parabola in log a
        y=np.log(agrid[k-1:k+2]); z=mse[k-1:k+2]; c=np.polyfit(y,z,2); a_star=math.exp(-c[1]/(2*c[0]))
    else: a_star=agrid[k]
    Q=float(np.sum(J**2)); P=float(np.sum(np.abs(J)))
    envs.append(dict(kind=kind,Q=Q,P=P,a_meas=a_star,a_predQ=kalman_gain(Q/T),mse_min=float(mse.min()),
                     mse_at_predQ=float(np.interp(math.log(kalman_gain(Q/T)),np.log(agrid),mse))))
lm=np.log([1/e['a_meas'] for e in envs]); lq=np.log([1/e['a_predQ'] for e in envs])
r2Q=1-np.sum((lm-lq)**2)/np.sum((lm-lm.mean())**2)
# best path-length law tau = c*sqrt(T/P) fit c (finite-P envs only)
fin=[e for e in envs if e['kind']!='brownian']
lmf=np.log([1/e['a_meas'] for e in fin]); lp=0.5*np.log([T/e['P'] for e in fin])
c=np.mean(lmf-lp); r2P=1-np.sum((lmf-lp-c)**2)/np.sum((lmf-lmf.mean())**2)
lqf=np.log([1/e['a_predQ'] for e in fin]); r2Qf=1-np.sum((lmf-lqf)**2)/np.sum((lmf-lmf.mean())**2)
# also free-slope regression on log P
sl,ic=np.polyfit(np.log([e['P'] for e in fin]),lmf,1)
r2Pfree=1-np.sum((lmf-(sl*np.log([e['P'] for e in fin])+ic))**2)/np.sum((lmf-lmf.mean())**2)
excess=[e['mse_at_predQ']/e['mse_min'] for e in envs]
# matched-P pair
Tm=150000; rng2=np.random.default_rng(5)
def pair(K,size):
    J=np.zeros(Tm); idx=rng2.choice(Tm,K,replace=False); J[idx]=size*rng2.choice([-1,1],K); return J
JA=pair(400,0.5); JB=pair(100,2.0); X=rng2.standard_normal(Tm)
mA=ema_grid(JA,X,agrid); mB=ema_grid(JB,X,agrid)
R['E3']=dict(envs=envs,r2_Q_noparam_all=float(r2Q),r2_Q_noparam_finiteP=float(r2Qf),r2_P_fitted_const=float(r2P),
   P_free_slope=float(sl),r2_P_free_slope=float(r2Pfree),excess_at_predQ_max=float(max(excess)),excess_at_predQ_median=float(np.median(excess)),
   pair=dict(PA=float(np.abs(JA).sum()),PB=float(np.abs(JB).sum()),QA=float((JA**2).sum()),QB=float((JB**2).sum()),
       tauA=float(1/agrid[np.argmin(mA)]),tauB=float(1/agrid[np.argmin(mB)]),
       tauA_pred=1/kalman_gain((JA**2).sum()/Tm),tauB_pred=1/kalman_gain((JB**2).sum()/Tm),
       curveA=mA.tolist(),curveB=mB.tolist()),agrid=agrid.tolist())
print('E3 r2Q',r2Q,'r2Qf',r2Qf,'r2P(fixed slope)',r2P,'free slope',sl,r2Pfree,'excess max',max(excess),R['E3']['pair']['tauA'],R['E3']['pair']['tauB'],R['E3']['pair']['tauA_pred'],R['E3']['pair']['tauB_pred'])
json.dump(R,open('results_part2.json','w'),indent=1)
