"""Figure 1 v2: identity with between-seed standard errors, plus a finite-T,
non-stationary boundary-term check. Vectorized over seeds and gammas."""
import numpy as np, json, math
R={}
gam=np.geomspace(0.03,3.0,13); NS=64; T=4000.0; dt=0.005; N=int(T/dt); burn=N//10
cfg=[('brownian',0.1,1.0),('brownian',0.3,0.5),('jump',0.1,1.0),('jump',0.3,0.5)]
lam=0.05
E1={}
for ci,(target,q,sig) in enumerate(cfg):
    r=np.random.default_rng(5000+ci)
    m=np.zeros((NS,1)); h=np.zeros((NS,len(gam))); acc=np.zeros((NS,len(gam)))
    for k in range(N):
        if target=='brownian': m=m+q*math.sqrt(dt)*r.standard_normal((NS,1))
        else:
            jmp=r.random((NS,1))<lam*dt; m=m+jmp*(q/math.sqrt(lam))*r.standard_normal((NS,1))
        dy=m*dt+sig*math.sqrt(dt)*r.standard_normal((NS,1))
        h=h+gam*(dy-h*dt)
        if k>=burn: acc+= (m-h)**2
    per=acc/(N-burn)                   # per-seed time-averaged MSE
    mean=per.mean(0); se=per.std(0,ddof=1)/math.sqrt(NS)
    th=q**2/(2*gam)+gam*sig**2/2
    a=gam*dt   # Euler scheme == discrete delta rule with a=gamma*dt, step volatility q^2 dt, noise sigma^2/dt
    th_euler=((1-a)**2*q*q*dt+a*a*sig*sig/dt)/(a*(2-a))
    z=(mean-th)/se; z_e=(mean-th_euler)/se
    E1[f"{target}_q{q}_s{sig}"]=dict(gamma=gam.tolist(),mean=mean.tolist(),se=se.tolist(),theory=th.tolist(),
        rel_err=((mean-th)/th).tolist(),z=z.tolist(),theory_euler=th_euler.tolist(),z_euler=z_e.tolist(),
        euler_vs_ct=((th_euler-th)/th).tolist(),max_abs_z_euler=float(np.max(abs(z_e))),max_abs_rel_err=float(np.max(abs(mean-th)/th)),max_abs_z=float(np.max(abs(z))))
    print(target,q,sig,'max rel',round(float(np.max(abs(mean-th)/th)),4),'max|z|',round(float(np.max(abs(z))),2),'max|z_euler|',round(float(np.max(abs(z_e))),2),flush=True)
R['E1v2']=dict(configs=E1,seeds=NS,T=T,dt=dt,burn_frac=0.1)
# ---- boundary-term check: start far from stationarity, short horizons, no burn-in
P=20000; dt=0.002; q=0.3; sig=0.5; g=0.5; e0=2.0
r=np.random.default_rng(9)
Tgrid=np.array([0.5,1,2,4,8,16,32])
m=np.zeros(P); h=np.full(P,-e0)        # e(0)=m-h=e0 deterministic
cum=np.zeros(P); out=[]; t=0.0; idx=0; nsteps=int(Tgrid[-1]/dt)
for k in range(1,nsteps+1):
    e_prev=m-h
    m=m+q*math.sqrt(dt)*r.standard_normal(P)
    dy=m*dt+sig*math.sqrt(dt)*r.standard_normal(P)
    h=h+g*(dy-h*dt)
    e=m-h
    cum+=0.5*(e_prev**2+e**2)*dt           # trapezoid
    if idx<len(Tgrid) and abs(k*dt-Tgrid[idx])<dt/2:
        Tt=Tgrid[idx]; lhs=cum.mean(); lhs_se=cum.std(ddof=1)/math.sqrt(P)
        eT2=(e**2).mean(); eT2_se=(e**2).std(ddof=1)/math.sqrt(P)
        full=(q*q*Tt+g*g*sig*sig*Tt+e0**2-eT2)/(2*g)
        stat=(q*q/(2*g)+g*sig*sig/2)*Tt
        out.append(dict(T=float(Tt),lhs=float(lhs),lhs_se=float(lhs_se),rhs_full=float(full),
                        rhs_stationary=float(stat),boundary=float((e0**2-eT2)/(2*g)),eT2=float(eT2)))
        print(out[-1],flush=True); idx+=1
R['boundary']=dict(paths=P,dt=dt,q=q,sigma=sig,gamma=g,e0=e0,rows=out)
json.dump(R,open('results_fig1v2.json','w'),indent=1)
