"""Numerical checks of the discrete-time results (Theorem 4, Proposition 5, Lemma 2, Proposition 6)."""
import numpy as np, math, json
R={}
rng=np.random.default_rng(42)
# --- Theorem 4: exact discrete identity, time-varying q_t and deterministic gain a_t, non-stationary start
P=200000; T=400; sig=0.7; e0=1.5
t=np.arange(1,T+1); q2=0.02*(1+np.sin(t/30.0))**2; a=0.05+0.25*(t%50<25)    # time-varying volatility and gain
m=np.zeros(P); mh=np.full(P,-e0); cum=np.zeros(P)
E=[e0**2]
for k in range(T):
    J=math.sqrt(q2[k])*rng.choice([-1,1],P)*(rng.random(P)<0.5)*math.sqrt(2)   # non-Gaussian martingale increments, E J^2=q2
    m+=J; x=m+sig*rng.standard_normal(P); mh+=a[k]*(x-mh); cum+=(m-mh)**2
    E.append((1-a[k])**2*(E[-1]+q2[k])+a[k]**2*sig**2)
R['thm4_recursion_vs_mc']=dict(mc_sum=float(cum.mean()),mc_se=float(cum.std()/math.sqrt(P)),recursion_sum=float(sum(E[1:])))
# constant gain closed form
a0=0.1; Q=float(q2.sum()); m=np.zeros(P); mh=np.full(P,-e0); cum=np.zeros(P)
for k in range(T):
    J=math.sqrt(q2[k])*rng.standard_normal(P); m+=J; x=m+sig*rng.standard_normal(P); mh+=a0*(x-mh); cum+=(m-mh)**2
eT2=float(((m-mh)**2).mean()); b=1-a0
closed=(b*b*Q+a0*a0*sig**2*T+b*b*(e0**2-eT2))/(a0*(2-a0))
R['thm4_closed_form']=dict(mc_sum=float(cum.mean()),mc_se=float(cum.std()/math.sqrt(P)),closed=closed)
# --- Lemma 2: observable one-step loss = latent error + q2 + sigma2, for a NONLINEAR estimator (median of last 5 obs)
P2=100000; L=300; q=0.1; h=0.02; D=1.0
on=rng.random((P2,L))<h; J=on*D*rng.standard_normal((P2,L)); M=np.cumsum(J,1); X=M+sig*rng.standard_normal((P2,L))
t0=200; est=np.median(X[:,t0-4:t0+1],axis=1)
lhs=np.mean((X[:,t0+1]-est)**2); rhs=np.mean((M[:,t0]-est)**2)+h*D*D+sig**2
R['lemma2_nonlinear']=dict(obs_loss=float(lhs),latent_plus=float(rhs))
# --- Proposition 5: discrete genie floor closed form vs simulation
def genie_series(S,h,s2=1.0,K=200000):
    D2=S*h*s2; k=np.arange(1,K+1); return float(np.sum(h*(1-h)**(k-1)*s2*D2/(s2+k*D2)))
E4=json.load(open('results.json'))['E4']
R['prop5_genie']=[dict(S=e['S'],series=genie_series(e['S'],0.01),sim=e['genie_sim']) for e in E4]
R['prop5_large_S_limit']=dict(formula=0.01*(-math.log(0.01))/(1-0.01),series_S1e6=genie_series(1e6,0.01))
# --- Proposition 6: S identification, S_kappa=(1-h)^2 S for Gaussian jumps
R['prop6']=[]
for S,h in [(10,0.01),(100,0.01),(30,0.05)]:
    D=math.sqrt(S*h); Jv=(rng.random(8_000_000)<h)*D*rng.standard_normal(8_000_000)
    q2v=np.mean(Jv**2); k4=np.mean(Jv**4)-3*q2v**2
    R['prop6'].append(dict(S=S,h=h,S_kappa_mc=float(k4**2/(9*q2v**3)),predicted=(1-h)**2*S))
json.dump(R,open('results_discrete.json','w'),indent=1); print(json.dumps(R,indent=1)[:2500])
