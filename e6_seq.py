"""E6: does the pre-training statistic S (estimated from observations only) predict when a
standard gated RNN beats a standard linear recurrence on one-step forecasting?
Models (both on differenced inputs, residual output x_t + head):
  LIN : diagonal linear recurrence, 32 real decays (LRU-style), linear readout   -> linear filter of x
  GRU : 2-layer GRU(32) + linear head                                          -> nonlinear
Usage: python e6_seq.py <dataset_id>"""
import sys, json, math, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
SIG=1.0
DATASETS={
 'g_h01_S1':   dict(fam='gauss',h=0.01,S=1),
 'g_h01_S10':  dict(fam='gauss',h=0.01,S=10),
 'g_h01_S100': dict(fam='gauss',h=0.01,S=100),
 'g_h01_S1000':dict(fam='gauss',h=0.01,S=1000),
 'g_h05_S3':   dict(fam='gauss',h=0.05,S=3),
 'g_h05_S30':  dict(fam='gauss',h=0.05,S=30),
 'l_h01_S3':   dict(fam='laplace',h=0.01,S=3),
 'l_h01_S100': dict(fam='laplace',h=0.01,S=100),
 'm_h01_S30':  dict(fam='mix',h=0.01,S=30),
 'm_h01_S300': dict(fam='mix',h=0.01,S=300),
 'brown_q01':  dict(fam='brown',q=0.1),
}
def jumps(spec,rng,shape):
    f=spec['fam']
    if f=='brown': return spec['q']*rng.standard_normal(shape)
    h=spec['h']; D=math.sqrt(spec['S']*h)*SIG
    on=rng.random(shape)<h
    if f=='gauss': J=on*D*rng.standard_normal(shape)
    elif f=='laplace': J=on*(D/math.sqrt(2))*rng.laplace(0,1,shape)
    elif f=='mix': J=on*D*rng.standard_normal(shape)+math.sqrt(h)*D*rng.standard_normal(shape)  # Brownian part with equal QV
    return J
def gen(spec,B,L,seed):
    rng=np.random.default_rng(seed); J=jumps(spec,rng,(B,L)); M=np.cumsum(J,1); X=M+SIG*rng.standard_normal((B,L))
    return X,M
def true_stats(spec):
    rng=np.random.default_rng(7); J=jumps(spec,rng,(4_000_000,))
    q2=float(np.mean(J**2)); k4=float(np.mean(J**4)-3*q2**2)
    return q2,k4,max(k4,0)**2/(9*q2**3*SIG**2)
def est_S(X, lags=(1,2,4,8,16,32,64,128,256)):
    """Pre-training estimate of S from observations only (no latent, no labels).
    Variance-time and cumulant-time regressions of k-lag differences:
      Var(x_{t+k}-x_t)   = k q^2 + 2 sigma^2
      kappa4(x_{t+k}-x_t) = k kappa4(J)      (Gaussian observation noise adds no 4th cumulant)
    then S_kappa = kappa4(J)^2 / (9 q^6 sigma^2)  (= Delta^2/(h sigma^2) for Gaussian jumps)."""
    ks=np.array(lags,float); v=[]; c4=[]
    for k in lags:
        d=(X[:,k:]-X[:,:-k]).ravel(); d=d-d.mean(); vk=np.mean(d**2); v.append(vk); c4.append(np.mean(d**4)-3*vk**2)
    v=np.array(v); c4=np.array(c4)
    A=np.vstack([ks,np.ones_like(ks)]).T; q2,ic=np.linalg.lstsq(A,v,rcond=None)[0]; s2=ic/2
    k4=float(np.sum(ks*c4)/np.sum(ks*ks))
    S=max(k4,0)**2/(9*q2**3*s2) if (q2>0 and s2>0) else float('nan')
    return dict(S_hat=float(S),q2_hat=float(q2),sigma2_hat=float(s2),k4_hat=k4)
TAUS=np.logspace(0,math.log10(3000),64)
def lin_features(X):
    """Linear recurrence on differenced input d_t: 64 channels h_c = alpha_c h_c + d_t (FFT), plus d_t."""
    X=np.asarray(X,float); d=np.diff(X,axis=1,prepend=X[:,:1]); B,L=d.shape
    al=1-1/TAUS; k=np.arange(L); K=al[:,None]**k[None,:]
    n2=2*L; H=np.fft.irfft(np.fft.rfft(d[:,None,:],n2)*np.fft.rfft(K,n2)[None],n2)[...,:L]
    return np.concatenate([H,d[:,None,:]],1)            # (B,65,L)
class LINLS:
    """Linear recurrence with least-squares readout: the best linear filter in a 64-decay class."""
    def fit(s,X,burn=500,ridge=1e-6):
        F=lin_features(X); Y=X[:,1:]-X[:,:-1]                # target: next increment
        A=F[:,:,burn:-1].transpose(0,2,1).reshape(-1,F.shape[1]); y=Y[:,burn:].reshape(-1)
        G=A.T@A; s.w=np.linalg.solve(G+ridge*np.trace(G)/len(G)*np.eye(len(G)),A.T@y); return s
    def predict(s,X):
        F=lin_features(X); return np.asarray(X)+np.einsum('bcl,c->bl',F,s.w)   # prediction of x_{t+1}
class GRUres(nn.Module):
    """Gated residual on top of the linear predictor: reads (d_t, linear innovation) and adds a correction."""
    def __init__(s,n=32):
        super().__init__(); s.g=nn.GRU(2,n,num_layers=2,batch_first=True); s.o=nn.Linear(n,1)
        nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)
    def forward(s,inp): return s.o(s.g(inp)[0]).squeeze(-1)
def gru_inputs(X,Plin,sc):
    Xt=torch.tensor(X,dtype=torch.float32); P=torch.tensor(Plin,dtype=torch.float32)
    d=torch.diff(Xt,dim=1,prepend=Xt[:,:1]); r=torch.cat([torch.zeros(Xt.shape[0],1),Xt[:,1:]-P[:,:-1]],1)  # innovation of linear model
    return Xt,P,torch.stack([d/sc,r/sc],-1)
def run(name):
    spec=DATASETS[name]; t0=time.time()
    q2,k4,S_true=true_stats(spec)
    Xs,_=gen(spec,64,20000,123); est=est_S(Xs); sc=float(np.std(np.diff(Xs,axis=1)))
    est['S_hat_blocks']=[est_S(Xs[i*16:(i+1)*16])['S_hat'] for i in range(4)]
    res=dict(name=name,spec=spec,q2=q2,k4=k4,S_kappa_true=S_true,**est)
    # exact best-linear excess (prediction error of x_{t+1} minus q2+sigma2) = J_d(a*)
    Pp=(q2+math.sqrt(q2*q2+4*q2*SIG**2))/2; a=Pp/(Pp+SIG**2); res['lin_opt_excess']=((1-a)**2*q2+a*a*SIG**2)/(a*(2-a))
    Xte,Mte=gen(spec,32,4000,999); burn=800
    lin=LINLS().fit(Xs)
    Pte=lin.predict(Xte); mse=float(np.mean((Pte[:,burn:-1]-Xte[:,burn+1:])**2))
    res['LIN_mse']=mse; res['LIN_excess']=float(np.mean((Pte[:,burn:-1]-Mte[:,burn:-1])**2))   # latent error = obs. error - q2 - sigma2 in expectation
    print(name,'LIN',round(mse,5),'excess',round(res['LIN_excess'],5),'opt',round(res['lin_opt_excess'],5),round(time.time()-t0),'s',flush=True)
    torch.manual_seed(0); m=GRUres(); opt=torch.optim.Adam(m.parameters(),lr=3e-3); IT=600
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,IT)
    for it in range(IT):
        X,_=gen(spec,32,600,10_000+it); Xt,P,inp=gru_inputs(X,lin.predict(X),sc)
        out=P+sc*m(inp); loss=((out[:,150:-1]-Xt[:,151:])**2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
    with torch.no_grad():
        Xt,P,inp=gru_inputs(Xte,Pte,sc); out=P+sc*m(inp)
        mse=float(((out[:,burn:-1]-Xt[:,burn+1:])**2).mean())
    res['GRU_mse']=mse; res['GRU_excess']=float(np.mean((out.numpy()[:,burn:-1]-Mte[:,burn:-1])**2))
    print(name,'GRU',round(mse,5),'excess',round(res['GRU_excess'],5),round(time.time()-t0),'s',flush=True)
    res['gain_lin_over_gru']=res['LIN_excess']/res['GRU_excess']
    json.dump(res,open(f'e6_{name}.json','w'),indent=1); print(res)
if __name__=='__main__': run(sys.argv[1])
