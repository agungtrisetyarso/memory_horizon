"""E7: trained recurrent estimators on the jump-tracking task (Neurocomputing version).
All models are in innovation form  m_hat_t = m_hat_{t-1} + u_t, nu_t = x_t - m_hat_{t-1},
so they are translation-equivariant and can follow an unbounded random walk.
  CG   : constant learnable gain  u_t = g * nu_t                     (linear, 1 parameter)
  LRNN : diagonal linear recurrence on innovations, 16 real decays    (linear, LTI)
  GRUG : GRU(16) reads [nu_t, |nu_t|] and emits a gain g_t in (0,1)   (nonlinear, surprise-gated)
Usage: python train_nets.py S [iters]"""
import sys, json, math, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
S=float(sys.argv[1]); ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 400
h=0.01; sig=1.0; D=math.sqrt(S*h)
def gen(B,L,seed):
    g=torch.Generator().manual_seed(seed)
    jump=(torch.rand(B,L,generator=g)<h).float()
    J=jump*D*torch.randn(B,L,generator=g)
    M=torch.cumsum(J,1); X=M+sig*torch.randn(B,L,generator=g)
    return X,M,jump
class CG(nn.Module):
    def __init__(s): super().__init__(); s.th=nn.Parameter(torch.logit(torch.tensor(0.05)))
    def forward(s,X):
        B,L=X.shape; mh=torch.zeros(B); out=[]; g=torch.sigmoid(s.th)
        for t in range(L):
            mh=mh+g*(X[:,t]-mh); out.append(mh)
        return torch.stack(out,1)
class LRNN(nn.Module):
    """Linear, LTI, unit-DC-gain memory: affine combination (weights sum to 1) of
    16 normalized leaky integrators with learnable decays. No feedback => always stable."""
    def __init__(s,n=16):
        super().__init__()
        s.lam=nn.Parameter(torch.logit(1-torch.logspace(-0.3,-2.7,n)))   # time constants ~2..500
        s.u=nn.Parameter(torch.zeros(n))
    def forward(s,X):
        B,L=X.shape; n=s.u.numel(); lam=torch.sigmoid(s.lam); w=s.u+(1-s.u.sum())/n
        e=torch.zeros(B,n); out=[]
        for t in range(L):
            e=lam*e+(1-lam)*X[:,t:t+1]; out.append(e@w)
        return torch.stack(out,1)
class GRUG(nn.Module):
    def __init__(s,n=16):
        super().__init__(); s.cell=nn.GRUCell(2,n); s.w=nn.Linear(n,1)
        nn.init.constant_(s.w.bias,float(torch.logit(torch.tensor(0.05))))
    def forward(s,X,return_gain=False):
        B,L=X.shape; mh=torch.zeros(B); hh=torch.zeros(B,s.cell.hidden_size); out=[]; gains=[]
        for t in range(L):
            nu=X[:,t]-mh; hh=s.cell(torch.stack([nu,nu.abs()],1),hh)
            g=torch.sigmoid(s.w(hh)).squeeze(1); mh=mh+g*nu; out.append(mh); gains.append(g)
        return (torch.stack(out,1),torch.stack(gains,1)) if return_gain else torch.stack(out,1)
def train(model,iters,B=64,L=800,burn=300,lr=3e-3,seed0=0):
    opt=torch.optim.Adam(model.parameters(),lr=lr); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iters)
    for it in range(iters):
        X,M,_=gen(B,L,seed0+it); P=model(X); loss=((P[:,burn:]-M[:,burn:])**2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
    return model
def kalman_gain(q2,s2=1.0):
    Pp=(q2+math.sqrt(q2*q2+4*q2*s2))/2; return Pp/(Pp+s2)
t0=time.time()
Xte,Mte,Jte=gen(16,20000,10**6+int(S*10)); burn=2000
res=dict(S=S,Delta=D,h=h,sigma=sig,iters=ITERS)
with torch.no_grad():
    a=kalman_gain(h*D*D); mh=torch.zeros(16); P=[]
    for t in range(Xte.shape[1]): mh=mh+a*(Xte[:,t]-mh); P.append(mh)
    P=torch.stack(P,1); res['linear_opt']=float(((P-Mte)[:,burn:]**2).mean())
    # discrete genie on test data
    n=torch.zeros(16); gv=[]
    for t in range(Xte.shape[1]):
        n=torch.where(Jte[:,t]>0,torch.ones(16),n+1); gv.append(1.0/(1.0/(D*D)+n/sig**2))
    res['genie']=float(torch.stack(gv,1)[:,burn:].mean())
    res['a_star']=a
for name,cls,it,lr in [('CG',CG,300,0.05),('LRNN',LRNN,ITERS,0.01),('GRUG',GRUG,ITERS,0.005)]:
    torch.manual_seed(1); m=cls(); train(m,it,lr=lr)
    with torch.no_grad():
        if name=='GRUG':
            P,G=m(Xte,return_gain=True)
            # gain around jumps vs elsewhere (diagnostic of surprise gating)
            Gb=G[:,burn:]; Jb=Jte[:,burn:]
            res['GRUG_gain_median']=float(Gb.median())
            near=torch.zeros_like(Jb,dtype=torch.bool)
            idx=Jb.nonzero()
            for b,t in idx.tolist():
                near[b,t:t+5]=True
            res['GRUG_gain_after_jump_mean']=float(Gb[near].mean()) if near.any() else None
            res['GRUG_gain_elsewhere_mean']=float(Gb[~near].mean())
        else: P=m(Xte)
        res[name]=float(((P-Mte)[:,burn:]**2).mean())
    if name=='CG': res['CG_gain']=float(torch.sigmoid(m.th))
    print(name,res[name],round(time.time()-t0),'s',flush=True)
json.dump(res,open(f'res_S{S:g}.json','w'),indent=1)
print(res)
