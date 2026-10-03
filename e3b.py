"""E3 (revision): 3 training seeds, per-sequence test errors, plus a Mamba-style selective SSM.
Models (innovation/normalized form, translation-equivariant):
  CG   constant gain (constant-rate delta rule)                         linear
  LRNN 16 normalized leaky channels, affine readout (LRU-style)          linear, LTI
  SEL  Mamba-style selective SSM: scalar state, ZOH with A=-1, B=C=1,
       step Delta_t = softplus(w . SiLU(causal depthwise conv4(in_proj(dx)))) -- input-only selection
  GRUG GRU(16) reads innovation (nu, |nu|), emits write rate             nonlinear, state-dependent
Usage: python e3b.py S seed"""
import sys, json, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
S=float(sys.argv[1]); SEED=int(sys.argv[2]); ITERS=600
ONLY=sys.argv[3].split(',') if len(sys.argv)>3 else None
h=0.01; sig=1.0; D=math.sqrt(S*h)
def gen(B,L,seed):
    g=torch.Generator().manual_seed(seed)
    jump=(torch.rand(B,L,generator=g)<h).float(); J=jump*D*torch.randn(B,L,generator=g)
    M=torch.cumsum(J,1); X=M+sig*torch.randn(B,L,generator=g); return X,M,jump
class CG(nn.Module):
    def __init__(s): super().__init__(); s.th=nn.Parameter(torch.logit(torch.tensor(0.05)))
    def forward(s,X):
        B,L=X.shape; mh=torch.zeros(B); out=[]; g=torch.sigmoid(s.th)
        for t in range(L): mh=mh+g*(X[:,t]-mh); out.append(mh)
        return torch.stack(out,1)
class LRNN(nn.Module):
    def __init__(s,n=16):
        super().__init__(); s.lam=nn.Parameter(torch.logit(1-torch.logspace(-0.3,-2.7,n))); s.u=nn.Parameter(torch.zeros(n))
    def forward(s,X):
        B,L=X.shape; n=s.u.numel(); lam=torch.sigmoid(s.lam); w=s.u+(1-s.u.sum())/n; e=torch.zeros(B,n); out=[]
        for t in range(L): e=lam*e+(1-lam)*X[:,t:t+1]; out.append(e@w)
        return torch.stack(out,1)
class SEL(nn.Module):
    def __init__(s,E=8,K=4):
        super().__init__(); s.inp=nn.Linear(1,E); s.conv=nn.Conv1d(E,E,K,groups=E,padding=K-1); s.dt=nn.Linear(E,1)
        nn.init.zeros_(s.dt.weight); nn.init.constant_(s.dt.bias,math.log(math.expm1(0.05)))   # Delta ~ 0.05 at init
    def deltas(s,X):
        d=torch.diff(X,dim=1,prepend=X[:,:1]); u=s.inp(d[...,None]).transpose(1,2)          # (B,E,L)
        v=F.silu(s.conv(u)[...,:X.shape[1]]).transpose(1,2)                                   # causal
        return F.softplus(s.dt(v)).squeeze(-1)                                                # (B,L), input-only
    def forward(s,X,return_gain=False):
        alt=torch.exp(-s.deltas(X)); al=alt.unbind(1); xs=X.unbind(1); B,L=X.shape; mh=torch.zeros(B); out=[]
        for t in range(L): mh=al[t]*mh+(1-al[t])*xs[t]; out.append(mh)
        P=torch.stack(out,1); return (P,1-alt) if return_gain else P
class GRUG(nn.Module):
    def __init__(s,n=16):
        super().__init__(); s.cell=nn.GRUCell(2,n); s.w=nn.Linear(n,1); nn.init.constant_(s.w.bias,float(torch.logit(torch.tensor(0.05))))
    def forward(s,X,return_gain=False):
        B,L=X.shape; mh=torch.zeros(B); hh=torch.zeros(B,s.cell.hidden_size); out=[]; gains=[]
        for t in range(L):
            nu=X[:,t]-mh; hh=s.cell(torch.stack([nu,nu.abs()],1),hh); g=torch.sigmoid(s.w(hh)).squeeze(1)
            mh=mh+g*nu; out.append(mh); gains.append(g)
        return (torch.stack(out,1),torch.stack(gains,1)) if return_gain else torch.stack(out,1)
def train(model,iters,lr,seed0,B=64,L=800,burn=300):
    opt=torch.optim.Adam(model.parameters(),lr=lr); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iters)
    for it in range(iters):
        X,M,_=gen(B,L,seed0+it); P=model(X); loss=((P[:,burn:]-M[:,burn:])**2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
def kg(q2):
    P=(q2+math.sqrt(q2*q2+4*q2))/2; return P/(P+1)
t0=time.time(); Xte,Mte,Jte=gen(16,20000,10**6+int(S*10)); burn=2000
res=dict(S=S,seed=SEED,h=h,a_star=kg(h*D*D))
with torch.no_grad():
    a=kg(h*D*D); mh=torch.zeros(16); P=[]
    for t in range(Xte.shape[1]): mh=mh+a*(Xte[:,t]-mh); P.append(mh)
    res['linear_opt_seq']=(((torch.stack(P,1)-Mte)[:,burn:])**2).mean(1).tolist()
    n=torch.zeros(16); gv=[]
    for t in range(Xte.shape[1]): n=torch.where(Jte[:,t]>0,torch.ones(16),n+1); gv.append(1/(1/(D*D)+n))
    res['genie_seq']=torch.stack(gv,1)[:,burn:].mean(1).tolist()
for name,cls,it,lr in [('CG',CG,300,0.05),('LRNN',LRNN,ITERS,0.01),('SEL',SEL,ITERS,0.005),('GRUG',GRUG,ITERS,0.005)]:
    if ONLY and name not in ONLY: continue
    torch.manual_seed(100*SEED+1); m=cls(); train(m,it,lr,seed0=SEED*100000)
    with torch.no_grad():
        if name in ('SEL','GRUG'):
            P,G=m(Xte,return_gain=True); Gb=G[:,burn:]; Jb=Jte[:,burn:].bool(); near=torch.zeros_like(Jb)
            for b,t in Jb.nonzero().tolist(): near[b,t:t+5]=True
            res[name+'_gain_after_jump']=float(Gb[near].mean()); res[name+'_gain_elsewhere']=float(Gb[~near].mean())
        else: P=m(Xte)
        res[name+'_seq']=(((P-Mte)[:,burn:])**2).mean(1).tolist()
    if name=='CG': res['CG_gain']=float(torch.sigmoid(m.th))
    print(S,SEED,name,np.mean(res[name+'_seq']),round(time.time()-t0),'s',flush=True)
json.dump(res,open(f'e3b_S{S:g}_seed{SEED}'+('_'+'-'.join(ONLY) if ONLY else '')+'.json','w'))
