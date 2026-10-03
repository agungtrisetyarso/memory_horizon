import torch, torch.nn as nn, torch.nn.functional as F, math, time
torch.set_num_threads(1)
@torch.jit.script
def seli_scan(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, K: int):
    B=X.shape[0]; L=X.shape[1]
    mh=torch.zeros(B,dtype=X.dtype); H=torch.zeros(B,2*K,dtype=X.dtype)
    outs=[]; gains=[]
    for t in range(L):
        nu=X[:,t]-mh
        H=torch.cat([H[:,2:],nu.unsqueeze(1),nu.abs().unsqueeze(1)],1)
        v=F.silu(H@W.t()+b)
        a=1-torch.exp(-F.softplus(v@w2+b2))
        mh=mh+a*nu; outs.append(mh); gains.append(a)
    return torch.stack(outs,1), torch.stack(gains,1)
class SELIJ(nn.Module):
    def __init__(s,E=8,K=4):
        super().__init__(); s.K=K; s.W=nn.Linear(2*K,E); s.w2=nn.Parameter(torch.zeros(E)); s.b2=nn.Parameter(torch.tensor(math.log(math.expm1(0.05))))
    def forward(s,X,return_gain=False):
        P,G=seli_scan(X,s.W.weight,s.W.bias,s.w2,s.b2,s.K); return (P,G) if return_gain else P
if __name__=='__main__':
    X=torch.randn(64,800).cumsum(1)*0.1+torch.randn(64,800)
    m=SELIJ(); opt=torch.optim.Adam(m.parameters())
    for i in range(4):
        t=time.time(); P=m(X); l=((P-X)**2).mean(); opt.zero_grad(); l.backward(); opt.step(); print(i,time.time()-t)
