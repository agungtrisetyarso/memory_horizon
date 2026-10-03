import numpy as np, json, math
exec(open('sims4.py').read().split('# ---------- E6')[0].split('oracle=run_oracle()')[0])
out={}
for b in [2.0,5.0,10.0,20.0,50.0]:
    for tl in [25.0,50.0,100.0,200.0]:
        out[f"{b}_{tl}"]=run_bank([2**k for k in range(1,9)],beta=b,tauL=tl)
        print(b,tl,round(out[f"{b}_{tl}"],4),flush=True)
# per-regime best-in-bank (oracle selection) to check cosh coverage
def regime_best(taus):
    a=1/np.array(taus); h=np.zeros_like(a); acc=np.zeros((2,len(a))); n=np.zeros(2)
    for t in range(T):
        h+=a*(X[t]-h)
        if t>=blk:
            r=(t//blk)%2; acc[r]+=(M[t]-h)**2; n[r]+=1
    return acc/n[:,None]
rb=regime_best([2**k for k in range(1,9)])
def kg(q2):
    Pp=(q2+math.sqrt(q2*q2+4*q2))/2; return Pp/(Pp+1)
def dtm(a,q2): return ((1-a)**2*q2+a*a)/(a*(2-a))
star=[dtm(kg(q*q),q*q) for q in qs]
out['regime_bank_best']=[float(rb[r].min()) for r in range(2)]
out['regime_star']=star
out['regime_ratio']=[float(rb[r].min()/star[r]) for r in range(2)]
print(out['regime_bank_best'],star,out['regime_ratio'])
json.dump(out,open('results_part5.json','w'),indent=1)
