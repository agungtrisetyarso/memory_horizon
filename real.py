"""E7: real forecasting data. Pre-training S_kappa on the training split, then one-step forecasting with
(i) naive last value, (ii) single EMA at Rule-1 gain from (q2_hat, sigma2_hat), (iii) LS-fitted linear recurrence,
(iv) linear recurrence + GRU residual trained on observations only (seeded). Evaluation on the test split.
Usage: python real.py <series> <seed>    series in ETTh1 ETTh2 ETTm1 ETTm2 FX (FX = pooled 8 exchange rates)"""
import sys, json, math, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'/home/claude/nc/npl/e6')
torch.set_num_threads(1)
src=open('/home/claude/nc/npl/e6/e6_seq.py').read()
ns={}; exec(src.split("if __name__")[0],ns)
est_S=ns['est_S']; LINLS=ns['LINLS']; GRUres=ns['GRUres']; gru_inputs=ns['gru_inputs']
D='/home/claude/nc/npl/real/data/'
def load(name):
    """returns list of 1-D arrays (series), split indices (tr_end, va_end, te_end), period, unit note"""
    if name.startswith('ETT'):
        import csv
        rows=list(csv.reader(open(D+name+'.csv')))[1:]; x=np.array([float(r[-1]) for r in rows])   # OT column
        per=24 if name[3]=='h' else 96; m=1 if name[3]=='h' else 4
        tr,va,te=12*30*24*m,(12+4)*30*24*m,(12+8)*30*24*m
        return [x[:te]],(tr,va,te),per
    if name.startswith('ELEC'):
        X=np.load(D+'electricity.npy'); n=len(X); tr,va=int(0.6*n),int(0.8*n)
        x=X.sum(1) if name=='ELECAGG' else X[:,int(name[4:])]
        return [x.astype(float)],(tr,va,n),24
    if name=='FX':
        X=np.loadtxt(D+'exchange_rate.txt',delimiter=','); n=len(X); tr,va=int(0.6*n),int(0.8*n)
        return [100*np.log(X[:,j]) for j in range(X.shape[1])],(tr,va,n),None
def deseason(x,tr,per):
    if per is None: return x.copy()
    ph=np.arange(len(x))%per; prof=np.array([x[:tr][ph[:tr]==k].mean() for k in range(per)]); prof-=prof.mean()
    return x-prof[ph]
def kg(q2,s2):
    P=(q2+math.sqrt(q2*q2+4*q2*s2))/2; return P/(P+s2)
def ema_pred(x,a):
    p=np.empty_like(x); m=x[0]
    for t in range(len(x)): m=m+a*(x[t]-m); p[t]=m
    return p
def block_boot(e1,e2,blk,B=5000,rng=np.random.default_rng(0)):
    """paired moving-block bootstrap of ratio of mean squared errors e1/e2 (arrays of squared errors)"""
    n=len(e1); nb=max(1,n//blk); starts=rng.integers(0,n-blk+1,(B,nb))
    idx=(starts[:,:,None]+np.arange(blk)[None,None,:]).reshape(B,-1)
    r=e1[idx].mean(1)/e2[idx].mean(1); return float(np.percentile(r,2.5)),float(np.percentile(r,97.5)),float(2*min((r<=1).mean(),(r>=1).mean()))
def main(name,seed):
    t0=time.time(); series,(tr,va,te),per=load(name)
    xs=[deseason(x,tr,per) for x in series]
    res=dict(series=name,seed=seed,split=dict(train=tr,val=va,test=te),period=per,n_series=len(xs))
    # ---- pre-training statistic on TRAIN split only
    ests=[est_S(x[None,:tr]) for x in xs]
    res['S_hat']=[e['S_hat'] for e in ests]; res['q2_hat']=[e['q2_hat'] for e in ests]; res['sigma2_hat']=[e['sigma2_hat'] for e in ests]
    res['S_hat_blocks']=[[est_S(x[None,k*tr//4:(k+1)*tr//4])['S_hat'] for k in range(4)] for x in xs]
    # ---- linear models fitted on train
    lins=[LINLS().fit(x[None,:tr],burn=300) for x in xs]
    sc=float(np.mean([np.std(np.diff(x[:tr])) for x in xs]))
    # ---- GRU residual, observation-only loss, crops from TRAIN split
    torch.manual_seed(seed); rng=np.random.default_rng(seed); m=GRUres()
    opt=torch.optim.Adam(m.parameters(),lr=3e-3); IT=600; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,IT); L=600
    Ptr=[lins[j].predict(xs[j][None,:tr])[0] for j in range(len(xs))]
    # validation: one-step loss on the VALIDATION split (model run causally over train+val)
    Pfull=[lins[j].predict(xs[j][None,:va])[0] for j in range(len(xs))]
    def val_loss():
        tot=0.0; n=0
        with torch.no_grad():
            for j in range(len(xs)):
                Xt,Pt,inp=gru_inputs(xs[j][None,:va],Pfull[j][None,:],sc); out=(Pt+sc*m(inp)).numpy()[0]
                e=(out[tr:va-1]-xs[j][tr+1:va])**2; tot+=e.sum(); n+=len(e)
        return tot/n
    import copy
    best=(val_loss(),0,copy.deepcopy(m.state_dict())); hist=[(0,best[0])]
    for it in range(1,IT+1):
        J=rng.integers(0,len(xs),32); st=rng.integers(0,tr-L,32)
        X=np.stack([xs[j][s:s+L] for j,s in zip(J,st)]); P=np.stack([Ptr[j][s:s+L] for j,s in zip(J,st)])
        Xt,Pt,inp=gru_inputs(X,P,sc); out=Pt+sc*m(inp); loss=((out[:,150:-1]-Xt[:,151:])**2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
        if it%50==0:
            v=val_loss(); hist.append((it,v))
            if v<best[0]: best=(v,it,copy.deepcopy(m.state_dict()))
    m.load_state_dict(best[2]); res['best_iter']=best[1]; res['val_hist']=hist
    # ---- evaluate on TEST split (models run causally over the whole series)
    per_series=[]
    for j,x in enumerate(xs):
        P_lin=lins[j].predict(x[None,:])[0]
        with torch.no_grad():
            Xt,Pt,inp=gru_inputs(x[None,:],P_lin[None,:],sc); P_gru=(Pt+sc*m(inp)).numpy()[0]
        a=kg(max(ests[j]['q2_hat'],1e-12),max(ests[j]['sigma2_hat'],1e-12)); P_ema=ema_pred(x,a)
        y=x[va+1:te]; sl=slice(va,te-1)        # predict x_{t+1} from x_{<=t}, t in test
        e={k:(p[sl]-y)**2 for k,p in [('naive',x),('ema',P_ema),('lin',P_lin),('gru',P_gru)]}
        blk=7*per if per else 20
        lo,hi,p=block_boot(e['lin'],e['gru'],blk)
        per_series.append(dict(mse={k:float(v.mean()) for k,v in e.items()},gain=float(e['lin'].mean()/e['gru'].mean()),gain_ci=[lo,hi],p=p,a_rule1=a,
                               sq_lin=e['lin'].tolist() if len(xs)>1 else None, sq_gru=e['gru'].tolist() if len(xs)>1 else None))
    res['per_series']=per_series
    if len(xs)>1:   # pooled FX
        el=np.concatenate([np.array(r['sq_lin']) for r in per_series]); eg=np.concatenate([np.array(r['sq_gru']) for r in per_series])
        lo,hi,p=block_boot(el,eg,20); res['pooled']=dict(gain=float(el.mean()/eg.mean()),gain_ci=[lo,hi],p=p)
        for r in per_series: r.pop('sq_lin'); r.pop('sq_gru')
    else:
        for r in per_series: r.pop('sq_lin'); r.pop('sq_gru')
    res['time_s']=round(time.time()-t0)
    json.dump(res,open(f'real_{name}_seed{seed}.json','w'),indent=1)
    print(name,seed,'best_it',res['best_iter'],'S_hat',[round(v,2) for v in res['S_hat']],[ (round(r['gain'],4),[round(c,4) for c in r['gain_ci']]) for r in per_series], res.get('pooled'), res['time_s'],'s',flush=True)
if __name__=='__main__': main(sys.argv[1],int(sys.argv[2]))
