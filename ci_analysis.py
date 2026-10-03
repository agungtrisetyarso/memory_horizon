"""Seed-level means, 95% t-intervals (n=3), and paired bootstrap over test sequences."""
import json, glob, math, numpy as np
from scipy import stats
rng=np.random.default_rng(0)
def tci(x):
    x=np.asarray(x,float); n=len(x); m=x.mean()
    if n<2: return m,float('nan')
    return m, stats.t.ppf(0.975,n-1)*x.std(ddof=1)/math.sqrt(n)
def boot_ratio(num_seq,den_seq,B=10000):
    """paired bootstrap over sequences of ratio of means; returns (ratio, lo, hi, p two-sided for ratio=1)"""
    num=np.asarray(num_seq); den=np.asarray(den_seq); n=len(num); r=num.mean()/den.mean()
    idx=rng.integers(0,n,(B,n)); rb=num[idx].mean(1)/den[idx].mean(1)
    p=2*min((rb<=1).mean(),(rb>=1).mean())
    return r,float(np.percentile(rb,2.5)),float(np.percentile(rb,97.5)),max(p,1/B)
out={'E3':[],'E6':[]}
# ---------------- E3
old={json.load(open(f))['S']:json.load(open(f)) for f in glob.glob('res_S*.json')}
for S in sorted(old):
    new=[json.load(open(f)) for f in sorted(glob.glob(f'r3/e3b_S{S:g}_seed[23].json'))]
    sel1=glob.glob(f'r3/e3b_S{S:g}_seed1_SEL.json')
    if len(new)<2 or not sel1: continue
    sel1=json.load(open(sel1[0]))
    lin=np.mean(new[0]['linear_opt_seq'])   # same test set for all seeds
    rec=dict(S=S,linear_opt=old[S]['linear_opt'],genie=old[S]['genie'])
    for mname in ['CG','LRNN','GRUG','SEL']:
        errs=([old[S][mname]] if mname!='SEL' else [float(np.mean(sel1['SEL_seq']))])+[float(np.mean(r[mname+'_seq'])) for r in new]
        ratios=[old[S]['linear_opt']/e for e in errs]
        m,hw=tci(errs); rm,rhw=tci(ratios)
        rec[mname]=dict(err_seeds=errs,err_mean=m,err_ci=hw,lin_over_model_seeds=ratios,ratio_mean=rm,ratio_ci=rhw)
    # paired bootstrap on new seeds: per-sequence errors averaged over seeds 2,3
    linseq=np.array(new[0]['linear_opt_seq'])
    for mname in ['GRUG','SEL','LRNN']:
        mseq=np.mean([r[mname+'_seq'] for r in new],axis=0)
        r,lo,hi,p=boot_ratio(linseq,mseq); rec[mname]['boot']=dict(ratio=r,lo=lo,hi=hi,p=p)
    for mname in ['GRUG','SEL']:
        rec[mname]['gain_after_jump']=float(np.mean([r[mname+'_gain_after_jump'] for r in new]))
        rec[mname]['gain_elsewhere']=float(np.mean([r[mname+'_gain_elsewhere'] for r in new]))
    rec['CG_gain_seeds']=[old[S]['CG_gain']]+[r['CG_gain'] for r in new]; rec['a_star']=old[S]['a_star']
    rec['genie_over_GRUG']=[g for g in [old[S]['genie']/e for e in rec['GRUG']['err_seeds']]]
    out['E3'].append(rec)
# ---------------- E6
for f in sorted(glob.glob('e6/e6_*.json')):
    r0=json.load(open(f)); name=r0['name']
    ns=[json.load(open(g)) for g in sorted(glob.glob(f'e6/e6s_{name}_seed[12].json'))]
    if len(ns)<2: continue
    gains=[r0['gain_lin_over_gru']]+[r['LIN_excess']/r['GRU_excess'] for r in ns]
    m,hw=tci(gains)
    linseq=np.array(ns[0]['LIN_seq']); gseq=np.mean([r['GRU_seq'] for r in ns],axis=0)
    rb,lo,hi,p=boot_ratio(linseq,gseq)
    out['E6'].append(dict(name=name,S_hat=r0['S_hat'],S_kappa_true=r0['S_kappa_true'],gains=gains,gain_mean=m,gain_ci=hw,boot=dict(ratio=rb,lo=lo,hi=hi,p=p)))
json.dump(out,open('results_ci.json','w'),indent=1)
for r in out['E3']:
    print('S=%g'%r['S'],' '.join('%s %.3f±%.3f'%(k,r[k]['ratio_mean'],r[k]['ratio_ci']) for k in ['CG','LRNN','SEL','GRUG']),'| boot GRUG p=%.4f SEL p=%.4f'%(r['GRUG']['boot']['p'],r['SEL']['boot']['p']))
for r in out['E6']: print(r['name'],'Shat %.1f gain %.3f±%.3f boot [%.3f,%.3f] p=%.4f'%(r['S_hat'],r['gain_mean'],r['gain_ci'],r['boot']['lo'],r['boot']['hi'],r['boot']['p']))
