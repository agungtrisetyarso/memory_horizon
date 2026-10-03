import json, glob, math, numpy as np, matplotlib
from scipy import stats
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']; ink='#52514e'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e6e3','grid.linewidth':0.6,'legend.frameon':False,'savefig.bbox':'tight'})
def tci(x):
    x=np.asarray(x,float); n=len(x); return float(x.mean()), float(stats.t.ppf(0.975,n-1)*x.std(ddof=1)/math.sqrt(n)) if n>1 else float('nan')
names=['ETTh1','ETTh2','ETTm1','ETTm2','ELEC103','ELEC319','ELEC99','ELEC21','ELEC64','ELEC1','ELEC232','ELECAGG','FX']
out=[]
for nm in names:
    rs=[json.load(open(f)) for f in sorted(glob.glob(f'real/real_{nm}_seed*.json'))]
    if not rs: continue
    if nm=='FX':
        r0=rs[0]; g=[r['pooled']['gain'] for r in rs]; m,h=tci(g)
        out.append(dict(name=nm,S_hat=r0['S_hat'],sigma2_hat=r0['sigma2_hat'],gain_seeds=g,gain=m,gain_ci=h,
                        boot=[r['pooled']['gain_ci'] for r in rs],p=[r['pooled']['p'] for r in rs],
                        per_currency_gain=[float(np.mean([r['per_series'][k]['gain'] for r in rs])) for k in range(r0['n_series'])],
                        mse={k:float(np.mean([np.mean([ps['mse'][k] for ps in r['per_series']]) for r in rs])) for k in ['naive','ema','lin','gru']}))
        continue
    r0=rs[0]; g=[r['per_series'][0]['gain'] for r in rs]; m,h=tci(g)
    out.append(dict(name=nm,S_hat=r0['S_hat'][0],S_hat_blocks=r0['S_hat_blocks'][0],sigma2_hat=r0['sigma2_hat'][0],q2_hat=r0['q2_hat'][0],
                    split=r0['split'],gain_seeds=g,gain=m,gain_ci=h,boot=[r['per_series'][0]['gain_ci'] for r in rs],p=[r['per_series'][0]['p'] for r in rs],
                    mse={k:float(np.mean([r['per_series'][0]['mse'][k] for r in rs])) for k in ['naive','ema','lin','gru']},a_rule1=r0['per_series'][0]['a_rule1']))
json.dump(out,open('results_real.json','w'),indent=1)
for o in out:
    print('%-8s Shat %10.2f  gain %.4f±%.4f seeds %s p %s | mse naive %.4g ema %.4g lin %.4g gru %.4g'%(o['name'],o['S_hat'] if isinstance(o['S_hat'],float) else -1,o['gain'],o['gain_ci'],['%.4f'%x for x in o['gain_seeds']],['%.3f'%x for x in o['p']],o['mse']['naive'],o['mse']['ema'],o['mse']['lin'],o['mse']['gru']))
# figure
FLOOR=0.05
f,ax=plt.subplots(figsize=(4.6,3.1))
for o in out:
    if o['name']=='FX': continue
    col,mk,lab=(C[0],'o','ETT oil temperature') if o['name'].startswith('ETT') else ((C[1],'s','electricity, single client') if o['name']!='ELECAGG' else (C[2],'^','electricity, aggregate'))
    ax.errorbar(max(o['S_hat'],FLOOR),o['gain'],yerr=o['gain_ci'] if np.isfinite(o['gain_ci']) else None,fmt=mk,color=col,ms=6,mfc='white',mew=1.4,capsize=2,elinewidth=0.8,label=lab)
h,l=ax.get_legend_handles_labels(); d=dict(zip(l,h)); ax.legend(d.values(),d.keys(),fontsize=6.5,loc='upper left')
ax.axhline(1,color=ink,lw=0.8); ax.set_xscale('log'); ax.axvspan(FLOOR*0.7,10,color='#f0f0ec',zorder=0)
ax.set_xlabel(r'pre-training estimate $\hat S_\kappa$ (training split only)'); ax.set_ylabel('gain from gating\n(linear / gated test MSE)')
ax.set_title('E7: real series (values 0 plotted at 0.05)',fontsize=9,loc='left')
f.savefig('fig_real.pdf'); print('fig ok')
