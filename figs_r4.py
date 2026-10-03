import json, math, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']; ink='#52514e'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e6e3','grid.linewidth':0.6,'lines.linewidth':1.4,'legend.frameon':False,'savefig.bbox':'tight'})
R=json.load(open('results_ci.json')); R4=json.load(open('results_r4.json')); E4={e['S']:e for e in json.load(open('results.json'))['E4']}
E3=R['E3']; S=np.array([r['S'] for r in E3])
def band(ax,x,vals,col,mk,lab,ls='-',mfc='white'):
    m=np.array([np.mean(v) for v in vals]); lo=np.array([min(v) for v in vals]); hi=np.array([max(v) for v in vals])
    ax.errorbar(x,m,yerr=[m-lo,hi-m],fmt=mk,color=col,ms=5,mfc=mfc,ls=ls,capsize=2,elinewidth=0.9,label=lab)
f,ax=plt.subplots(1,2,figsize=(7.4,3.1))
ax[0].plot(S,[r['linear_opt']/r['genie'] for r in E3],color=ink,ls='--',lw=1,label='ceiling (linear / genie floor)')
ax[0].plot(S,[r['linear_opt']/E4[r['S']]['runlength'] for r in E3],'^',color=C[2],ls='-.',ms=5,mfc='white',label='Bayesian run-length filter')
band(ax[0],S,[r['GRUG']['lin_over_model_seeds'] for r in E3],C[1],'s','GRU-gated (latent), 3 seeds','--')
go=R4['GRUG_obs']; ax[0].plot([r['S'] for r in go],[r['mean'] for r in go],'x',color=C[1],ms=7,mew=1.5,ls='',label='GRU-gated (obs.-only), 3 seeds')
band(ax[0],[r['S'] for r in R4['SELI']],[r['ratios'] for r in R4['SELI']],C[0],'D',r'SEL-$\nu$ (innovation-selected), 5 seeds',':')
band(ax[0],[r['S'] for r in R4['SELx']],[r['ratios'] for r in R4['SELx']],C[4],'v',r'SEL-$x$ (input-selected), 3-10 seeds',':')
band(ax[0],S,[r['LRNN']['lin_over_model_seeds'] for r in E3],ink,'o','LRNN-16 (linear)','')
ax[0].axhline(1,color=ink,lw=0.6); ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel(r'regime SNR $S=\Delta^2/(h\sigma^2)$'); ax[0].set_ylabel('gain over best linear memory')
ax[0].legend(fontsize=5.6,loc='upper left'); ax[0].set_title('(a) gains (mean and range over seeds)',fontsize=9,loc='left')
ax[1].plot(S,[r['a_star'] for r in E3],color=ink,lw=1,label=r'Kalman gain $a^\star$')
ax[1].plot(S,[r['GRUG']['gain_after_jump'] for r in E3],'s',color=C[1],ms=5,ls='--',label='GRU-gated: after a jump')
ax[1].plot(S,[r['GRUG']['gain_elsewhere'] for r in E3],'s',color=C[1],ms=5,mfc='white',ls=':',label='GRU-gated: elsewhere')
ax[1].plot([r['S'] for r in R4['SELI']],[r['after'] for r in R4['SELI']],'D',color=C[0],ms=4.5,ls='--',label=r'SEL-$\nu$: after a jump')
ax[1].plot([r['S'] for r in R4['SELI']],[r['elsewhere'] for r in R4['SELI']],'D',color=C[0],ms=4.5,mfc='white',ls=':',label=r'SEL-$\nu$: elsewhere')
ax[1].plot(S,[r['SEL']['gain_after_jump'] for r in E3],'v',color=C[4],ms=5,ls='--',label=r'SEL-$x$: after a jump')
ax[1].plot(S,[r['SEL']['gain_elsewhere'] for r in E3],'v',color=C[4],ms=5,mfc='white',ls=':',label=r'SEL-$x$: elsewhere')
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel(r'$S$'); ax[1].set_ylabel('mean write rate $a_t$'); ax[1].legend(fontsize=5.6)
ax[1].set_title('(b) learned gating',fontsize=9,loc='left')
f.subplots_adjust(wspace=0.3); f.savefig('fig_nets.pdf'); print('ok')
