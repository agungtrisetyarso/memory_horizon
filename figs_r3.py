import json, math, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']; ink='#52514e'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e6e3','grid.linewidth':0.6,'lines.linewidth':1.5,'legend.frameon':False,'savefig.bbox':'tight'})
R=json.load(open('results_ci.json')); E4=json.load(open('results.json'))['E4']; h=0.01
def Jd(a,q2): return ((1-a)**2*q2+a*a)/(a*(2-a))
def kg(q2):
    P=(q2+math.sqrt(q2*q2+4*q2))/2; return P/(P+1)
# ---- Fig nets (E3)
E3=R['E3']; S=np.array([r['S'] for r in E3])
f,ax=plt.subplots(1,2,figsize=(7.4,3.0))
Sl=np.geomspace(0.25,1300,200); ax[0].plot(Sl,[Jd(kg(h*s*h),h*s*h)/h for s in Sl],color=C[0],label='best linear (exact)')
for key,lab,col,mk,ls in [('LRNN','LRNN-16 (linear)',ink,'x',''),('SEL','selective SSM (Mamba-style)',C[4],'v','--'),('GRUG','GRU-gated delta rule',C[1],'s','--')]:
    m=np.array([r[key]['err_mean'] for r in E3])/h; ci=np.array([r[key]['err_ci'] for r in E3])/h
    ax[0].errorbar(S,m,yerr=ci,fmt=mk,color=col,ms=5,mfc='white' if mk!='x' else col,ls=ls,capsize=2,elinewidth=0.8,label=lab+' (3 seeds)')
e4=[e for e in E4 if 0.25<=e['S']<=1300]
ax[0].plot([e['S'] for e in e4],[e['runlength']/h for e in e4],'^',color=C[2],ms=5,mfc='white',ls='-.',label='run-length mixture (Bayes)')
ax[0].plot(S,[r['genie']/h for r in E3],'D',color=C[3],ms=4.5,mfc='white',ls=':',label='genie floor (lower bound)')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel(r'regime SNR $S=\Delta^2/(h\sigma^2)$'); ax[0].set_ylabel(r'test MSE $/(h\sigma^2)$')
ax[0].legend(fontsize=6,loc='upper left'); ax[0].set_title('(a) trained memories vs. theory',fontsize=9,loc='left')
ax[1].plot(S,[r['a_star'] for r in E3],color=C[0],label=r'Kalman gain $a^\star$')
ax[1].plot(S,[r['GRUG']['gain_after_jump'] for r in E3],'s',color=C[1],ms=5,ls='--',label='GRU-gated: after a jump')
ax[1].plot(S,[r['GRUG']['gain_elsewhere'] for r in E3],'s',color=C[1],ms=5,mfc='white',ls=':',label='GRU-gated: elsewhere')
ax[1].plot(S,[r['SEL']['gain_after_jump'] for r in E3],'v',color=C[4],ms=5,ls='--',label='selective SSM: after a jump')
ax[1].plot(S,[r['SEL']['gain_elsewhere'] for r in E3],'v',color=C[4],ms=5,mfc='white',ls=':',label='selective SSM: elsewhere')
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel(r'$S$'); ax[1].set_ylabel('mean write rate $a_t$'); ax[1].legend(fontsize=6)
ax[1].set_title('(b) learned gating',fontsize=9,loc='left')
f.subplots_adjust(wspace=0.32); f.savefig('fig_nets.pdf')
# ---- Fig e6 with CIs
E6=R['E6']; fam={'g_':('Gaussian jumps',C[0],'o'),'l_':('Laplace jumps',C[1],'s'),'m_':('jumps + diffusion',C[2],'^'),'br':('pure diffusion',C[3],'D')}
FLOOR=0.05
def genie(s,h):
    D2=s*h; k=np.arange(1,200000); return np.sum(h*(1-h)**(k-1)*D2/(1+k*D2))
f,ax=plt.subplots(1,2,figsize=(7.4,3.0))
for pre,(lab,col,mk) in fam.items():
    rr=[r for r in E6 if r['name'].startswith(pre)]
    for k,xkey in enumerate(['S_hat','S_kappa_true']):
        ax[k].errorbar([max(r[xkey],FLOOR) for r in rr],[r['gain_mean'] for r in rr],yerr=[r['gain_ci'] for r in rr],fmt=mk,color=col,ms=6,mfc='white',mew=1.4,capsize=2,elinewidth=0.8,label=lab)
Ss=np.geomspace(0.3,2000,60); ax[1].plot(Ss,[Jd(kg(h*s*h),h*s*h)/genie(s,h) for s in Ss],color=ink,lw=1,ls='--',label=r'ceiling $\rho_d(S)$, $h=0.01$')
for a_ in ax:
    a_.axhline(1,color=ink,lw=0.8); a_.set_xscale('log'); a_.set_yscale('log'); a_.set_ylim(0.8,12); a_.axvspan(FLOOR*0.7,10,color='#f0f0ec',zorder=0)
ax[0].set_xlabel(r'pre-training estimate $\hat S_\kappa$ (observations only)'); ax[0].set_ylabel('gain from gating (linear / gated error)')
ax[0].set_title('(a) predicted before training',fontsize=9,loc='left'); ax[0].legend(fontsize=6.5,loc='upper left')
ax[1].set_xlabel(r'true $S_\kappa$'); ax[1].set_title('(b) against the genie ceiling',fontsize=9,loc='left'); ax[1].legend(fontsize=6.3,loc='upper left')
f.subplots_adjust(wspace=0.3); f.savefig('fig_e6.pdf'); print('ok')
