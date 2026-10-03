import json, glob, math, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']; ink='#52514e'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e6e3','grid.linewidth':0.6,'lines.linewidth':1.5,'legend.frameon':False,'savefig.bbox':'tight'})
rs=[json.load(open(f)) for f in glob.glob('e6/e6_*.json')]
fam={'gauss':('Gaussian jumps',C[0],'o'),'laplace':('Laplace jumps',C[1],'s'),'mix':('jumps + diffusion',C[2],'^'),'brown':('pure diffusion',C[3],'D')}
FLOOR=0.05
def kg(q2):
    P=(q2+math.sqrt(q2*q2+4*q2))/2; return P/(P+1)
def Jd(a,q2): return ((1-a)**2*q2+a*a)/(a*(2-a))
def genie(S,h):
    D2=S*h; k=np.arange(1,200000); return np.sum(h*(1-h)**(k-1)*D2/(1+k*D2))
f,ax=plt.subplots(1,2,figsize=(7.4,3.0))
for key,(lab,col,mk) in fam.items():
    rr=[r for r in rs if r['spec']['fam']==key]
    if not rr: continue
    ax[0].plot([max(r['S_hat'],FLOOR) for r in rr],[r['gain_lin_over_gru'] for r in rr],mk,color=col,ms=6,mfc='white',mew=1.4,label=lab)
    ax[1].plot([max(r['S_kappa_true'],FLOOR) for r in rr],[r['gain_lin_over_gru'] for r in rr],mk,color=col,ms=6,mfc='white',mew=1.4,label=lab)
Ss=np.geomspace(0.3,2000,60); h=0.01
ceil=[Jd(kg(h*s*h),h*s*h)/genie(s,h) for s in Ss]
ax[1].plot(Ss,ceil,color=ink,lw=1,ls='--',label=r'ceiling $\rho_d(S)$, $h=0.01$')
for a_ in ax:
    a_.axhline(1,color=ink,lw=0.8); a_.set_xscale('log'); a_.set_yscale('log'); a_.set_ylim(0.8,12)
    a_.axvspan(FLOOR*0.7,10,color='#f0f0ec',zorder=0)
ax[0].set_xlabel(r'pre-training estimate $\hat S_\kappa$ (from observations only)'); ax[0].set_ylabel('gain from gating (linear / gated error)')
ax[0].set_title('(a) predicted before training',fontsize=9,loc='left'); ax[0].legend(fontsize=6.5,loc='upper left')
ax[1].set_xlabel(r'true $S_\kappa$'); ax[1].set_title('(b) against the genie ceiling',fontsize=9,loc='left'); ax[1].legend(fontsize=6.3,loc='upper left')
f.subplots_adjust(wspace=0.3); f.savefig('fig_e6.pdf'); print('ok')
