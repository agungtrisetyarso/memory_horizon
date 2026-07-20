import numpy as np, scipy.io as sio

# ============================================================
# ONLINE PORTFOLIO SELECTION (Cover's universal portfolio setting)
# Each day t: learner holds portfolio b(t) in simplex over d assets.
# Price relatives x(t) in R^d_+ revealed. Wealth multiplies by <b(t), x(t)>.
# Objective: maximize log-wealth  sum_t log <b(t), x(t)>.
# Regret vs best constant-rebalanced portfolio (BCRP) or best single stock.
#
# Connection to our framework: the per-day LOSS is  l_i(t) = -log x_i(t)
# (negative log-return of asset i). Low loss = high return. Experts = assets.
# A non-stationary market => the best asset drifts => dynamic-regret regime.
# This is a REAL, standard benchmark (OLPS: djia, nyse-o), not synthetic.
# ============================================================

def load(name):
    m=sio.loadmat(name); X=np.array(m['data'])   # T x d price relatives
    return X

def losses_from_prices(X):
    # loss_i(t) = -log(x_i(t)); shift/scale into a bounded surrogate for the
    # experts algorithms that assume l in [0,1], but keep TRUE returns for wealth.
    logret = np.log(X)                 # T x d
    return logret

def run_wealth(B, X):
    # B: T x d portfolios; returns cumulative log-wealth trajectory
    daily = np.log((B*X).sum(1))
    return np.cumsum(daily)

# ---- learners operate on the loss stream l = -log x, output portfolios ----
def to_loss01(X):
    L=-np.log(X)                       # positive-ish; normalize per-day to [0,1]
    lo,hi=L.min(),L.max()
    return (L-lo)/(hi-lo+1e-12)

def dissipative_mw(Ln,eta,gamma0):
    T,d=Ln.shape; hatE=np.zeros(d); p=np.ones(d)/d; a=1-np.exp(-gamma0); B=np.zeros((T,d))
    for t in range(T):
        B[t]=p
        hatE=(1-a)*hatE+a*Ln[t]
        hh=eta*hatE/max(a,1e-9); w=np.exp(-(hh-hh.min())); w/=w.sum()
        p=p+a*(w-p); p=np.clip(p,1e-12,None); p/=p.sum()
    return B

def discounted_hedge(Ln,eta,gamma):
    T,d=Ln.shape; L=np.zeros(d); B=np.zeros((T,d)); g=1-np.exp(-gamma)
    for t in range(T):
        w=np.exp(-eta*L); w/=w.sum(); B[t]=w
        L=(1-g)*L+g*Ln[t]/max(g,1e-9)
    return B

def fixed_share(Ln,eta,alpha):
    T,d=Ln.shape; w=np.ones(d)/d; B=np.zeros((T,d))
    for t in range(T):
        B[t]=w/w.sum(); w=w*np.exp(-eta*Ln[t]); w/=w.sum(); w=(1-alpha)*w+alpha/d; w/=w.sum()
    return B

def dynamic_mirror_descent(Ln,eta,alpha):
    T,d=Ln.shape; th=np.zeros(d); B=np.zeros((T,d))
    for t in range(T):
        w=np.exp(th-th.max()); w/=w.sum(); B[t]=w; th=th-eta*Ln[t]; th=(1-alpha)*th
    return B

def adahedge(Ln):
    T,d=Ln.shape; L=np.zeros(d); delta=0.0; B=np.zeros((T,d))
    for t in range(T):
        if delta<=0: w=np.ones(d)/d; eta=np.inf
        else:
            eta=np.log(d)/delta; m=L.min(); u=np.exp(-eta*(L-m)); w=u/u.sum()
        B[t]=w; l=Ln[t]; hl=w@l
        if np.isfinite(eta):
            mix=hl+(1/eta)*np.log((w*np.exp(-eta*l)).sum()); delta+=max(0.0,mix)
        L+=l
    return B

def ucrp(Ln):  # uniform constant-rebalanced portfolio (classic baseline)
    T,d=Ln.shape; return np.ones((T,d))/d

def best_stock_hindsight(X):
    # log-wealth of best single stock
    return np.log(X).sum(0).max()

def bcrp_hindsight(X, iters=300):
    # best constant-rebalanced portfolio in hindsight (concave -> exponentiated grad)
    T,d=X.shape; b=np.ones(d)/d
    for _ in range(iters):
        denom=(X@b)
        grad=(X/denom[:,None]).sum(0)     # gradient of sum log(<b,x>)
        b=b*grad; b/=b.sum()
    return np.log((X@b)).sum()

if __name__=="__main__":
    for name in ["djia.mat","nyse_o.mat"]:
        X=load(name); Ln=to_loss01(X)
        print(f"\n=== {name}: T={X.shape[0]}, d={X.shape[1]} ===")
        bcrp=bcrp_hindsight(X); bstock=best_stock_hindsight(X)
        print(f"  BCRP log-wealth (hindsight): {bcrp:.4f}   best-stock: {bstock:.4f}")
        methods={
          "UCRP":ucrp(Ln),
          "Dissipative MW":dissipative_mw(Ln,4,0.1),
          "Discounted Hedge":discounted_hedge(Ln,4,0.1),
          "Fixed-Share":fixed_share(Ln,4,0.02),
          "Dynamic Mirror Descent":dynamic_mirror_descent(Ln,4,0.02),
          "AdaHedge":adahedge(Ln),
        }
        for k,B in methods.items():
            w=run_wealth(B,X)[-1]
            print(f"  {k:24s} final log-wealth={w:7.4f}   (regret vs BCRP={bcrp-w:7.4f})")
