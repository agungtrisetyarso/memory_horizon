"""Simulations for the Neural Computation manuscript.
All results written to results.json; figures to fig_*.pdf. Seeded."""
import numpy as np, json, math
from scipy.special import exp1
from scipy.optimize import minimize_scalar
R = {}
rng = np.random.default_rng(20261002)

# ---------- E1: continuous-time identity, Euler-Maruyama -------------
def ct_mse(gammas, q, sigma, target, T=4000.0, dt=0.005, seeds=4):
    out = np.zeros(len(gammas))
    for s in range(seeds):
        r = np.random.default_rng(1000+s)
        g = np.array(gammas); h = np.zeros_like(g); m = 0.0; acc = np.zeros_like(g); n=0
        N = int(T/dt); burn = int(N*0.1)
        lam = 0.05  # jump rate for compound Poisson
        for k in range(N):
            if target == 'brownian':
                m += q*math.sqrt(dt)*r.standard_normal()
            else:  # compound Poisson, Gaussian jumps with variance q^2/lam
                if r.random() < lam*dt: m += (q/math.sqrt(lam))*r.standard_normal()
            dy = m*dt + sigma*math.sqrt(dt)*r.standard_normal()
            h += g*(dy - h*dt)
            if k >= burn: acc += (m-h)**2; n+=1
        out += acc/n
    return out/seeds
gam = np.geomspace(0.03, 3.0, 13)
E1 = {}
for target in ['brownian','jump']:
    for q,sig in [(0.1,1.0),(0.3,0.5)]:
        sim = ct_mse(gam, q, sig, target)
        th = q**2/(2*gam) + gam*sig**2/2
        E1[f"{target}_q{q}_s{sig}"] = dict(gamma=gam.tolist(), sim=sim.tolist(), theory=th.tolist(),
            max_rel_err=float(np.max(np.abs(sim-th)/th)))
R['E1_identity'] = E1
print('E1', {k:round(v['max_rel_err'],3) for k,v in E1.items()})

# ---------- E2: discrete-time exact formula and Kalman gain ----------
def dt_mse(a, q2, s2): return ((1-a)**2*q2 + a**2*s2)/(a*(2-a))
def kalman_gain(q2, s2):
    # steady-state: P = Pp*s2/(Pp+s2), Pp = P+q2 -> Pp^2 - q2 Pp - q2 s2 = 0
    Pp = (q2 + math.sqrt(q2**2 + 4*q2*s2))/2
    return Pp/(Pp+s2)
E2=[]
for q2 in [1e-4,1e-3,1e-2,1e-1,1.0,10.0]:
    res = minimize_scalar(lambda a: dt_mse(a,q2,1.0), bounds=(1e-6,1-1e-9), method='bounded', options={'xatol':1e-12})
    E2.append(dict(q2=q2, a_opt=res.x, kalman=kalman_gain(q2,1.0), sqrt_rule=math.sqrt(q2)))
# simulate discrete EMA to check formula
def sim_ema(a_list, q2, s2, T=400000, seed=7, jumps=None):
    r=np.random.default_rng(seed); a=np.array(a_list); h=np.zeros_like(a); m=0.0; acc=np.zeros_like(a)
    J = r.standard_normal(T)*math.sqrt(q2) if jumps is None else jumps
    X = r.standard_normal(T)*math.sqrt(s2)
    burn=T//20
    for t in range(T):
        m += J[t]; h += a*(m+X[t]-h)
        if t>=burn: acc += (m-h)**2
    return acc/(T-burn)
aa = np.array([0.01,0.03,0.1,0.3])
simv = sim_ema(aa, 0.01, 1.0)
R['E2_discrete'] = dict(table=E2, check_a=aa.tolist(), sim=simv.tolist(), theory=[dt_mse(a,0.01,1.0) for a in aa])
print('E2', E2[:3], simv, [dt_mse(a,0.01,1.0) for a in aa])
R['cosh_factor2'] = math.cosh(math.log(2))
R['cosh_bank'] = {str(r_): math.cosh(0.5*math.log(r_)) for r_ in [1.5,2,3,math.e,4,10]}
json.dump(R, open('results_part1.json','w'), indent=1)
