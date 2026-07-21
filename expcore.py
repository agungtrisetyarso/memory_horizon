
import json
import numpy as np

RNG_SEED = 20260721


# ----------------------------------------------------------------------
# Environments
# ----------------------------------------------------------------------
def switching_stream(T, d, K, sigma, rng):
    """Best expert switches every T/K rounds. Clean losses in [0,1]."""
    seg = T // K
    best = rng.integers(0, d, size=K)
    clean = np.full((T, d), 0.6)
    for k in range(K):
        lo, hi = k * seg, min((k + 1) * seg, T)
        clean[lo:hi, :] = 0.6
        clean[lo:hi, best[k]] = 0.2
    obs = clean + sigma * rng.standard_normal((T, d))
    # path length of the per-round best-expert comparator (1-hot), in L1
    P_T = 2 * (np.sum(np.diff(best) != 0))
    return clean, obs, P_T


def drifting_stream(T, d, sigma, rng, speed=1.0):
    """Each expert's mean loss is a phase-shifted sinusoid; best rotates."""
    t = np.arange(T)[:, None]
    phase = np.linspace(0, 2 * np.pi, d, endpoint=False)[None, :]
    clean = 0.4 + 0.2 * np.sin(2 * np.pi * speed * t / T * 4 + phase)
    obs = clean + sigma * rng.standard_normal((T, d))
    best = np.argmin(clean, axis=1)
    P_T = 2 * np.sum(np.diff(best) != 0)
    return clean, obs, P_T


# ----------------------------------------------------------------------
# Learners  (all operate on observed losses, evaluated on clean losses)
# ----------------------------------------------------------------------
def discounted_hedge(obs, eta, gamma, tau_relax=None):
    """Exponential weights on exponentially-discounted cumulative loss.
    If tau_relax is not None, the *played* distribution relaxes toward the
    Gibbs target at continuous rate 1/tau_relax (the dissipative variant).
    Returns played distribution p(t) each round."""
    T, d = obs.shape
    E = np.zeros(d)                      # discounted cumulative loss
    p = np.full(d, 1.0 / d)              # played distribution
    P = np.zeros((T, d))
    relax = None if tau_relax is None else 1.0 / tau_relax
    for t in range(T):
        E = (1 - gamma) * E + obs[t]     # discrete discount, discount ~ gamma
        w = np.exp(-eta * E)
        pi = w / w.sum()                 # Gibbs target
        if relax is None:
            p = pi
        else:
            p = p + relax * (pi - p)     # finite-rate relaxation (Euler step)
            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
        P[t] = p
    return P


def fixed_share(obs, eta, alpha):
    T, d = obs.shape
    w = np.full(d, 1.0 / d)
    P = np.zeros((T, d))
    for t in range(T):
        w = w * np.exp(-eta * obs[t])
        w = w / w.sum()
        w = (1 - alpha) * w + alpha / d
        P[t] = w
    return P


def dynamic_mirror_descent(obs, eta, alpha):
    T, d = obs.shape
    w = np.full(d, 1.0 / d)
    P = np.zeros((T, d))
    for t in range(T):
        w = w * np.exp(-eta * obs[t])
        w = w / w.sum()
        w = (1 - alpha) * w + alpha / d   # shrink-to-uniform dynamical model
        P[t] = w
    return P


def adahedge(obs):
    T, d = obs.shape
    L = np.zeros(d)
    Delta = 1e-8
    P = np.zeros((T, d))
    for t in range(T):
        eta = np.log(d) / max(Delta, 1e-8)
        w = np.exp(-eta * (L - L.min()))
        w = w / w.sum()
        P[t] = w
        loss = obs[t]
        mix = w @ loss
        # mixability gap (clipped for stability)
        m = -1.0 / eta * np.log(np.clip(w @ np.exp(-eta * (loss - loss.min())), 1e-300, None)) + loss.min()
        Delta += max(0.0, mix - m)
        L += loss
    return P


def dynamic_regret(P, clean):
    """Regret vs per-round best expert (1-hot comparator)."""
    played = np.sum(P * clean, axis=1)
    best = clean.min(axis=1)
    return float(np.sum(played - best))


# ----------------------------------------------------------------------
# Experiment 1: U-shape and optimal-horizon law
# ----------------------------------------------------------------------
def exp_ushape(seeds=15):
    d, T = 10, 2000
    taus = np.unique(np.round(np.logspace(np.log10(1.5), np.log10(400), 22)).astype(int))
    eta = 1.0
    results = {"taus": taus.tolist()}
    for env_name, K, sigma in [("switching", 16, 0.5),
                               ("switching_highnoise", 16, 0.9),
                               ("drifting", None, 0.5)]:
        curve = np.zeros((seeds, len(taus)))
        Ps = []
        for s in range(seeds):
            rng = np.random.default_rng(RNG_SEED + s)
            if env_name.startswith("switching"):
                clean, obs, P_T = switching_stream(T, d, K, sigma, rng)
            else:
                clean, obs, P_T = drifting_stream(T, d, sigma, rng)
            Ps.append(P_T)
            for j, tau in enumerate(taus):
                gamma = 1.0 / tau
                Pl = discounted_hedge(obs, eta, gamma)
                curve[s, j] = dynamic_regret(Pl, clean)
        mean = curve.mean(0)
        results[env_name] = {
            "mean": mean.tolist(),
            "std": curve.std(0).tolist(),
            "P_T": float(np.mean(Ps)),
            "tau_star": int(taus[np.argmin(mean)]),
            "sigma": sigma,
        }
    return results


# ----------------------------------------------------------------------
# Experiment 2: tau* vs path length (scaling law)
# ----------------------------------------------------------------------
def exp_scaling(seeds=15):
    d, T, sigma = 10, 2000, 0.5
    Ks = [2, 4, 8, 16, 32, 64, 128]
    taus = np.unique(np.round(np.logspace(np.log10(1.5), np.log10(400), 22)).astype(int))
    eta = 1.0
    tau_star, P_list = [], []
    for K in Ks:
        best_tau_seed = []
        Pvals = []
        for s in range(seeds):
            rng = np.random.default_rng(RNG_SEED + 100 * K + s)
            clean, obs, P_T = switching_stream(T, d, K, sigma, rng)
            Pvals.append(P_T)
            reg = [dynamic_regret(discounted_hedge(obs, eta, 1.0 / tau), clean) for tau in taus]
            best_tau_seed.append(taus[int(np.argmin(reg))])
        tau_star.append(float(np.mean(best_tau_seed)))
        P_list.append(float(np.mean(Pvals)))
    # fit tau* ~ P^exponent
    logP, logtau = np.log(P_list), np.log(tau_star)
    A = np.vstack([logP, np.ones_like(logP)]).T
    slope, intercept = np.linalg.lstsq(A, logtau, rcond=None)[0]
    return {"K": Ks, "P_T": P_list, "tau_star": tau_star,
            "fit_exponent": float(slope), "fit_intercept": float(intercept)}


# ----------------------------------------------------------------------
# Experiment 3: tau* vs noise sigma
# ----------------------------------------------------------------------
def exp_noise(seeds=15):
    d, T, K = 10, 2000, 16
    sigmas = [0.2, 0.35, 0.5, 0.7, 0.9]
    taus = np.unique(np.round(np.logspace(np.log10(1.5), np.log10(400), 22)).astype(int))
    eta = 1.0
    out = {"sigma": sigmas, "tau_star": [], "tau_std": []}
    for sg in sigmas:
        best = []
        for s in range(seeds):
            rng = np.random.default_rng(RNG_SEED + 7000 + int(1000 * sg) + s)
            clean, obs, _ = switching_stream(T, d, K, sg, rng)
            reg = [dynamic_regret(discounted_hedge(obs, eta, 1.0 / tau), clean) for tau in taus]
            best.append(taus[int(np.argmin(reg))])
        out["tau_star"].append(float(np.mean(best)))
        out["tau_std"].append(float(np.std(best)))
    return out


# ----------------------------------------------------------------------
# Experiment 4: baseline comparison table
# ----------------------------------------------------------------------
def _grid_best(learner_fn, obs, clean, grids):
    best = np.inf
    keys = list(grids.keys())
    import itertools
    for combo in itertools.product(*[grids[k] for k in keys]):
        kw = dict(zip(keys, combo))
        P = learner_fn(obs, **kw)
        r = dynamic_regret(P, clean)
        if r < best:
            best = r
    return best


def exp_baselines(seeds=10):
    d, T = 10, 2000
    eta_grid = [0.1, 0.2, 0.5, 1, 2, 5]
    gamma_grid = list(1.0 / np.unique(np.round(np.logspace(np.log10(2), np.log10(300), 10)).astype(int)))
    alpha_grid = [0.005, 0.02, 0.05, 0.1, 0.2]
    envs = {
        "switching": dict(K=16, sigma=0.5),
        "drifting": dict(K=None, sigma=0.5),
        "switching_highnoise": dict(K=16, sigma=0.9),
    }
    table = {}
    for env_name, cfg in envs.items():
        rows = {m: [] for m in ["DiscHedge", "DMD", "FixedShare",
                                "DissipMW", "AdaHedge"]}
        for s in range(seeds):
            rng = np.random.default_rng(RNG_SEED + 555 * s)
            if cfg["K"]:
                clean, obs, _ = switching_stream(T, d, cfg["K"], cfg["sigma"], rng)
            else:
                clean, obs, _ = drifting_stream(T, d, cfg["sigma"], rng)
            rows["DiscHedge"].append(_grid_best(
                lambda o, eta, gamma: discounted_hedge(o, eta, gamma),
                obs, clean, {"eta": eta_grid, "gamma": gamma_grid}))
            rows["DMD"].append(_grid_best(
                dynamic_mirror_descent, obs, clean,
                {"eta": eta_grid, "alpha": alpha_grid}))
            rows["FixedShare"].append(_grid_best(
                fixed_share, obs, clean,
                {"eta": eta_grid, "alpha": alpha_grid}))
            # dissipative MW: extra relaxation parameter
            def dissip(o, eta, gamma, tau_relax):
                return discounted_hedge(o, eta, gamma, tau_relax=tau_relax)
            rows["DissipMW"].append(_grid_best(
                dissip, obs, clean,
                {"eta": eta_grid, "gamma": gamma_grid, "tau_relax": [2, 5, 10]}))
            rows["AdaHedge"].append(dynamic_regret(adahedge(obs), clean))
        table[env_name] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in rows.items()}
    return table


if __name__ == "__main__":
    out = {}
    print("exp_ushape..."); out["ushape"] = exp_ushape()
    print("exp_scaling..."); out["scaling"] = exp_scaling()
    print("exp_noise..."); out["noise"] = exp_noise()
    print("exp_baselines..."); out["baselines"] = exp_baselines()
    with open("results_core.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results_core.json")
    print("scaling fit exponent:", out["scaling"]["fit_exponent"])
    for e, r in out["ushape"].items():
        if isinstance(r, dict):
            print(f"  {e}: tau*={r['tau_star']}  P_T={r['P_T']:.1f}")
