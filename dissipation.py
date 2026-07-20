# ==============================================================================
# Scaling experiments for "Dissipation as a Memory Horizon"
# Colab-ready simulation of the experiments in §5.8 / Figure (scaling)
# ==============================================================================
# Install only if needed (Colab already has them)
# !pip install -q numpy matplotlib scipy tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Environment: switching experts
# ----------------------------------------------------------------------
def generate_switching(T, d, K, sigma, mu_good=0.25, mu_bad=0.75, seed=0):
    """
    Returns
    -------
    true_losses : (T, d)   clean losses in [0,1]
    obs_losses  : (T, d)   noisy observations
    best        : (T,)     index of the best expert at each round
    Path        : float    path length ≈ 2K
    """
    rng = np.random.RandomState(seed)
    period = max(1, T // K)
    best = np.zeros(T, dtype=int)
    true_losses = np.zeros((T, d))

    for k in range(K):
        t0 = k * period
        t1 = min(T, (k + 1) * period)
        expert = k % d
        best[t0:t1] = expert
        true_losses[t0:t1, :] = mu_bad
        true_losses[t0:t1, expert] = mu_good

    # residual rounds
    if K * period < T:
        expert = K % d
        best[K*period:] = expert
        true_losses[K*period:, :] = mu_bad
        true_losses[K*period:, expert] = mu_good

    noise = rng.normal(0, sigma, size=(T, d))
    obs_losses = np.clip(true_losses + noise, 0.0, 1.0)
    Path = 2.0 * K          # each switch costs ~2 in L1
    return true_losses, obs_losses, best, Path


# ----------------------------------------------------------------------
# 2. Discounted Hedge (instantaneous relaxation limit)
# ----------------------------------------------------------------------
def discounted_hedge(obs_losses, eta, gamma0):
    """
    Discrete-time Discounted Hedge:
        E_t = exp(-gamma0) * E_{t-1} + ell_t
        p_t ∝ exp(-eta * E_t)
    Returns sequence of distributions p (T, d)
    """
    T, d = obs_losses.shape
    E = np.zeros(d)
    ps = np.zeros((T, d))
    discount = np.exp(-gamma0)

    for t in range(T):
        E = discount * E + obs_losses[t]
        logits = -eta * E
        ps[t] = np.exp(logits - logsumexp(logits))
    return ps


def dynamic_regret(ps, true_losses, best):
    """Dynamic regret against the sequence of per-round best experts."""
    T = len(best)
    regret = 0.0
    for t in range(T):
        # <p_t - e_{best}, true_loss_t>
        regret += np.dot(ps[t], true_losses[t]) - true_losses[t, best[t]]
    return regret


# ----------------------------------------------------------------------
# 3. Grid search for empirical tau-star
# ----------------------------------------------------------------------
def find_best_tau(T, d, K, sigma, eta=1.0,
                  tau_grid=None, n_seeds=5):
    """
    Returns
    -------
    best_tau : float
    mean_regret_at_best : float
    std_regret_at_best  : float
    all_regrets         : array (len(tau_grid),)
    """
    if tau_grid is None:
        # log-spaced horizons from ~1 to T/2
        tau_grid = np.unique(np.round(np.logspace(0, np.log10(T/2), 18))).astype(float)

    regrets = np.zeros((len(tau_grid), n_seeds))

    for s in range(n_seeds):
        true_l, obs_l, best, _ = generate_switching(T, d, K, sigma, seed=1000+s)
        for i, tau in enumerate(tau_grid):
            gamma0 = 1.0 / tau
            ps = discounted_hedge(obs_l, eta=eta, gamma0=gamma0)
            regrets[i, s] = dynamic_regret(ps, true_l, best)

    mean_r = regrets.mean(axis=1)
    std_r  = regrets.std(axis=1)
    idx    = np.argmin(mean_r)
    return tau_grid[idx], mean_r[idx], std_r[idx], mean_r, tau_grid


# ----------------------------------------------------------------------
# 4. Scaling experiment – Left panel data
# ----------------------------------------------------------------------
print("=== Collecting data for Left panel (tau* vs Path) ===")

Ts     = [2000, 5000]
sigmas = [0.3, 0.5]
Ks     = [4, 8, 12, 16, 24, 32, 48, 64, 96]   # Path ≈ 2K

results_left = []   # list of dicts

for T in Ts:
    for sigma in sigmas:
        for K in tqdm(Ks, desc=f"T={T}, σ={sigma}"):
            d = 10
            tau_star, reg, reg_std, _, _ = find_best_tau(
                T=T, d=d, K=K, sigma=sigma, eta=1.0, n_seeds=5
            )
            Path = 2.0 * K
            results_left.append({
                "T": T, "sigma": sigma, "K": K, "Path": Path,
                "tau_star": tau_star, "regret": reg, "regret_std": reg_std
            })

# ----------------------------------------------------------------------
# 5. Scaling experiment – Right panel data (regret vs sqrt(Path T log d))
# ----------------------------------------------------------------------
print("\n=== Collecting data for Right panel (regret scaling) ===")

configs_right = [
    # (T, d, K, sigma)
    (2000, 10, 8, 0.5),
    (2000, 10, 16, 0.5),
    (2000, 10, 32, 0.5),
    (2000, 30, 16, 0.5),
    (2000, 100, 16, 0.5),
    (5000, 10, 16, 0.5),
    (5000, 10, 40, 0.5),
    (10000, 10, 20, 0.5),
    (10000, 30, 30, 0.5),
]

results_right = []
for T, d, K, sigma in tqdm(configs_right):
    tau_star, reg, reg_std, _, _ = find_best_tau(
        T=T, d=d, K=K, sigma=sigma, eta=np.sqrt(np.log(d)/T)*2, n_seeds=5
    )
    Path = 2.0 * K
    x = np.sqrt(Path * T * np.log(d))
    results_right.append({
        "T": T, "d": d, "Path": Path, "x": x,
        "regret": reg, "regret_std": reg_std, "tau_star": tau_star
    })

# ----------------------------------------------------------------------
# 6. Plots (matches the caption of the scaling figure)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- Left: tau* vs Path ----
ax = axes[0]
markers = {2000: "o", 5000: "s"}
colors  = {0.3: "C0", 0.5: "C1"}

for T in Ts:
    for sigma in sigmas:
        pts = [r for r in results_left if r["T"]==T and r["sigma"]==sigma]
        Paths = [p["Path"] for p in pts]
        taus  = [p["tau_star"] for p in pts]
        ax.plot(Paths, taus, marker=markers[T], color=colors[sigma],
                label=f"T={T}, σ={sigma}", linewidth=2, markersize=7)

        # theoretical reference ~ c * sqrt(T / Path)  (vertically shifted)
        P_ref = np.array(Paths)
        theory = 0.7 * np.sqrt(T / P_ref) * (0.5/sigma)   # rough vertical offset
        ax.plot(P_ref, theory, "--", color=colors[sigma], alpha=0.5, linewidth=1.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Path length $P_T$", fontsize=12)
ax.set_ylabel(r"Empirical optimal horizon $\tau^\star$", fontsize=12)
ax.set_title(r"Left: $\tau^\star$ tracks $\sqrt{T/P_T}$", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, which="both", ls="--", alpha=0.4)

# ---- Right: regret vs sqrt(Path T log d) ----
ax = axes[1]
xs, regs, regstds = [], [], []
for r in results_right:
    xs.append(r["x"])
    regs.append(r["regret"])
    regstds.append(r["regret_std"])

xs = np.array(xs)
regs = np.array(regs)
regstds = np.array(regstds)

ax.errorbar(xs, regs, yerr=regstds, fmt="o", color="C2",
            markersize=8, capsize=4, label="Optimally-tuned Discounted Hedge")

# linear reference
slope = np.polyfit(xs, regs, 1)[0]
x_line = np.linspace(xs.min()*0.9, xs.max()*1.05, 100)
ax.plot(x_line, slope * x_line, "--", color="gray",
        label=rf"slope $\approx {slope:.2f}$  ($\propto\sqrt{{P_T T\log d}}$)")

ax.set_xlabel(r"$\sqrt{P_T\, T\, \log d}$", fontsize=12)
ax.set_ylabel("Dynamic regret (optimally tuned)", fontsize=12)
ax.set_title("Right: minimax rate of Corollary 1", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("scaling_experiments.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved figure → scaling_experiments.png")
print("Done.")
