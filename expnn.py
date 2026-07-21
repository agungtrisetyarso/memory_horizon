
import json
import numpy as np
import torch
import torch.nn as nn

SEED = 20260721
torch.manual_seed(SEED)
np.random.seed(SEED)


# ======================================================================
# (A) Forget-gate equivalence
# ======================================================================
def leaky_average(x, g):
    """Causal leaky integrator with forget gate g in [0,1):
       h_t = g h_{t-1} + (1-g) x_t.  h is the gated running average."""
    T = x.shape[0]
    h = np.zeros_like(x[0])
    out = np.zeros_like(x)
    for t in range(T):
        h = g * h + (1 - g) * x[t]
        out[t] = h
    return out


def exp_forget_gate(seeds=20):
    """Task: track a piecewise-constant target signal corrupted by noise.
    The learner outputs a gated running average; MSE-to-target is U-shaped
    in the gate's effective horizon tau = 1/(1-g).  Optimal tau should
    follow sqrt(sigma^2 T / P) with P the number of target switches."""
    T, d = 2000, 1
    gates = 1 - 1.0 / np.unique(np.round(np.logspace(np.log10(1.5), np.log10(300), 20)).astype(int))
    taus = 1.0 / (1 - gates)
    configs = [("lowP_lowN", 8, 0.5), ("lowP_highN", 8, 1.0),
               ("highP_lowN", 64, 0.5)]
    out = {"taus": taus.tolist()}
    for name, K, sigma in configs:
        curve = np.zeros((seeds, len(gates)))
        for s in range(seeds):
            rng = np.random.default_rng(SEED + 13 * s + K)
            seg = T // K
            target = np.zeros((T, d))
            for k in range(K):
                target[k * seg:(k + 1) * seg] = rng.uniform(0, 1)
            x = target + sigma * rng.standard_normal((T, d))
            for j, g in enumerate(gates):
                est = leaky_average(x, g)
                curve[s, j] = float(np.mean((est - target) ** 2))
        mean = curve.mean(0)
        out[name] = {"mean": mean.tolist(), "std": curve.std(0).tolist(),
                     "tau_star": float(taus[int(np.argmin(mean))]),
                     "P": 2 * (K - 1), "sigma": sigma}
    return out


# ======================================================================
# (B) Meta-learned adaptive horizon
# ======================================================================
class HorizonNet(nn.Module):
    """Reads (log drift estimate, log noise estimate, log t) -> log gamma_0."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))

    def forward(self, feat):
        return self.net(feat)  # log gamma_0


def make_switching(T, d, K, sigma, rng):
    seg = T // K
    best = rng.integers(0, d, size=K)
    clean = np.full((T, d), 0.6)
    for k in range(K):
        clean[k * seg:(k + 1) * seg, best[k]] = 0.2
    obs = clean + sigma * rng.standard_normal((T, d))
    return clean, obs


def run_disc_hedge_regret(obs, clean, eta, gamma_traj):
    """gamma_traj: array of per-round discount rates."""
    T, d = obs.shape
    E = np.zeros(d)
    reg = 0.0
    best = clean.min(1)
    for t in range(T):
        g = gamma_traj[t]
        E = (1 - g) * E + obs[t]
        w = np.exp(-eta * (E - E.min()))
        p = w / w.sum()
        reg += float(p @ clean[t] - best[t])
    return reg


def running_estimates(obs, win=50):
    """Cheap online drift & noise estimates from observed losses."""
    T, d = obs.shape
    drift = np.zeros(T)
    noise = np.zeros(T)
    buf = []
    for t in range(T):
        buf.append(obs[t])
        if len(buf) > win:
            buf.pop(0)
        arr = np.array(buf)
        if len(buf) > 2:
            # drift: movement of argmin over the window
            am = arr.argmin(1)
            drift[t] = np.mean(am[1:] != am[:-1]) + 1e-3
            # noise: mean coordinate std within window
            noise[t] = arr.std(0).mean() + 1e-3
        else:
            drift[t] = 1e-2
            noise[t] = 0.5
    return drift, noise


def exp_meta_horizon(seeds=8, epochs=150):
    """Train HorizonNet across many switching episodes to output gamma_0(t)
    minimizing dynamic regret, then test against oracle-constant and doubling."""
    d, T, eta = 10, 1500, 1.0
    net = HorizonNet()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)

    # ---- training: differentiable surrogate via REINFORCE-free direct search ----
    # We train the net by matching its output to the per-episode oracle gamma_0*,
    # which we compute by grid search (supervised meta-regression). Honest label:
    # this is supervised meta-learning of the horizon law, not RL.
    taus_grid = np.unique(np.round(np.logspace(np.log10(2), np.log10(250), 14)).astype(int))
    train_feats, train_targets = [], []
    rng = np.random.default_rng(SEED)
    for ep in range(60):
        K = int(rng.integers(2, 80))
        sigma = float(rng.uniform(0.2, 1.0))
        clean, obs = make_switching(T, d, K, sigma, rng)
        drift, noise = running_estimates(obs)
        # oracle constant gamma_0*
        best_tau, best_r = None, np.inf
        for tau in taus_grid:
            r = run_disc_hedge_regret(obs, clean, eta, np.full(T, 1.0 / tau))
            if r < best_r:
                best_r, best_tau = r, tau
        # feature at mid-stream (drift/noise are slowly varying)
        for t in range(100, T, 200):
            f = [np.log(drift[t]), np.log(noise[t]), np.log(t + 1)]
            train_feats.append(f)
            train_targets.append(np.log(1.0 / best_tau))
    X = torch.tensor(np.array(train_feats), dtype=torch.float32)
    y = torch.tensor(np.array(train_targets), dtype=torch.float32).unsqueeze(1)
    for e in range(epochs):
        opt.zero_grad()
        pred = net(X)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
    final_train_loss = float(loss.item())

    # ---- test: compare adaptive net vs oracle-constant vs doubling ----
    def doubling_traj(obs, clean, eta):
        T, d = obs.shape
        gamma = 1.0 / 4.0
        traj = np.zeros(T)
        E = np.zeros(d); epoch_loss = 0.0; epoch_len = 0
        thresh = 0.15
        for t in range(T):
            traj[t] = gamma
            E = (1 - gamma) * E + obs[t]
            w = np.exp(-eta * (E - E.min())); p = w / w.sum()
            epoch_loss += float(p @ obs[t]); epoch_len += 1
            if epoch_len > 100 and epoch_loss / epoch_len > thresh:
                gamma = min(0.5, gamma * 2.0)
                epoch_loss = 0.0; epoch_len = 0; E = np.zeros(d)
        return traj

    rows = {"adaptive": [], "oracle_const": [], "doubling": []}
    for s in range(seeds):
        rng = np.random.default_rng(SEED + 999 + s)
        K = int(rng.integers(2, 80)); sigma = float(rng.uniform(0.2, 1.0))
        clean, obs = make_switching(T, d, K, sigma, rng)
        drift, noise = running_estimates(obs)
        feats = np.stack([np.log(drift), np.log(noise),
                          np.log(np.arange(T) + 1)], 1)
        with torch.no_grad():
            logg = net(torch.tensor(feats, dtype=torch.float32)).numpy().ravel()
        gamma_traj = np.clip(np.exp(logg), 1.0 / 250, 0.5)
        rows["adaptive"].append(run_disc_hedge_regret(obs, clean, eta, gamma_traj))
        # oracle constant
        br = min(run_disc_hedge_regret(obs, clean, eta, np.full(T, 1.0 / tau))
                for tau in taus_grid)
        rows["oracle_const"].append(br)
        rows["doubling"].append(
            run_disc_hedge_regret(obs, clean, eta, doubling_traj(obs, clean, eta)))
    summary = {m: [float(np.mean(v)), float(np.std(v))] for m, v in rows.items()}
    summary["overhead_adaptive"] = float(np.mean(rows["adaptive"]) / np.mean(rows["oracle_const"]))
    summary["overhead_doubling"] = float(np.mean(rows["doubling"]) / np.mean(rows["oracle_const"]))
    summary["train_loss"] = final_train_loss
    return summary


# ======================================================================
# (C) Continuous-depth recurrent memory layer
# ======================================================================
class LeakyMemoryCell(nn.Module):
    """Learnable-decay leaky integrator = discretized dissipative filter.
       h_t = sigmoid(rho) * h_{t-1} + (1 - sigmoid(rho)) * W x_t
       rho is a single learnable scalar => learnable memory horizon."""
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.W = nn.Linear(d_in, d_hidden, bias=False)
        self.readout = nn.Linear(d_hidden, d_in)
        self.rho = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 => tau=2

    def forward(self, x):
        # x: (T, d_in). Vectorized leaky integrator:
        #   h_t = g h_{t-1} + (1-g) proj_t
        #       = (1-g) sum_{s<=t} g^{t-s} proj_s
        g = torch.sigmoid(self.rho)
        T = x.shape[0]
        proj = self.W(x)                       # (T, H)
        idx = torch.arange(T, dtype=torch.float32).unsqueeze(1)  # (T,1)
        # weight matrix L[t,s] = (1-g) g^{t-s} for s<=t else 0
        diff = idx - idx.transpose(0, 1)       # (T,T) = t - s
        mask = (diff >= 0).float()
        L = (1 - g) * torch.pow(g, torch.clamp(diff, min=0.0)) * mask
        h = L @ proj                           # (T, H)
        outs = self.readout(h)                 # (T, d_in)
        return outs, g


def exp_continuous_depth(seeds=6, steps=250):
    """Non-stationary smoothing/denoising task: recover a piecewise-constant
    vector target from noisy observations. Train the leaky cell (decay learnable)
    and read off the learned horizon tau = 1/(1-g); compare to the grid-optimal
    horizon for the same stream."""
    d, T = 4, 800
    learned, optimal = [], []
    for s in range(seeds):
        rng = np.random.default_rng(SEED + 4242 + s)
        K = 16; sigma = 0.6; seg = T // K
        target = np.zeros((T, d))
        for k in range(K):
            target[k * seg:(k + 1) * seg] = rng.uniform(0, 1, size=d)
        x = target + sigma * rng.standard_normal((T, d))
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(target, dtype=torch.float32)
        model = LeakyMemoryCell(d, 16)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(steps):
            opt.zero_grad()
            pred, g = model(xt)
            loss = ((pred - yt) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            g = torch.sigmoid(model.rho).item()
        learned.append(1.0 / (1 - g))
        # grid-optimal horizon (pure leaky average, MSE)
        gates = 1 - 1.0 / np.unique(np.round(np.logspace(np.log10(1.5), np.log10(200), 20)).astype(int))
        mses = []
        for gg in gates:
            h = np.zeros(d); est = np.zeros_like(x)
            for t in range(T):
                h = gg * h + (1 - gg) * x[t]
                est[t] = h
            mses.append(np.mean((est - target) ** 2))
        opt_tau = 1.0 / (1 - gates[int(np.argmin(mses))])
        optimal.append(float(opt_tau))
    return {"learned_tau": [float(v) for v in learned],
            "optimal_tau": [float(v) for v in optimal],
            "learned_mean": float(np.mean(learned)),
            "optimal_mean": float(np.mean(optimal)),
            "corr": float(np.corrcoef(learned, optimal)[0, 1]) if len(learned) > 1 else None}


if __name__ == "__main__":
    out = {}
    print("(A) forget gate..."); out["forget_gate"] = exp_forget_gate()
    print("(C) continuous depth..."); out["continuous_depth"] = exp_continuous_depth()
    print("(B) meta horizon..."); out["meta_horizon"] = exp_meta_horizon()
    json.dump(out, open("results_nn.json", "w"), indent=2)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if not isinstance(vv, list)})
                      for k, v in out.items()}, indent=2))
