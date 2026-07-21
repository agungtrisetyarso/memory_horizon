import json, math
import numpy as np
import torch
import torch.nn as nn
from exp_arch import LSTMHorizon, GRUHorizon, DiagSSM, TCNHorizon, train_eval

SEED = 20260721
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(1)


def load_series(path, col):
    import csv
    vals = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            try:
                vals.append(float(row[col]))
            except (ValueError, IndexError):
                pass
    x = np.array(vals, np.float32)
    x = (x - x.mean()) / (x.std() + 1e-8)   # standardize
    return x


def make_windows(x, L, horizon=1):
    T = len(x)
    Xs, Ys = [], []
    for t in range(L, T - horizon + 1):
        Xs.append(x[t-L:t][:, None])
        Ys.append(x[t + horizon - 1:t + horizon])   # value h steps ahead
    return torch.tensor(np.stack(Xs)), torch.tensor(np.stack(Ys))


def estimate_drift_noise(x, win=30):
    """Rough effective drift (local-level change) and noise (residual to a
    short moving average) of a real series, for ordering vs. the horizon law."""
    ma = np.convolve(x, np.ones(win)/win, mode="valid")
    drift = float(np.mean(np.abs(np.diff(ma))))          # how fast the level moves
    resid = x[win-1:] - ma
    noise = float(np.std(resid))                          # short-scale noise
    return drift, noise


def sweep_forecast(x, arch, taus, L=32, steps=120, seeds=3, max_windows=1200, horizon=1):
    X, Y = make_windows(x, L, horizon=horizon)
    n = len(X)
    if n > max_windows:
        X, Y = X[:max_windows], Y[:max_windows]; n = max_windows
    split = int(0.7 * n)
    Xtr, Ytr, Xte, Yte = X[:split], Y[:split], X[split:], Y[split:]
    curve = np.zeros((seeds, len(taus)))
    for s in range(seeds):
        for j, tau in enumerate(taus):
            torch.manual_seed(SEED + s)
            if arch == "lstm":
                g = 1 - 1.0/tau; bf = math.log(g/(1-g)); m = LSTMHorizon(forget_bias=2*bf)
            elif arch == "gru":
                g = 1 - 1.0/tau; bz = math.log(g/(1-g)); m = GRUHorizon(update_bias=2*bz)
            elif arch == "ssm":
                m = DiagSSM(tau=tau)
            elif arch == "tcn":
                levels = max(1, int(round(math.log2(max(2, tau)))))
                m = TCNHorizon(levels=levels); tau = m.tau_val()
            curve[s, j] = train_eval(m, Xtr, Ytr, Xte, Yte, steps=steps)
    mean = curve.mean(0)
    return {"taus": [float(t) for t in taus], "mean": mean.tolist(),
            "std": curve.std(0).tolist(),
            "tau_star": float(taus[int(np.argmin(mean))])}


if __name__ == "__main__":
    series = {
        "temperatures": load_series("daily-min-temperatures.csv", 1),
        "sunspots": load_series("sunspots.csv", 1),
    }
    out = {}
    for name, x in series.items():
        d, nz = estimate_drift_noise(x)
        out[name] = {"len": len(x), "drift": d, "noise": nz, "arch": {}}
        print(f"{name}: len={len(x)} drift={d:.4f} noise={nz:.3f}", flush=True)
        for arch in ["ssm", "gru"]:
            taus = [2, 4, 8, 16, 32, 64]
            r = sweep_forecast(x, arch, taus, seeds=2 if arch != "ssm" else 3)
            out[name]["arch"][arch] = r
            print(f"   {arch}: tau*={r['tau_star']} "
                  f"mse={[round(m,4) for m in r['mean']]}", flush=True)
    json.dump(out, open("results_forecast.json", "w"), indent=2)
    print("saved results_forecast.json")
