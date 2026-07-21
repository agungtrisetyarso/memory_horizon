
import json, math, time
import numpy as np
import torch
import torch.nn as nn

SEED = 20260721
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(1)


# ----------------------------------------------------------------------
# Synthetic non-stationary sequence task: predict next target value of a
# piecewise-constant (switching) signal from noisy observations.
# switches ~ P (drift), noise ~ sigma.
# ----------------------------------------------------------------------
def make_stream(T, K, sigma, rng, d=1):
    seg = max(1, T // K)
    y = np.zeros((T, d), np.float32)
    for k in range(0, T, seg):
        y[k:k+seg] = rng.uniform(-1, 1, size=d)
    x = y + sigma * rng.standard_normal((T, d)).astype(np.float32)
    return x, y


def windowed(x, y, L):
    """Build (N, L, d)->(N, d) one-step-ahead prediction pairs."""
    T = x.shape[0]
    Xs, Ys = [], []
    for t in range(L, T):
        Xs.append(x[t-L:t])
        Ys.append(y[t])           # predict clean target at t
    return (torch.tensor(np.stack(Xs)), torch.tensor(np.stack(Ys)))


# ----------------------------------------------------------------------
# Architectures with an exposed, freezable horizon knob
# ----------------------------------------------------------------------
class LSTMHorizon(nn.Module):
    """Standard LSTM; forget-gate bias frozen to set the memory time-constant."""
    def __init__(self, d=1, h=32, forget_bias=1.0, train_bias=False):
        super().__init__()
        self.cell = nn.LSTM(d, h, batch_first=True)
        self.readout = nn.Linear(h, d)
        self.h = h
        with torch.no_grad():
            # bias_ih + bias_hh; forget gate is the 2nd quarter of the 4*h bias
            for name in ("bias_ih_l0", "bias_hh_l0"):
                b = getattr(self.cell, name)
                b[h:2*h].fill_(forget_bias / 2.0)
        if not train_bias:
            self._freeze_forget()

    def _freeze_forget(self):
        # register hook to zero grads on the forget-gate bias slice
        h = self.h
        def hook_factory():
            def hook(grad):
                g = grad.clone(); g[h:2*h] = 0.0; return g
            return hook
        self.cell.bias_ih_l0.register_hook(hook_factory())
        self.cell.bias_hh_l0.register_hook(hook_factory())

    def forget_tau(self):
        h = self.h
        bf = (self.cell.bias_ih_l0[h:2*h] + self.cell.bias_hh_l0[h:2*h]).mean().item()
        g = 1 / (1 + math.exp(-bf))
        return 1.0 / max(1e-3, 1 - g)

    def forward(self, x):
        out, _ = self.cell(x)
        return self.readout(out[:, -1])


class GRUHorizon(nn.Module):
    def __init__(self, d=1, h=32, update_bias=1.0):
        super().__init__()
        self.cell = nn.GRU(d, h, batch_first=True)
        self.readout = nn.Linear(h, d)
        with torch.no_grad():
            for name in ("bias_ih_l0", "bias_hh_l0"):
                b = getattr(self.cell, name)
                b[h:2*h].fill_(update_bias / 2.0)  # update gate = 2nd quarter (r,z,n)
        hh = h
        def hook(grad):
            g = grad.clone(); g[hh:2*hh] = 0.0; return g
        self.cell.bias_ih_l0.register_hook(hook)
        self.cell.bias_hh_l0.register_hook(hook)
        self.h = h

    def update_tau(self):
        h = self.h
        bz = (self.cell.bias_ih_l0[h:2*h] + self.cell.bias_hh_l0[h:2*h]).mean().item()
        z = 1 / (1 + math.exp(-bz))
        return 1.0 / max(1e-3, 1 - z)

    def forward(self, x):
        out, _ = self.cell(x)
        return self.readout(out[:, -1])


class DiagSSM(nn.Module):
    """S4-style diagonal SSM in leaky mode: h_t = a h_{t-1} + (1-a) B x_t,
    with a = exp(-exp(logdecay)) fixed to set the horizon tau = 1/(1-a)."""
    def __init__(self, d=1, h=32, tau=8.0):
        super().__init__()
        a = 1 - 1.0 / tau
        self.register_buffer("logdecay", torch.tensor(math.log(-math.log(a))))
        self.B = nn.Linear(d, h, bias=False)
        self.readout = nn.Linear(h, d)
        self.h = h

    def tau_val(self):
        a = math.exp(-math.exp(self.logdecay.item()))
        return 1.0 / (1 - a)

    def forward(self, x):
        # x: (N, L, d); vectorized decay-weighted cumulative sum
        a = torch.exp(-torch.exp(self.logdecay))
        N, L, d = x.shape
        proj = self.B(x)                        # (N, L, h)
        idx = torch.arange(L, dtype=torch.float32)
        diff = idx.view(L, 1) - idx.view(1, L)  # weight[t,s] for s<=t
        W = (1 - a) * torch.pow(a, torch.clamp(diff, min=0.0)) * (diff >= 0).float()
        hstate = torch.einsum("ts,nsh->nth", W, proj)  # (N,L,h)
        return self.readout(hstate[:, -1])


class TCNHorizon(nn.Module):
    """Temporal conv net; receptive field (=horizon) set by dilation depth."""
    def __init__(self, d=1, h=32, levels=3, k=2):
        super().__init__()
        layers = []
        cin = d
        for i in range(levels):
            dil = 2 ** i
            pad = (k - 1) * dil
            layers += [nn.Conv1d(cin, h, k, padding=pad, dilation=dil), nn.ReLU()]
            cin = h
        self.net = nn.ModuleList(layers)
        self.readout = nn.Linear(h, d)
        self.rf = 1 + sum((k - 1) * 2 ** i for i in range(levels))
        self.pads = [(k - 1) * 2 ** i for i in range(levels)]

    def tau_val(self):
        return float(self.rf)

    def forward(self, x):
        # x: (N, L, d) -> (N, d, L)
        z = x.transpose(1, 2)
        li = 0
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                pad = self.pads[li]; li += 1
                z = layer(z)
                if pad:
                    z = z[:, :, :-pad]          # causal crop
            else:
                z = layer(z)
        return self.readout(z[:, :, -1])


# ----------------------------------------------------------------------
# Train / eval one model on a stream, return held-out MSE
# ----------------------------------------------------------------------
def train_eval(model, Xtr, Ytr, Xte, Yte, steps=120, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad()
        pred = model(Xtr)
        loss = lossf(pred, Ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        te = lossf(model(Xte), Yte).item()
    return te


# ----------------------------------------------------------------------
# Experiment A: horizon sweep per architecture, two regimes
#   regime 1: low drift + high noise  -> long optimal horizon
#   regime 2: high drift + low noise  -> short optimal horizon
# ----------------------------------------------------------------------
def sweep_architecture(arch, taus, regime, seeds=3, T=1000, L=32):
    K, sigma = regime
    curve = np.zeros((seeds, len(taus)))
    for s in range(seeds):
        rng = np.random.default_rng(SEED + 31 * s + K)
        x, y = make_stream(T, K, sigma, rng)
        split = int(0.7 * (T - L))
        X, Y = windowed(x, y, L)
        Xtr, Ytr, Xte, Yte = X[:split], Y[:split], X[split:], Y[split:]
        for j, tau in enumerate(taus):
            torch.manual_seed(SEED + s)
            if arch == "lstm":
                bf = math.log(1 / (1 - (1 - 1.0 / tau)) - 1) if tau > 1 else 0.0
                # invert tau=1/(1-sigmoid(bf)) -> sigmoid(bf)=1-1/tau
                g = 1 - 1.0 / tau
                bf = math.log(g / (1 - g))
                m = LSTMHorizon(forget_bias=2 * bf)
            elif arch == "gru":
                g = 1 - 1.0 / tau; bz = math.log(g / (1 - g))
                m = GRUHorizon(update_bias=2 * bz)
            elif arch == "ssm":
                m = DiagSSM(tau=tau)
            elif arch == "tcn":
                # tau ~ receptive field; pick levels achieving ~tau
                levels = max(1, int(round(math.log2(max(2, tau)))))
                m = TCNHorizon(levels=levels)
                tau = m.tau_val()
            curve[s, j] = train_eval(m, Xtr, Ytr, Xte, Yte)
    mean = curve.mean(0)
    return {"taus": [float(t) for t in taus],
            "mean": mean.tolist(), "std": curve.std(0).tolist(),
            "tau_star": float(taus[int(np.argmin(mean))]),
            "K": K, "sigma": sigma}


def exp_architectures():
    out = {}
    regimes = {"lowdrift_highnoise": (6, 0.8),   # long horizon expected
               "highdrift_lownoise": (60, 0.3)}  # short horizon expected
    tau_grid = [2, 4, 8, 16, 32, 64]
    for arch in ["lstm", "gru", "ssm"]:
        out[arch] = {}
        ns = 3 if arch == "ssm" else 3
        for rname, reg in regimes.items():
            t0 = time.time()
            out[arch][rname] = sweep_architecture(arch, tau_grid, reg, seeds=ns)
            print(f"  {arch}/{rname}: tau*={out[arch][rname]['tau_star']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # TCN uses discrete receptive fields
    out["tcn"] = {}
    tcn_taus = [2, 4, 8, 16, 32]
    for rname, reg in regimes.items():
        out["tcn"][rname] = sweep_architecture("tcn", tcn_taus, reg)
        print(f"  tcn/{rname}: tau*={out['tcn'][rname]['tau_star']}")
    return out


if __name__ == "__main__":
    res = {}
    print("architecture horizon sweeps...")
    res["architectures"] = exp_architectures()
    json.dump(res, open("results_arch.json", "w"), indent=2)
    print("saved results_arch.json")
