import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 140})

arch = json.load(open("results_arch.json"))["architectures"]
fc = json.load(open("fc_multistep.json"))

# ---------- Figure: architecture horizon sweeps ----------
fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
titles = {"ssm": "(a) Diagonal SSM (S4-style)", "gru": "(b) GRU update gate",
          "lstm": "(c) LSTM forget gate", "tcn": "(d) TCN receptive field"}
order = ["ssm", "gru", "lstm", "tcn"]
for ax, a in zip(axes, order):
    for reg, lab, c in [("lowdrift_highnoise", "low drift, high noise", "C3"),
                        ("highdrift_lownoise", "high drift, low noise", "C0")]:
        d = arch[a][reg]
        taus = np.array(d["taus"]); m = np.array(d["mean"]); sd = np.array(d["std"])
        ax.plot(taus, m, "-o", ms=4, color=c, label=lab)
        ax.fill_between(taus, m - sd, m + sd, color=c, alpha=0.12)
        ax.axvline(d["tau_star"], color=c, ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"memory horizon $\tau$")
    ax.set_title(titles[a], fontsize=10)
    if a == "ssm":
        ax.set_ylabel("held-out MSE")
    ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_architectures.pdf"); plt.close()

# ---------- Figure: real-data multi-step forecasting ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
panels = [("temperatures", "Daily-min-temperatures"),
          ("sunspots", "Monthly sunspots")]
for a, (key, title) in zip(ax, panels):
    for H, lab, c in [("H1", "1-step ahead", "C0"), ("H12", "12-step ahead", "C3")]:
        d = fc[key][H]
        taus = np.array(d["taus"]); m = np.array(d["mean"]); sd = np.array(d["std"])
        mn = m / m.min()
        a.plot(taus, mn, "-o", ms=4, color=c, label=lab)
        a.fill_between(taus, mn - sd/m.min(), mn + sd/m.min(), color=c, alpha=0.12)
        a.axvline(d["tau_star"], color=c, ls=":", lw=1)
    a.set_xscale("log", base=2)
    a.set_xlabel(r"memory horizon $\tau$")
    a.set_ylabel("normalized test MSE")
    a.set_title(title, fontsize=10)
    a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_forecasting.pdf"); plt.close()
print("architecture + forecasting figures written")
