import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 140})

ush = json.load(open("r_ushape.json"))["ushape"]
scal = json.load(open("r_scaling.json"))
noise = json.load(open("r_noise.json"))
forget = json.load(open("r_forget.json"))
meta = json.load(open("r_meta.json"))
cont = json.load(open("r_contdepth.json"))
base = json.load(open("r_baselines.json"))

# ---------- Figure 1: U-shape + scaling law ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
taus = np.array(ush["taus"])
for name, lab, c in [("switching", "switching, $\\sigma$=0.5", "C0"),
                     ("switching_highnoise", "switching, $\\sigma$=0.9", "C3"),
                     ("drifting", "drifting, $\\sigma$=0.5", "C2")]:
    m = np.array(ush[name]["mean"]); sd = np.array(ush[name]["std"])
    ax[0].plot(taus, m, "-o", ms=3, color=c, label=lab)
    ax[0].fill_between(taus, m - sd/np.sqrt(15), m + sd/np.sqrt(15), color=c, alpha=0.15)
    ts = ush[name]["tau_star"]
    ax[0].axvline(ts, color=c, ls=":", lw=1)
ax[0].set_xscale("log"); ax[0].set_xlabel(r"memory horizon $\tau=1/\gamma_0$")
ax[0].set_ylabel("dynamic regret"); ax[0].set_title("(a) U-shape in the memory horizon")
ax[0].legend(fontsize=8)

P = np.array(scal["P_T"]); ts = np.array(scal["tau_star"])
ax[1].loglog(P, ts, "ko", ms=6, label="empirical $\\tau^\\star$")
b, a = scal["fit_exponent"], scal["fit_intercept"]
xx = np.linspace(P.min(), P.max(), 50)
ax[1].loglog(xx, np.exp(a) * xx**b, "r-", label=f"fit $\\propto P^{{{b:.2f}}}$")
ax[1].loglog(xx, ts[0]*(xx/P[0])**-0.5, "b--", alpha=.6, label="ideal $P^{-0.5}$")
ax[1].set_xlabel("path length $P_T$"); ax[1].set_ylabel(r"optimal horizon $\tau^\star$")
ax[1].set_title("(b) $\\tau^\\star$ decreases with drift"); ax[1].legend(fontsize=8)

sg = np.array(noise["sigma"]); tn = np.array(noise["tau_star"]); tsd = np.array(noise["tau_std"])
ax[2].errorbar(sg, tn, yerr=tsd/np.sqrt(12), fmt="ks-", capsize=3, label="empirical")
ax[2].plot(sg, tn[0]*(sg/sg[0]), "b--", alpha=.6, label="linear-in-$\\sigma$ guide")
ax[2].set_xlabel("noise scale $\\sigma$"); ax[2].set_ylabel(r"optimal horizon $\tau^\star$")
ax[2].set_title("(c) $\\tau^\\star$ increases with noise"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_horizon_law.pdf"); plt.close()

# ---------- Figure 2: forget-gate equivalence ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ft = np.array(forget["taus"])
labels = {"lowP_lowN": ("P=14, $\\sigma$=0.5", "C0"),
          "lowP_highN": ("P=14, $\\sigma$=1.0", "C3"),
          "highP_lowN": ("P=126, $\\sigma$=0.5", "C2")}
for k, (lab, c) in labels.items():
    m = np.array(forget[k]["mean"])
    ax[0].plot(ft, m/m.min(), "-o", ms=3, color=c, label=lab)
    ax[0].axvline(forget[k]["tau_star"], color=c, ls=":", lw=1)
ax[0].set_xscale("log"); ax[0].set_xlabel(r"gate horizon $\tau=1/(1-g)$")
ax[0].set_ylabel("normalized MSE"); ax[0].set_title("(a) Forget-gate MSE is U-shaped")
ax[0].legend(fontsize=8)

# bar of tau* showing monotone shifts
names = ["highP_lowN", "lowP_lowN", "lowP_highN"]
vals = [forget[n]["tau_star"] for n in names]
cols = [labels[n][1] for n in names]
ax[1].bar([labels[n][0] for n in names], vals, color=cols, alpha=.75)
ax[1].set_ylabel(r"optimal gate horizon $\tau^\star$")
ax[1].set_title("(b) Optimal gate follows the law")
plt.setp(ax[1].get_xticklabels(), rotation=15, ha="right", fontsize=8)
plt.tight_layout(); plt.savefig("fig_forget_gate.pdf"); plt.close()

# ---------- Figure 3: meta-learned horizon + continuous depth ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
methods = ["oracle_const", "adaptive", "doubling"]
disp = {"oracle_const": "Oracle-tuned\n$\\gamma_0^\\star$",
        "adaptive": "Meta-learned\n$\\gamma_0(t)$ (ours)",
        "doubling": "Doubling\nschedule"}
means = [meta[m][0] for m in methods]
stds = [meta[m][1] for m in methods]
cols2 = ["#888", "C0", "C1"]
ax[0].bar([disp[m] for m in methods], means, yerr=np.array(stds)/np.sqrt(8),
          capsize=4, color=cols2, alpha=.8)
for i, m in enumerate(methods):
    ov = "" if m == "oracle_const" else f"{meta[means.index(meta[m][0]) == i and 'overhead_adaptive' or 'overhead_doubling']:.2f}$\\times$" if False else ""
ax[0].text(1, means[1]+8, f"{meta['overhead_adaptive']:.2f}$\\times$", ha="center", fontsize=10)
ax[0].text(2, means[2]+8, f"{meta['overhead_doubling']:.2f}$\\times$", ha="center", fontsize=10)
ax[0].set_ylabel("dynamic regret (test episodes)")
ax[0].set_title("(a) Meta-learned horizon ~ oracle")

lt = np.array(cont["learned_tau"]); ot = np.array(cont["optimal_tau"])
ax[1].scatter(ot, lt, c="C2", s=60, zorder=3)
lim = [min(lt.min(), ot.min())-2, max(lt.max(), ot.max())+2]
ax[1].plot(lim, lim, "k--", alpha=.5, label="$y=x$")
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel(r"grid-optimal horizon $\tau^\star$")
ax[1].set_ylabel(r"learned horizon $\tau_{\rm learned}$")
ax[1].set_title("(b) Learnable-decay cell finds $\\tau^\\star$")
ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("fig_meta_horizon.pdf"); plt.close()

print("figures written")
print("meta overhead adaptive=%.3f doubling=%.3f" % (meta["overhead_adaptive"], meta["overhead_doubling"]))
