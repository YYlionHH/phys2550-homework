"""
PHYS 2550 Homework 1 - full analysis pipeline.
Linear regression from scratch: NumPy + matplotlib only.
Run:  python hw1_analysis.py
Outputs: figures/fig_*.png and results.json
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng_note = "deterministic, no RNG used"
np.set_printoptions(suppress=True)

# ---------------------------------------------------------------- (a) data
df = pd.read_csv("real_estate.csv")
print("shape:", df.shape)
print(df.head(), "\n")

cols = {
    "X1 transaction date": "transaction date (year + fraction of year, e.g. 2013.250 = March 2013)",
    "X2 house age": "house age (years)",
    "X3 distance to the nearest MRT station": "distance to nearest MRT station (metres)",
    "X4 number of convenience stores": "number of convenience stores within walking distance (count)",
    "X5 latitude": "latitude (degrees)",
    "X6 longitude": "longitude (degrees)",
    "Y house price of unit area": "house price per unit area (10,000 TWD / ping, 1 ping = 3.3 m^2)",
}
for c, desc in cols.items():
    print(f"{desc:75s}  range [{df[c].min():.3f}, {df[c].max():.3f}]  mean {df[c].mean():.3f}")
print("missing values:", int(df.isna().sum().sum()))

feat_cols = list(cols.keys())[:-1]
target = "Y house price of unit area"

fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
for ax, c in zip(axes.flat, [target] + feat_cols):
    ax.hist(df[c], bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(c.replace("X", "X"), fontsize=9)
fig.suptitle("(a) Histograms of target and features")
fig.tight_layout()
fig.savefig("figures/fig_a_histograms.png", dpi=150, bbox_inches="tight")
plt.close(fig)

y = df[target].to_numpy(float)
N = len(y)

# ------------------------------------------------- (b) one feature: house age
x_b = df["X2 house age"].to_numpy(float)

def mse(pred, y):
    return np.mean((pred - y) ** 2)

# L(w0,w1) = (1/N) sum (w0 + w1 x - y)^2
# dL/dw0 = (2/N) sum (yhat - y) ;  dL/dw1 = (2/N) sum (yhat - y) x
def gd_single(x, y, lr, epochs):
    w0, w1 = 0.0, 0.0
    hist = []
    for ep in range(epochs):
        pred = w0 + w1 * x
        err = pred - y
        hist.append(mse(pred, y))
        w0 -= lr * 2.0 * err.mean()
        w1 -= lr * 2.0 * (err * x).mean()
    return w0, w1, np.array(hist)

lr_b, ep_b = 0.001, 50000
w0_b, w1_b, hist_b = gd_single(x_b, y, lr_b, ep_b)
loss_b = hist_b[-1]
print(f"\n(b) lr={lr_b}, epochs={ep_b}  ->  w0={w0_b:.4f}, w1={w1_b:.4f}, MSE={loss_b:.4f}")
print(f"    (lr=0.01 diverges on this raw feature; lr=0.001 is stable)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist_b, color="#4C72B0")
ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss (log scale)")
ax.set_title(f"(b) Gradient descent, lr={lr_b}")
fig.tight_layout(); fig.savefig("figures/fig_b_loss.png", dpi=150, bbox_inches="tight"); plt.close(fig)

pred_b = w0_b + w1_b * x_b
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(x_b, y, s=12, alpha=0.6, color="#4C72B0", label="data")
xs = np.linspace(x_b.min(), x_b.max(), 100)
axes[0].plot(xs, w0_b + w1_b * xs, color="#DD8452", lw=2,
             label=f"fit: y = {w0_b:.2f} + {w1_b:.3f} x")
axes[0].set_xlabel("house age (years)"); axes[0].set_ylabel("price per unit area")
axes[0].legend(); axes[0].set_title("(b) data and fitted line")
axes[1].scatter(y, pred_b, s=12, alpha=0.6, color="#4C72B0")
lims = [0, max(y.max(), pred_b.max()) * 1.05]
axes[1].plot(lims, lims, "k--", lw=1, label="perfect prediction")
axes[1].set_xlabel("true price"); axes[1].set_ylabel("predicted price")
axes[1].legend(); axes[1].set_title("(b) predicted vs true")
fig.tight_layout(); fig.savefig("figures/fig_b_scatter.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# --------------------------- (c) multi-feature, vectorised, raw vs normalised
use = ["X2 house age", "X3 distance to the nearest MRT station",
       "X4 number of convenience stores"]
X_raw = df[use].to_numpy(float)                      # (N, 3)
mu, sd = X_raw.mean(0), X_raw.std(0)
X_norm = (X_raw - mu) / sd

def design(X):
    return np.column_stack([np.ones(len(X)), X])     # append constant 1

def gd_vec(X, y, lr, epochs):
    """Gradient descent; stops and returns the finite prefix if the loss
    blows up past 1e15 (divergence), so histories stay plottable."""
    Xd = design(X)
    w = np.zeros(Xd.shape[1])
    hist = []
    for ep in range(epochs):
        err = Xd @ w - y
        loss = np.mean(err ** 2)
        if not np.isfinite(loss) or loss > 1e15:     # diverged: stop early
            break
        hist.append(loss)
        w -= lr * 2.0 * (Xd.T @ err) / len(y)
    return w, np.array(hist)

lr_c, ep_c = 0.01, 5000
lr_div = 1e-6
w_raw_bad, hist_raw_bad = gd_vec(X_raw, y, lr_div, ep_c)    # unstable on raw -> diverges
w_raw_slow, hist_raw_slow = gd_vec(X_raw, y, 1e-7, ep_c)    # tiny lr -> crawls
w_norm, hist_norm = gd_vec(X_norm, y, lr_c, ep_c)           # normalised -> fine
loss_c = hist_norm[-1]
div_ep = len(hist_raw_bad)
print(f"\n(c) features: {use}")
print(f"    raw, lr={lr_div}: loss blew past 1e15 by epoch {div_ep} (diverges)")
print(f"    raw, lr=1e-7: final loss = {hist_raw_slow[-1]:.2f} after {ep_c} epochs (still far from optimum)")
print(f"    normalised, lr={lr_c}: w = {np.round(w_norm,4)}, MSE = {loss_c:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(np.arange(div_ep), hist_raw_bad, color="#C44E52", lw=1.5,
        label=f"raw features, lr={lr_div:g} (diverges)")
ax.plot(hist_raw_slow, color="#DD8452", lw=1.5, label="raw features, lr=1e-7 (crawls)")
ax.plot(hist_norm, color="#4C72B0", lw=1.5, label=f"normalised, lr={lr_c}")
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss (log scale)")
ax.set_title("(c) Gradient descent: raw vs normalised features")
ax.legend()
fig.tight_layout(); fig.savefig("figures/fig_c_loss.png", dpi=150, bbox_inches="tight"); plt.close(fig)

pred_c = design(X_norm) @ w_norm

# ------------------------------------------------------------- (d) compare
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y, pred_b, s=14, alpha=0.6, color="#DD8452", label=f"(b) age only, MSE={loss_b:.2f}")
ax.scatter(y, pred_c, s=14, alpha=0.6, color="#4C72B0", label=f"(c) 3 features, MSE={loss_c:.2f}")
lims = [0, max(y.max(), pred_c.max()) * 1.05]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlabel("true price"); ax.set_ylabel("predicted price")
ax.set_title("(d) predicted vs true, both models")
ax.legend()
fig.tight_layout(); fig.savefig("figures/fig_d_pred_vs_true.png", dpi=150, bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, c in zip(axes.flat, feat_cols):
    ax.scatter(df[c], y, s=10, alpha=0.55, color="#4C72B0")
    r = np.corrcoef(df[c], y)[0, 1]
    ax.set_xlabel(c, fontsize=8); ax.set_ylabel("price")
    ax.set_title(f"corr = {r:+.2f}", fontsize=9)
fig.suptitle("(d) price vs each feature")
fig.tight_layout(); fig.savefig("figures/fig_d_price_vs_features.png", dpi=150, bbox_inches="tight"); plt.close(fig)

corr = {c: float(np.corrcoef(df[c], y)[0, 1]) for c in feat_cols}

# ---------------------------------------------------------- bonus: normal eq
Xd = design(X_norm)
w_star = np.linalg.solve(Xd.T @ Xd, Xd.T @ y)         # (X^T X)^-1 X^T y
loss_star = mse(Xd @ w_star, y)
print(f"\nbonus: normal equation w* = {np.round(w_star,6)}, MSE = {loss_star:.6f}")
print(f"       GD (c) reached w = {np.round(w_norm,6)}, MSE = {loss_c:.6f}")
print(f"       |w_GD - w*|_max = {np.max(np.abs(w_norm - w_star)):.2e}")

# how many epochs for GD to get within 1e-4 (relative) of the closed-form loss?
tol = 1e-4
need = next(i for i, l in enumerate(hist_norm) if (l - loss_star) / loss_star < tol)
print(f"       GD within 0.01% of closed-form loss after {need} epochs")

# closed form for the single-feature model, for reference
Xd_b = design(x_b.reshape(-1, 1))
w_star_b = np.linalg.solve(Xd_b.T @ Xd_b, Xd_b.T @ y)
loss_b_star = mse(Xd_b @ w_star_b, y)
print(f"       (b) closed form: w0={w_star_b[0]:.4f}, w1={w_star_b[1]:.4f}, "
      f"MSE={loss_b_star:.4f} (GD got {w0_b:.4f}, {w1_b:.4f}, MSE={loss_b:.4f})")

results = {
    "N": int(N),
    "lr_b": lr_b, "ep_b": ep_b, "w0_b": float(w0_b), "w1_b": float(w1_b),
    "loss_b": float(loss_b),
    "w0_b_star": float(w_star_b[0]), "w1_b_star": float(w_star_b[1]),
    "loss_b_star": float(loss_b_star),
    "features_c": use, "lr_c": lr_c, "ep_c": ep_c, "lr_div": lr_div,
    "w_norm": w_norm.tolist(), "loss_c": float(loss_c),
    "raw_diverge_epoch": int(div_ep),
    "hist_raw_slow_final": float(hist_raw_slow[-1]),
    "w_star": w_star.tolist(), "loss_star": float(loss_star),
    "w_diff_max": float(np.max(np.abs(w_norm - w_star))),
    "epochs_to_closed_form": int(need),
    "corr": corr,
    "col_stats": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                      "mean": float(df[c].mean())} for c in [target] + feat_cols},
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved results.json and figures/fig_*.png")
