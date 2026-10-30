"""AstroNet (trained on Kepler DR24) applied to Kaggle "Exoplanet Hunting in Deep Space".

Inputs: per-star predictions from score_records.py over all Kaggle shards.
Reports AUC, precision/recall at 0.5, precision@k, where the confirmed-planet hosts rank,
and how the result depends on BLS finding a real signal; writes a slide figure.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

preds = sys.argv[1] if len(sys.argv) > 1 else "data/kaggle_preds.csv"
fig_path = sys.argv[2] if len(sys.argv) > 2 else "../figures/kaggle_results.png"
d = pd.read_csv(preds)
y = (d.kaggle_label == 2).astype(int).values
p = d.pred_planet_prob.values


def auc(y, s):
  order = np.argsort(s, kind="mergesort")
  r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
  npos = y.sum(); nneg = len(y) - npos
  return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def report(mask, name):
  yy, pp = y[mask], p[mask]
  yh = pp > 0.5
  tp = int((yh & (yy == 1)).sum()); fp = int((yh & (yy == 0)).sum())
  fn = int((~yh & (yy == 1)).sum()); tn = int((~yh & (yy == 0)).sum())
  rank = np.argsort(-pp)
  hosts_rank = np.sort(np.where(yy[rank] == 1)[0] + 1)
  print("\n== %s: N=%d, hosts=%d (base rate %.4f)" % (name, len(yy), yy.sum(), yy.mean()))
  print("AUC %.3f | at p>0.5: TP %d FN %d FP %d TN %d | recall %.2f precision %.3f"
        % (auc(yy, pp), tp, fn, fp, tn, tp / max(tp + fn, 1), tp / max(tp + fp, 1)))
  for k in (10, 50, 100):
    print("  precision@%d = %.2f (%d hosts in top %d)" % (k, yy[rank[:k]].mean(), yy[rank[:k]].sum(), k))
  print("  host ranks:", hosts_rank.tolist())
  return hosts_rank


print("Planet probability by label:")
print(d.groupby("kaggle_label").pred_planet_prob.describe()[["count", "mean", "50%", "max"]].round(3))
report(np.ones(len(y), bool), "ALL (train+test)")
report((d.kaggle_split == "test").values, "Kaggle TEST split only")

# Does AstroNet add anything over the BLS detection statistic alone?
print("\nAUC of BLS depth-SNR alone: %.3f   AUC of AstroNet: %.3f" % (auc(y, d.bls_snr.values), auc(y, p)))

# Stratify by how strong the BLS signal was.
q = np.quantile(d.bls_snr, [0, .5, .9, 1.0])
for lo, hi, lab in [(q[0], q[1], "weakest 50%"), (q[1], q[2], "50-90th pct"), (q[2], q[3] + 1, "strongest 10%")]:
  m = (d.bls_snr >= lo) & (d.bls_snr < hi)
  print("  BLS SNR %-14s n=%4d hosts=%2d  frac p>0.5 = %.2f (non-hosts), %.2f (hosts)" % (
      lab, m.sum(), y[m].sum(), (p[m & (y == 0)] > .5).mean(),
      (p[m & (y == 1)] > .5).mean() if y[m].sum() else float("nan")))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
bins = np.linspace(0, 1, 21)
ax[0].hist(p[y == 0], bins=bins, color="#9aa5b1", density=True, label="no planet (n=%d)" % (y == 0).sum())
ax[0].hist(p[y == 1], bins=bins, color="#1f5aa6", density=True, alpha=0.8, label="confirmed host (n=%d)" % y.sum())
ax[0].set_xlabel("AstroNet planet probability"); ax[0].set_ylabel("density (per class)")
ax[0].set_title("Score distribution by true label"); ax[0].legend(fontsize=8)

order = np.argsort(-p)
cum = np.cumsum(y[order])
ax[1].plot(np.arange(1, len(y) + 1), cum / y.sum(), color="#1f5aa6", label="AstroNet (AUC %.2f)" % auc(y, p))
o2 = np.argsort(-d.bls_snr.values)
ax[1].plot(np.arange(1, len(y) + 1), np.cumsum(y[o2]) / y.sum(), color="#d9822b", ls="--",
           label="BLS SNR alone (AUC %.2f)" % auc(y, d.bls_snr.values))
ax[1].plot([1, len(y)], [0, 1], color="#bbb", lw=1, label="random")
ax[1].set_xlabel("stars reviewed, ranked by score")
ax[1].set_ylabel("fraction of planet hosts found"); ax[1].set_title("Ranking the 5,657 Kaggle stars")
ax[1].legend(fontsize=8, loc="upper left")
plt.suptitle("AstroNet (trained on Kepler DR24) on Kaggle 'Exoplanet Hunting in Deep Space' — no retraining", fontsize=10)
plt.tight_layout(); plt.savefig(fig_path, dpi=140)
print("\nfigure ->", fig_path)
