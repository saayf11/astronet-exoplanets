"""Bar chart of the repeated-CV results in results/snr_experiments.txt."""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/snr_experiments.txt"
out = sys.argv[2] if len(sys.argv) > 2 else "../figures/kaggle_snr_experiments.png"
cv = open(src).read().split("Repeated stratified")[1]
rows = re.findall(r"  ([A-F] [^\n]+?)\s+AUC ([\d.]+) \+/- ([\d.]+)\s+AP ([\d.]+) \+/- ([\d.]+)", cv)
names = [r[0] for r in rows][::-1]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (i, lab, ref) in zip(axes, [(1, "ROC AUC", 0.5), (3, "Average precision", 0.0074)]):
  v = [float(r[i]) for r in rows][::-1]; e = [float(r[i + 1]) for r in rows][::-1]
  ax.barh(names, v, xerr=e, color=["#d9822b" if n.startswith("D") else "#4a78a8" for n in names])
  ax.axvline(ref, color="grey", ls="--", lw=1)
  ax.set_xlabel(lab + " (mean ± sd, 10×5-fold CV)")
axes[1].text(0.0074, -0.9, " chance", color="grey", fontsize=8)
fig.suptitle("Kaggle: adding BLS SNR to AstroNet (42 hosts / 5,657 stars)")
fig.tight_layout(); fig.savefig(out, dpi=150); print("figure ->", out)
