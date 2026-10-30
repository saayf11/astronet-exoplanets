"""Injection-recovery analysis of AstroNet on the constructed dataset.

Reads the per-example predictions from score_records.py and reports:
  * overall binary metrics (PC = 1, AFP/NTP = 0) at threshold 0.5, same convention as the paper
  * planet recall as a function of injected SNR (the injection-recovery curve)
  * how often each false-positive type is (wrongly) called a planet
and writes a two-panel figure for the slides.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

preds = sys.argv[1] if len(sys.argv) > 1 else "data/preds_v1.csv"
fig_path = sys.argv[2] if len(sys.argv) > 2 else "../figures/injection_recovery.png"
rows = list(csv.DictReader(open(preds)))
p = np.array([float(r["pred_planet_prob"]) for r in rows])
y = np.array([r["av_training_set"] == "PC" for r in rows]).astype(int)
ty = np.array([r["inj_type"] for r in rows])
snr = np.array([float(r["inj_snr"]) for r in rows])


def auc(y, s):
  order = np.argsort(s)
  ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
  npos = y.sum(); nneg = len(y) - npos
  return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


yhat = (p > 0.5).astype(int)
tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
print("N = %d  (PC %d, non-PC %d)" % (len(y), y.sum(), len(y) - y.sum()))
print("accuracy %.3f  AUC %.3f  precision %.3f  recall %.3f" %
      ((tp + tn) / len(y), auc(y, p), tp / max(tp + fp, 1), tp / max(tp + fn, 1)))
print("confusion: TP %d  FN %d  FP %d  TN %d" % (tp, fn, fp, tn))

bins = [5, 7.1, 15, 30, 60, 120, 1e9]
labels = ["5-7.1", "7.1-15", "15-30", "30-60", "60-120", ">120"]
print("\nPlanet recall by injected SNR (threshold 0.5):")
rec = []
for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
  m = (ty == "planet") & (snr >= lo) & (snr < hi)
  r = yhat[m].mean() if m.sum() else np.nan
  rec.append((lab, r, int(m.sum())))
  print("  SNR %-7s  n=%3d  recall %.2f" % (lab, m.sum(), r))

print("\nFraction called 'planet' (p > 0.5), per signal type:")
fooled = []
for t in ["eb_secondary", "eb_oddeven", "grazing_eb", "none", "planet"]:
  m = ty == t
  fooled.append((t, yhat[m].mean(), int(m.sum())))
  print("  %-13s n=%3d  %.3f   (median p = %.3f)" % (t, m.sum(), yhat[m].mean(), np.median(p[m])))

names = {"eb_secondary": "EB +\nsecondary", "eb_oddeven": "EB odd/\neven depth",
         "grazing_eb": "Grazing\nEB", "none": "No signal", "planet": "Planet\n(recall)"}
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
lab, r, n = zip(*rec)
ax[0].bar(range(len(lab)), r, color="#1f5aa6")
for i, (ri, ni) in enumerate(zip(r, n)):
  ax[0].text(i, ri + 0.02, "%.2f\n(n=%d)" % (ri, ni), ha="center", fontsize=7.5)
ax[0].set_xticks(range(len(lab))); ax[0].set_xticklabels(lab)
ax[0].set_ylim(0, 1.15); ax[0].set_xlabel("injected transit SNR"); ax[0].set_ylabel("fraction recovered")
ax[0].set_title("Planet recall vs SNR (injection-recovery)")
t, f, n = zip(*fooled)
cols = ["#d9822b"] * 4 + ["#1f5aa6"]
ax[1].bar(range(len(t)), f, color=cols)
for i, (fi, ni) in enumerate(zip(f, n)):
  ax[1].text(i, fi + 0.02, "%.2f" % fi, ha="center", fontsize=8)
ax[1].set_xticks(range(len(t))); ax[1].set_xticklabels([names[x] for x in t], fontsize=8)
ax[1].set_ylim(0, 1.1); ax[1].set_ylabel("fraction classified as planet")
ax[1].set_title("What AstroNet calls a planet, by true type")
plt.suptitle("AstroNet (trained on Kepler DR24) on the constructed synthetic-injection dataset, N=%d" % len(y), fontsize=10)
plt.tight_layout(); plt.savefig(fig_path, dpi=140)
print("\nfigure ->", fig_path)
