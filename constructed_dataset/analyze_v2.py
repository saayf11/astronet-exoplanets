"""Constructed dataset v2: recovery vs SNR and false-positive rates vs dilution.

v2 sets every signal's SNR exactly (log-uniform 3-100), so recovery is resolved around
Kepler's 7.1-sigma threshold. Compares the threshold vetting rule (llm_judge/rule_baseline.py)
and, when predictions are given, the DR24-trained AstroNet.

  python analyze_v2.py --inj data/injections_v2.csv --diag ../llm_judge/diag_v2.csv \
      [--preds data/preds_v2.csv] --fig ../figures/injection_recovery_v2.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm_judge"))
from rule_baseline import rule  # noqa: E402

BINS = [3, 5, 7.1, 10, 15, 25, 50, 100]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--inj", required=True)
  ap.add_argument("--diag", required=True)
  ap.add_argument("--preds")
  ap.add_argument("--fig")
  f = ap.parse_args()
  d = pd.read_csv(f.inj).merge(pd.read_csv(f.diag), on="idx")
  d["rule_pc"] = [rule(r) == "PC" for r in d.itertuples()]
  methods = {"threshold rule": "rule_pc"}
  if f.preds:
    p = pd.read_csv(f.preds)[["inj_idx", "pred_planet_prob"]].rename(columns={"inj_idx": "idx"})
    d = d.merge(p, on="idx"); d["astronet_pc"] = d.pred_planet_prob > 0.5
    methods["AstroNet"] = "astronet_pc"
  d["snr_bin"] = pd.cut(d.snr, BINS)

  print("== called 'planet', by type and true SNR bin")
  for name, col in methods.items():
    print("\n[%s]" % name)
    print(d.pivot_table(index="snr_bin", columns="type", values=col, aggfunc="mean").round(2))
  afp = d[d.label == "AFP"].copy()
  afp["dil_bin"] = pd.qcut(afp.dilution, 3, labels=["most diluted", "middle", "least diluted"])
  print("\nNTP (no signal) called planet: " + ", ".join(
      "%s %.3f" % (n, d[d.type == "none"][c].mean()) for n, c in methods.items()))
  print("\n== AFP called 'planet', by effective dilution tercile (SNR >= 7.1 only)")
  for name, col in methods.items():
    print("[%s]" % name)
    print(afp[afp.snr >= 7.1].pivot_table(index="dil_bin", columns="type", values=col,
                                          aggfunc="mean").round(2))

  if f.fig:
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 4), squeeze=False)
    mids = np.sqrt(np.array(BINS[:-1]) * np.array(BINS[1:]))
    for ax, (name, col) in zip(axes[0], methods.items()):
      for t, g in d[d.type != "none"].groupby("type"):
        r = g.groupby("snr_bin", observed=False)[col].mean().reindex(d.snr_bin.cat.categories).values
        ax.plot(mids, r, "o-", label=t)
      ax.axvline(7.1, color="grey", ls="--", lw=1)
      ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
      ax.set_xticks([3, 5, 7.1, 10, 25, 50, 100])
      ax.set_xticklabels(["3", "5", "7.1", "10", "25", "50", "100"]); ax.minorticks_off()
      ax.set_xlabel("injected SNR"); ax.set_ylabel("fraction called planet"); ax.set_title(name)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f.fig, dpi=150)
    print("figure ->", f.fig)


if __name__ == "__main__":
  main()
