"""Agreement analysis: LLM judge vs ground truth vs AstroNet vs human labels.

Reports, on the blind sample:
  * 3-class (PC/AFP/NTP) accuracy, confusion matrix and Cohen's kappa: judge vs truth,
    human vs truth, judge vs human
  * binary planet-vs-not (AstroNet is binary): accuracy and kappa for all pairs
  * per-injected-type "called planet" rate for judge vs AstroNet (where each one fails)

Usage: python analyze_judge.py --key judge_inputs/key.csv --judge judge_outputs.json \
           [--human judge_inputs/hand_labels.csv]
"""
import argparse
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def kappa_row(name, a, b):
  return {"pair": name, "n": len(a), "agreement": (a == b).mean(), "cohen_kappa": cohen_kappa_score(a, b)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--key", required=True)
  ap.add_argument("--judge", required=True)
  ap.add_argument("--human")
  ap.add_argument("--out", default="judge_agreement.csv")
  f = ap.parse_args()
  key = pd.read_csv(f.key)
  j = pd.DataFrame(json.load(open(f.judge))["verdicts"])[["id", "label", "confidence"]]
  df = key.merge(j.rename(columns={"label": "judge"}), on="id", how="left")
  assert df.judge.notna().all(), "missing verdicts for %s" % list(df.id[df.judge.isna()])
  df["astronet_pc"] = df.pred_planet_prob > 0.5
  if f.human:
    h = pd.read_csv(f.human).rename(columns={"label": "human"})[["id", "human"]]
    df = df.merge(h, on="id")
    df = df[df.human.isin(["PC", "AFP", "NTP"])]

  labs = ["PC", "AFP", "NTP"]
  print("== 3-class, n=%d" % len(df))
  print("judge vs truth confusion (rows=truth PC/AFP/NTP, cols=judge):")
  print(confusion_matrix(df.label, df.judge, labels=labs))
  rows = [kappa_row("judge vs truth (3-class)", df.judge, df.label)]
  if f.human:
    rows += [kappa_row("human vs truth (3-class)", df.human, df.label),
             kappa_row("judge vs human (3-class)", df.judge, df.human)]
  b = lambda s: s == "PC"
  rows += [kappa_row("judge vs truth (planet/not)", b(df.judge), b(df.label)),
           kappa_row("AstroNet vs truth (planet/not)", df.astronet_pc, b(df.label)),
           kappa_row("judge vs AstroNet (planet/not)", b(df.judge), df.astronet_pc)]
  if f.human:
    rows += [kappa_row("human vs truth (planet/not)", b(df.human), b(df.label)),
             kappa_row("human vs AstroNet (planet/not)", b(df.human), df.astronet_pc)]
  res = pd.DataFrame(rows)
  print(res.round(3).to_string(index=False))
  res.to_csv(f.out, index=False)

  print("\n== fraction called 'planet' by injected type")
  t = df.groupby("type").agg(judge=("judge", lambda s: (s == "PC").mean()),
                             astronet=("astronet_pc", "mean"))
  if f.human:
    t["human"] = df.groupby("type").human.apply(lambda s: (s == "PC").mean())
  print(t.round(2))
  wrong = df[df.judge != df.label]
  print("\n== judge errors (%d):" % len(wrong))
  print(wrong[["id", "type", "label", "judge", "confidence"]].to_string(index=False))


if __name__ == "__main__":
  main()
