"""Threshold ("Robovetter-lite") baseline on the same diagnostics the LLM judge sees.
Writes verdicts in the judge's JSON format, so analyze_judge.py treats both the same way.

  python rule_baseline.py --diag diag_v1.csv --key judge_inputs/key.csv --out rule_outputs.json
"""
import argparse
import json

import pandas as pd


def rule(r):
  if not r.depth_snr >= 7.1:
    return "NTP"
  if (r.odd_even_sigma > 3 or r.secondary_sigma > 4 or r.shape_inner_over_full > 1.4
      or r.implied_radius_earth > 22):
    return "AFP"
  return "PC"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--diag", required=True)
  ap.add_argument("--key", required=True)
  ap.add_argument("--out", default="rule_outputs.json")
  f = ap.parse_args()
  df = pd.read_csv(f.key).merge(pd.read_csv(f.diag), on="idx")
  v = [{"id": r.id, "label": rule(r), "confidence": 5, "reason": "threshold rule"} for r in df.itertuples()]
  json.dump({"model": "rule-baseline", "verdicts": v}, open(f.out, "w"), indent=1)
  print("wrote", len(v), "verdicts ->", f.out)


if __name__ == "__main__":
  main()
