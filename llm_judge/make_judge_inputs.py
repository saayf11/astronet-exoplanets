"""Build blind LLM-as-judge inputs from the constructed dataset.

Draws a stratified random sample (default 20 per injected type = 100 TCEs), shuffles it,
assigns anonymous IDs (J001...), and writes:
  judge_inputs/prompt.md          the judge instructions + rubric (system prompt)
  judge_inputs/batch_<n>.txt      25 summaries per batch (paste into a FRESH chat, or use the API)
  judge_inputs/key.csv            HIDDEN answer key: anon id -> idx, true type/label, AstroNet prob
  judge_inputs/hand_labels.csv    blank sheet for the human (student) labels, same IDs

Usage: python make_judge_inputs.py --diag diag_v1.csv --preds ../constructed_dataset/preds.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from vetting_diagnostics import summary_text

PROMPT = """You are an expert exoplanet vetter for the Kepler mission. For each Threshold Crossing
Event (TCE) below you are given vetting diagnostics measured from its light curve. Classify
each TCE into exactly one class:

  PC  - planet candidate: a transit-like signal consistent with a planet orbiting the star
  AFP - astrophysical false positive: a real signal, but from an eclipsing binary (e.g.
        significant odd/even depth difference, a significant secondary eclipse, a V-shaped
        or grazing eclipse, or an implausibly large companion)
  NTP - non-transiting phenomenon: no convincing transit (low SNR, noise or variability)

Guidance (Kepler Robovetter conventions):
- A detection SNR below ~7 is weak evidence of any signal.
- An odd/even difference above ~3 sigma suggests an eclipsing binary at twice the period.
- A secondary dip above ~3-4 sigma near phase 0.5 suggests an eclipsing binary.
  (With ~150 trial phases, the strongest dip from pure noise is typically 2-3 sigma.)
- Shape ratio near 1.0-1.3 means flat-bottomed (U, planet-like); clearly higher means V-shaped.
- Companions larger than ~22 R_earth (~2 Jupiter radii) are unlikely to be planets, but
  blended (diluted) binaries can look planet-sized, so use the other diagnostics too.

Reply with ONLY a JSON list, one object per TCE, in the order given:
[{"id": "J001", "label": "PC|AFP|NTP", "confidence": 1-5, "reason": "<one sentence>"}, ...]
"""


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--diag", required=True)
  ap.add_argument("--preds", required=True)
  ap.add_argument("--injections", required=True)
  ap.add_argument("--per_type", type=int, default=20)
  ap.add_argument("--batch", type=int, default=25)
  ap.add_argument("--seed", type=int, default=8240)
  ap.add_argument("--out", default="judge_inputs")
  f = ap.parse_args()
  diag = pd.read_csv(f.diag)
  inj = pd.read_csv(f.injections)[["idx", "type", "label"]]
  preds = pd.read_csv(f.preds)[["inj_idx", "pred_planet_prob"]].rename(columns={"inj_idx": "idx"})
  df = diag.merge(inj, on="idx").merge(preds, on="idx")
  rng = np.random.RandomState(f.seed)
  sample = pd.concat([g.sample(f.per_type, random_state=rng) for _, g in df.groupby("type")])
  sample = sample.sample(frac=1, random_state=rng).reset_index(drop=True)
  sample["id"] = ["J%03d" % (i + 1) for i in range(len(sample))]

  os.makedirs(f.out, exist_ok=True)
  with open(os.path.join(f.out, "prompt.md"), "w") as fh:
    fh.write(PROMPT)
  for b in range(0, len(sample), f.batch):
    part = sample.iloc[b:b + f.batch]
    text = "\n\n".join(summary_text(r["id"], r) for _, r in part.iterrows())
    with open(os.path.join(f.out, "batch_%d.txt" % (b // f.batch + 1)), "w") as fh:
      fh.write(text + "\n")
  sample[["id", "idx", "type", "label", "pred_planet_prob"]].to_csv(
      os.path.join(f.out, "key.csv"), index=False)
  pd.DataFrame({"id": sample.id, "label": "", "notes": ""}).to_csv(
      os.path.join(f.out, "hand_labels.csv"), index=False)
  print("wrote %d TCEs in %d batches to %s" % (len(sample), -(-len(sample) // f.batch), f.out))


if __name__ == "__main__":
  main()
