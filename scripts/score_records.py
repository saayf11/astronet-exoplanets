"""Score a TFRecord file with a trained AstroNet checkpoint and dump per-example predictions.

Output CSV has one row per example, in file order: the model's planet probability plus every
scalar feature in the record (label, injection parameters, TCE period, kepid, ...).

Usage (from exoplanet-ml/exoplanet-ml, TF1.15 env, PYTHONPATH=.):
  python /path/to/repo/scripts/score_records.py --model_dir model \
      --records out_v2/injected_v2-00000-of-00001 --out preds_v2.csv
"""
import argparse
import csv

import tensorflow as tf

from astronet import models
from astronet.util import estimator_util
from tf_util import configdict


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model_dir", required=True)
  ap.add_argument("--records", required=True)
  ap.add_argument("--out", required=True)
  f = ap.parse_args()

  config = configdict.ConfigDict(models.get_model_config("AstroCNNModel", "local_global"))
  estimator = estimator_util.create_estimator(
      models.get_model_class("AstroCNNModel"), config.hparams, model_dir=f.model_dir)
  input_fn = estimator_util.create_input_fn(
      file_pattern=f.records, input_config=config.inputs, mode=tf.estimator.ModeKeys.EVAL)
  probs = [float(p[0]) for p in estimator.predict(input_fn)]

  rows = []
  for rec in tf.compat.v1.io.tf_record_iterator(f.records):
    ex = tf.train.Example.FromString(rec)
    row = {}
    for name, feat in ex.features.feature.items():
      if name in ("global_view", "local_view"):
        continue
      kind = feat.WhichOneof("kind")
      vals = getattr(feat, kind).value
      if len(vals) == 1:
        v = vals[0]
        row[name] = v.decode() if isinstance(v, bytes) else v
    rows.append(row)
  assert len(rows) == len(probs), (len(rows), len(probs))
  for r, p in zip(rows, probs):
    r["pred_planet_prob"] = p
  keys = sorted({k for r in rows for k in r})
  with open(f.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
  print("scored", len(rows), "examples ->", f.out)


if __name__ == "__main__":
  main()
