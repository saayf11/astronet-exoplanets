"""Constructed dataset for COMP8240: synthetic Kepler-like TCEs with labels known by construction.

Light curves come from simulate.py (see its docstring for v1 vs v2). Each curve then goes
through AstroNet's OWN preprocessing (astronet/data/preprocess.py): spline detrending ->
phase folding -> 2001-bin global view + 201-bin local view, written as tf.train.Examples with
the same schema as the DR24 TFRecords (av_training_set = PC / AFP / NTP) plus the injection
parameters, so the DR24-trained model reads them directly.

Run from exoplanet-ml/exoplanet-ml in the TF1.15 env (PYTHONPATH=.):
  python /path/to/build_injection_dataset.py --version 2 --n_per_type 200 --out_dir out_v2
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate  # noqa: E402
from astronet.data import preprocess  # noqa: E402
from tf_util import example_util  # noqa: E402

VERSION = [1]


def make_example(job):
  all_t, all_f, tce, info = simulate.simulate(*job, version=VERSION[0])
  tt, ff = preprocess.process_light_curve(all_t, all_f)
  ex = preprocess.generate_example_for_tce(tt, ff, tce)
  example_util.set_bytes_feature(ex, "av_training_set", [info["label"]])
  example_util.set_bytes_feature(ex, "inj_type", [info["type"]])
  example_util.set_int64_feature(ex, "inj_idx", [info["idx"]])
  for key in ("snr", "depth_ppm", "k", "b", "sigma_ppm"):
    example_util.set_float_feature(ex, "inj_" + key, [info[key]])
  example_util.set_float_feature(ex, "inj_dilution", [info.get("dilution", 1.0)])
  return ex.SerializeToString(), info


def _init(v):
  VERSION[0] = v


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--version", type=int, default=2)
  ap.add_argument("--n_per_type", type=int, default=200)
  ap.add_argument("--out_dir", required=True)
  ap.add_argument("--workers", type=int, default=2)
  ap.add_argument("--seed", type=int, default=8240)
  f = ap.parse_args()
  os.makedirs(f.out_dir, exist_ok=True)
  jobs = simulate.jobs(f.n_per_type, seed=f.seed, version=f.version)
  rows = []
  path = os.path.join(f.out_dir, "injected_v%d-00000-of-00001" % f.version)
  with tf.io.TFRecordWriter(path) as w, mp.Pool(f.workers, _init, (f.version,)) as pool:
    for n, (rec, info) in enumerate(pool.imap(make_example, jobs, chunksize=4)):
      w.write(rec); rows.append(info)
      if (n + 1) % 50 == 0:
        print("%d / %d" % (n + 1, len(jobs)), flush=True)
  keys = sorted({k for r in rows for k in r})
  with open(os.path.join(f.out_dir, "injections_v%d.csv" % f.version), "w", newline="") as fh:
    cw = csv.DictWriter(fh, fieldnames=keys); cw.writeheader(); cw.writerows(rows)
  print("wrote", len(rows), "examples to", path)


if __name__ == "__main__":
  main()
