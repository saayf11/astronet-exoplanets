"""Regenerate the constructed light curves (exactly, from the seed) and compute vetting
diagnostics for every TCE. No TensorFlow needed.

  python compute_diagnostics.py --version 1 --n_per_type 200 --out diag_v1.csv
"""
import argparse
import multiprocessing as mp
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "constructed_dataset"))
import simulate  # noqa: E402
import vetting_diagnostics as vd  # noqa: E402

VERSION = 1


def run(job):
  T, F, tce, info = simulate.simulate(*job, version=VERSION)
  d = vd.diagnostics(T, F, tce["tce_period"], tce["tce_time0bk"], tce["tce_duration"],
                     info["rstar"], info["mstar"])
  d["idx"] = info["idx"]
  return d


def main():
  global VERSION
  ap = argparse.ArgumentParser()
  ap.add_argument("--version", type=int, default=1)
  ap.add_argument("--n_per_type", type=int, default=200)
  ap.add_argument("--workers", type=int, default=2)
  ap.add_argument("--out", required=True)
  f = ap.parse_args()
  VERSION = f.version
  with mp.Pool(f.workers, initializer=globals().__setitem__, initargs=("VERSION", f.version)) as p:
    rows = p.map(run, simulate.jobs(f.n_per_type, version=f.version), chunksize=10)
  pd.DataFrame(rows).to_csv(f.out, index=False)
  print("wrote", len(rows), "rows ->", f.out)


if __name__ == "__main__":
  main()
