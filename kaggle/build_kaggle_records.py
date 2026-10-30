"""Existing new dataset: Kaggle "Exoplanet Hunting in Deep Space" -> AstroNet inputs.

https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data
Each row is one star: LABEL (2 = confirmed-planet host, 1 = not) + 3,197 flux values at
Kepler long cadence (29.4 min, ~65 days). There are no timestamps and no TCE ephemeris,
and the flux is already centred near zero, so we:

  1. build a time axis from the cadence;
  2. clip upward outliers (> median + 5 MAD; the data has cosmic-ray-like spikes);
  3. detrend with AstroNet's own robust spline (third_party/kepler_spline), SUBTRACTING
     the spline because flux is not normalised (AstroNet's views are rescaled to
     median 0 / min -1 anyway, so this gives the same kind of input);
  4. search for a transit with Box Least Squares (astropy), periods 0.5 d to baseline/2
     (at least 2 transits) -> the strongest signal's period, t0 and duration play the
     role of the TCE;
  5. build global/local views with AstroNet's generate_example_for_tce and write TFRecords
     with av_training_set = PC (LABEL 2) or NTP (LABEL 1), plus the BLS statistics.

Usage (exoplanet-ml/exoplanet-ml, TF1.15 env, PYTHONPATH=.):
  python /path/to/repo/kaggle/build_kaggle_records.py --data_dir ~/kaggle_exo --out_dir records
"""
import argparse
import csv
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.timeseries import BoxLeastSquares

from astronet.data import preprocess
from tf_util import example_util
from third_party.kepler_spline import kepler_spline

CADENCE = 29.4 / 60 / 24
DURATIONS = np.array([0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32])


def process(args):
  split, row, label, flux = args
  t = np.arange(flux.size) * CADENCE
  med = np.median(flux)
  mad = 1.4826 * np.median(np.abs(flux - med))
  keep = flux < med + 5 * mad
  t, y = t[keep], flux[keep].astype(float)

  spline = kepler_spline.fit_kepler_spline([t], [y], verbose=False)[0][0]
  ok = np.isfinite(spline)
  t, r = t[ok], (y - spline)[ok]

  baseline = t[-1] - t[0]
  bls = BoxLeastSquares(t, r)
  periods = np.exp(np.linspace(np.log(0.5), np.log(baseline / 2), 4000))
  res = bls.power(periods, DURATIONS, objective="snr")
  i = int(np.argmax(res.power))
  period, t0, dur = float(res.period[i]), float(res.transit_time[i]), float(res.duration[i])

  tce = {"tce_period": period, "tce_duration": dur, "tce_time0bk": t0}
  ex = preprocess.generate_example_for_tce(t, r, tce)
  example_util.set_bytes_feature(ex, "av_training_set", ["PC" if label == 2 else "NTP"])
  example_util.set_bytes_feature(ex, "kaggle_split", [split])
  example_util.set_int64_feature(ex, "kaggle_row", [row])
  example_util.set_int64_feature(ex, "kaggle_label", [label])
  example_util.set_float_feature(ex, "bls_snr", [float(res.depth_snr[i])])
  example_util.set_float_feature(ex, "bls_power", [float(res.power[i])])
  info = dict(split=split, row=row, label=label, period=period, t0=t0, duration=dur,
              bls_depth_snr=float(res.depth_snr[i]), bls_power=float(res.power[i]),
              n_transits=int(baseline // period))
  return ex.SerializeToString(), info


def all_jobs(limit, data_dir):
  out = []
  for split in ("train", "test"):
    d = pd.read_csv(os.path.join(data_dir, "exo%s.csv" % split.capitalize()))
    if limit:
      d = d.iloc[:limit]
    vals = d.iloc[:, 1:].values
    for i, lab in enumerate(d.LABEL.values):
      out.append((split, i, int(lab), vals[i]))
  return out


def main():
  """Writes one shard per CHUNK stars and skips shards already on disk, so a killed run resumes."""
  import os
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", required=True, help="folder with exoTrain.csv / exoTest.csv")
  ap.add_argument("--out_dir", required=True)
  ap.add_argument("--limit", type=int, default=0, help="rows per split (0 = all)")
  ap.add_argument("--workers", type=int, default=2)
  ap.add_argument("--chunk", type=int, default=250)
  f = ap.parse_args()
  os.makedirs(f.out_dir, exist_ok=True)
  jobs = all_jobs(f.limit, f.data_dir)
  n_shards = (len(jobs) + f.chunk - 1) // f.chunk
  with mp.Pool(f.workers) as pool:
    for s in range(n_shards):
      path = os.path.join(f.out_dir, "kaggle-%05d-of-%05d" % (s, n_shards))
      if os.path.exists(path + ".bls.csv"):
        continue
      rows = []
      with tf.io.TFRecordWriter(path + ".tmp") as w:
        for rec, info in pool.imap(process, jobs[s * f.chunk:(s + 1) * f.chunk], chunksize=8):
          w.write(rec); rows.append(info)
      os.rename(path + ".tmp", path)
      with open(path + ".bls.csv", "w", newline="") as fh:
        cw = csv.DictWriter(fh, fieldnames=list(rows[0])); cw.writeheader(); cw.writerows(rows)
      print("shard %d/%d done" % (s + 1, n_shards), flush=True)
  print("all shards done", flush=True)


if __name__ == "__main__":
  main()
