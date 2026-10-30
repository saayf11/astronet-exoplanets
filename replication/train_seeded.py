"""Run AstroNet's UNMODIFIED astronet/train.py with a fixed TensorFlow graph seed.

astronet.train builds tf.estimator.RunConfig(keep_checkpoint_max=1) without a seed, so every run
starts from a different random initialisation. To measure run-to-run variance reproducibly we
wrap RunConfig to add tf_random_seed=<seed>, then call train.main with the usual flags.
(Input shuffling is not seeded by AstroNet and stays random.)

From exoplanet-ml/exoplanet-ml, TF1.15 env, PYTHONPATH=.:
  python /path/to/repo/replication/train_seeded.py --seed 1 --model=AstroCNNModel \
      --config_name=local_global --train_files='tfrecord/train*' --eval_files='tfrecord/val*' \
      --model_dir=runs/seed1
"""
import argparse
import functools
import sys

import tensorflow as tf

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--seed", type=int, required=True)
seed_args, rest = pre.parse_known_args()
tf.estimator.RunConfig = functools.partial(tf.estimator.RunConfig, tf_random_seed=seed_args.seed)

from astronet import train  # noqa: E402  (imported after the patch)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  train.FLAGS, unparsed = train.parser.parse_known_args(rest)
  tf.app.run(main=train.main, argv=[sys.argv[0]] + unparsed)
