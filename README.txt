COMP8240 replication demo: AstroNet on the Kepler DR24 test set
=================================================================

What it does
  Runs the ORIGINAL exoplanet-ml repo's evaluate script (astronet/evaluate.py, unmodified)
  with our trained model on the 1,574 held-out test TCEs, inside Google's official
  TensorFlow 1.15 Docker image. It takes about a minute and prints accuracy, AUC and the
  confusion matrix.

  Expected output: accuracy 0.963, AUC 0.990, TP 350 / FN 10 / FP 48 / TN 1166.

Contents
  code/        astronet, tf_util, light_curve, third_party from github.com/google-research/exoplanet-ml
  model/       our checkpoint after 625 training steps (weights only; optimizer state removed)
  data/        test-00000-of-00001: the README's pre-computed DR24 test TFRecord
  run_demo.sh  the one command to run

One-time setup (Mac)
  1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/  (Apple Silicon version)
  2. Open Docker Desktop and go to Settings > General. Tick
     "Use Rosetta for x86_64/amd64 emulation on Apple Silicon", then Apply & restart.
     TF 1.15 only exists for Intel CPUs, so the Mac emulates one.
  3. In Terminal, pull the image once (about 1 GB):
       docker pull --platform linux/amd64 tensorflow/tensorflow:1.15.5-py3

Run
  cd ~/Downloads/comp8240_demo
  ./run_demo.sh

If it says "Illegal instruction"
  The emulator lacks AVX, which TF 1.15 needs. Check that the Rosetta box from step 2 is ticked
  and that macOS is version 15 (Sequoia) or later.
