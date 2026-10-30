#!/usr/bin/env bash
# Rebuilds the TF1.15 environment used for this replication (Linux; ~10 min).
# AstroNet needs TensorFlow 1.15 (tf.contrib), which needs Python <= 3.7.
set -e
git clone --depth 1 --branch v3.7.17 https://github.com/python/cpython.git cpython37
(cd cpython37 && ./configure --prefix="$PWD/../py37" && make -j"$(nproc)" && make install)
py37/bin/python3.7 -m venv tf1 && . tf1/bin/activate
pip install "pip<24"
pip install tensorflow==1.15.5 protobuf==3.20.3 "h5py<3" astropy pydl==0.7.0 absl-py matplotlib batman-package
# pydl 0.7.0 bug fix: cholesky_band's LinAlgError fallback returns a padded array.
F=$(python -c "import pydl.pydlutils.bspline as b; print(b.__file__)")
sed -i 's/^    L\[:, 0:n\] = lower$/    L[:, 0:n] = lower[:, 0:n]  # patch/' "$F"
git clone --depth 1 https://github.com/google-research/exoplanet-ml.git
