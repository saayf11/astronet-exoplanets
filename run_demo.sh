#!/usr/bin/env bash
# COMP8240 replication demo: evaluate our trained AstroNet model on the Kepler DR24 test set.
# Runs the ORIGINAL repo's evaluate script inside Google's official TensorFlow 1.15 Docker image.
# Usage:  cd comp8240_demo && ./run_demo.sh
set -uo pipefail
cd "$(dirname "$0")"
IMAGE=tensorflow/tensorflow:1.15.5-py3

# The model weights were shipped in two parts (file-size limit); join them once.
CKPT=model/model.ckpt-625.data-00000-of-00001
if [ ! -f "$CKPT" ] && ls model/model.ckpt-625.data.part_* >/dev/null 2>&1; then
  cat model/model.ckpt-625.data.part_* > "$CKPT" && rm model/model.ckpt-625.data.part_*
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found. Install Docker Desktop (https://www.docker.com/products/docker-desktop/), open it once, then rerun."; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed but not running. Open Docker Desktop, wait for it to say 'running', then rerun."; exit 1
fi

echo "== Environment: $IMAGE (Python 3.6, TensorFlow 1.15.5), linux/amd64"
echo "== Command: python -m astronet.evaluate --model=AstroCNNModel --config_name=local_global \\"
echo "             --eval_files=data/test-00000-of-00001 --model_dir=model"
echo

docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work/code "$IMAGE" \
  bash -c 'python -c "import tensorflow as tf; print(\"TensorFlow\", tf.__version__)" 2>/dev/null; \
           python -m astronet.evaluate --model=AstroCNNModel --config_name=local_global \
             --eval_files=../data/test-00000-of-00001 --model_dir=../model' \
  2>&1 | tee run_log.txt | grep -E "^TensorFlow|^INFO:tensorflow:(Building input pipeline|Restoring parameters|Starting evaluation at|Finished evaluation)" | sed 's/^INFO:tensorflow://' 
status=${PIPESTATUS[0]}
rm -rf model/eval_test

if [ "$status" -eq 132 ] || grep -q "Illegal instruction" run_log.txt; then
  echo; echo "CPU emulation lacks AVX. In Docker Desktop: Settings > General > tick"
  echo "'Use Rosetta for x86_64/amd64 emulation on Apple Silicon', Apply & restart, then rerun."
  echo "(Needs macOS 15 Sequoia or later for AVX support under Rosetta.)"; exit 1
fi

line=$(grep "Saving dict for global step" run_log.txt | tail -1)
if [ -z "$line" ]; then echo; echo "Evaluation did not finish; see run_log.txt"; exit 1; fi
get() { echo "$line" | grep -o "$1 = [0-9.]*" | awk '{print $3}'; }
tn=$(get label_0_pred_0); fp=$(get label_0_pred_1); fn=$(get label_1_pred_0); tp=$(get label_1_pred_1)
echo
echo "================  AstroNet on Kepler DR24 TEST set  ================"
printf " Test TCEs evaluated : %d\n" "$(get num_examples | cut -d. -f1)"
printf " Accuracy            : %.3f\n" "$(get accuracy/accuracy)"
printf " AUC                 : %.3f   (paper reports ~0.988)\n" "$(get auc)"
echo
printf "   %-18s %22s %20s\n" "" "predicted not planet" "predicted planet"
printf "   %-18s %22d %20d\n" "true not planet" "${tn%.*}" "${fp%.*}"
printf "   %-18s %22d %20d\n" "true planet" "${fn%.*}" "${tp%.*}"
echo "===================================================================="
echo "Full log saved to run_log.txt"
