# Finding exoplanets with deep learning: replicating and extending AstroNet

I replicated **AstroNet** ([Shallue & Vanderburg, 2018](https://iopscience.iop.org/article/10.3847/1538-3881/aa9e09)), the convolutional neural network from Google Brain that found the exoplanets Kepler-90i and Kepler-80g in NASA Kepler data. Then I tested it on two datasets it had never seen, and compared it against explicit vetting diagnostics and an LLM-as-judge.

**Headline:** my retrained model matches the paper on 1,574 held-out Kepler signals: **AUC 0.990** (the paper reports 0.988) and **96.3% accuracy**. It recovered 350 of the 360 real planet candidates. Across 5 independently seeded training runs, the AUC is **0.989 ± 0.001**.

## The problem

Kepler watched about 200,000 stars for four years. An automated pipeline flags periodic dips in brightness, called "threshold crossing events" (TCEs), but most of those dips are not planets. They come from eclipsing binary stars, instrumental artefacts or background stars. AstroNet treats vetting as binary classification: given a signal, it outputs P(planet).

## Model

Each signal is phase-folded on its period into two views:

- **Global view:** the whole orbit, 2,001 bins, fed through 5 convolutional blocks (16 → 256 filters)
- **Local view:** a zoom on the transit, 201 bins, fed through 2 convolutional blocks (16 → 32 filters)

The two towers are concatenated, passed through 4 dense layers of 512 units, and end in a sigmoid that outputs P(planet).

## Replication results (Kepler DR24 test set)

| Metric | My run | 5 seeds (mean ± sd) | Published |
|---|---|---|---|
| AUC | **0.990** | 0.989 ± 0.001 | 0.988 (paper) |
| Accuracy | **0.963** | 0.961 ± 0.003 | 0.963 (repo) |
| Planets recovered | 350 / 360 | | 350 / 360 (repo) |
| False alarms | 48 / 1,214 | | 49 / 1,214 (repo) |

The model was trained from scratch on 12,589 expert-labelled signals: 625 steps, batch size 256, about 35 minutes on a CPU. I checked the result in two environments (Linux, and macOS via Docker) and got identical numbers. The extra seeds use `replication/train_seeded.py`, which runs the unmodified `astronet/train.py` with a fixed TensorFlow seed.

## Extensions: two datasets AstroNet had never seen

**1. Kaggle "Exoplanet Hunting in Deep Space"** (5,657 Kepler stars, of which only 0.74% host planets). This dataset has raw flux, no timestamps and no candidate periods, so I built a pipeline to turn it into AstroNet's input: spline detrending, then a box-least-squares period search, then global and local views.

- Without retraining, the model reached **AUC 0.66** (0.76 on Kaggle's test split).
- AstroNet rescales every transit to depth −1, which throws away how strong the signal is. Combining its score with the BLS signal strength and the number of transits in a logistic regression lifts the cross-validated AUC to **0.82 ± 0.06** (`kaggle/snr_experiments.py`). The same model *without* AstroNet reaches only 0.76, so the CNN's shape information and the signal strength complement each other.

**2. Synthetic transit-injection set (built by me).** I simulated two-year Kepler-like light curves with 29.4-minute cadence, quarter gaps, stellar variability and noise. Into them I injected planets, three kinds of eclipsing binary, and no-signal controls using [batman](https://github.com/lkreidberg/batman). Because I generated the data, every label and parameter (SNR, depth, impact parameter, dilution) is exact.

- **v1** (1,000 TCEs): 98% of injected planets were recovered, but 56% of odd-even and 65% of grazing eclipsing binaries were called planets.
- **v2** (1,000 TCEs, each signal's SNR set exactly between 3 and 100): planet recovery rises from 11% below Kepler's 7.1σ threshold to 60% at 7–15σ and 100% above 30σ. Odd-even and grazing binaries are called planets *more* often as they get stronger, up to 72–81%.

## Vetting diagnostics and LLM-as-judge

AstroNet's two views have no odd/even or secondary-eclipse channel, and those are exactly the checks Kepler's Robovetter makes. `llm_judge/` measures them for every synthetic TCE: odd/even depth difference, secondary eclipse, V/U shape and implied companion size.

- A simple **threshold rule** on these diagnostics calls only about 5% of the binaries planets, against AstroNet's 55–65%. It recovers fewer faint planets, though.
- An **LLM judge** reads the same diagnostics as a short text summary and classifies 100 blind TCEs as PC, AFP or NTP. I labelled the same 100 by hand. Agreement with the truth, with AstroNet and with my labels is reported as Cohen's κ (`llm_judge/analyze_judge.py`).

## Reproduce it

**Evaluation, one command, any OS with Docker.** This runs the **unmodified** original code inside Google's official TensorFlow 1.15 image:

```bash
docker pull --platform linux/amd64 tensorflow/tensorflow:1.15.5-py3
./run_demo.sh
```

The script prints accuracy, AUC and the confusion matrix in about a minute. On Apple Silicon, turn on Rosetta in Docker Desktop first (Settings → General) and use macOS 15 or later. Full setup notes are in `README.txt`.

**Everything else (Linux).** `environment/setup_env.sh` builds Python 3.7 + TensorFlow 1.15.5 + AstroNet in about 10 minutes, including a one-line pydl patch. Run TensorFlow commands from `exoplanet-ml/exoplanet-ml` with `PYTHONPATH=.` set.

```bash
# replication (needs the DR24 TFRecords, see Data)
python -m astronet.train --model=AstroCNNModel --config_name=local_global \
  --train_files='tfrecord/train*' --eval_files='tfrecord/val*' --model_dir=model
python $REPO/replication/train_seeded.py --seed 1 ...same flags... --model_dir=runs/seed1

# score any TFRecord file with the trained model
python $REPO/scripts/score_records.py --model_dir $REPO/model --records <file> --out preds.csv

# constructed dataset (deterministic, seed 8240)
python $REPO/constructed_dataset/build_injection_dataset.py --version 2 --n_per_type 200 --out_dir out_v2

# Kaggle (download exoTrain.csv / exoTest.csv first)
python $REPO/kaggle/build_kaggle_records.py --data_dir ~/kaggle_exo --out_dir records
```

The analysis scripts, which need no TensorFlow (just `numpy pandas scipy scikit-learn matplotlib batman-package`), are run from their own folders:

```bash
cd kaggle && python snr_experiments.py data/kaggle_preds.csv results/snr_experiments.txt
cd constructed_dataset && python analyze_v2.py --inj data/injections_v2.csv \
  --diag ../llm_judge/diag_v2.csv --preds data/preds_v2.csv --fig ../figures/injection_recovery_v2.png
cd llm_judge && python compute_diagnostics.py --version 1 --out diag_v1.csv   # see llm_judge/README.md
```

## Data

| Data | Where |
|---|---|
| Kepler DR24 test TFRecord | `data/` (included) |
| DR24 train/val TFRecords (~130 MB) | the pre-computed TFRecords linked in the [AstroNet README](https://github.com/google-research/exoplanet-ml/tree/master/exoplanet-ml/astronet) |
| Trained checkpoint | `model/` (included) |
| Kaggle light curves | [kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data) |
| Kaggle BLS results + predictions | `kaggle/data/` (included) |
| Synthetic datasets v1 and v2 (TFRecords + parameters + predictions) | `constructed_dataset/data/` (included; **simulated, not real observations**) |

## Repository layout

```
code/                 astronet, light_curve, tf_util, third_party (unmodified, from google-research/exoplanet-ml)
model/                my trained checkpoint (step 625)
data/                 Kepler DR24 test TFRecord
run_demo.sh           one-command evaluation (Docker)
environment/          setup_env.sh: native Linux TF1.15 build
replication/          seeded-training wrapper + logs of every run
scripts/              score_records.py: score any TFRecord with a checkpoint
kaggle/               Kaggle → AstroNet pipeline, analysis, SNR experiments
constructed_dataset/  light-curve simulator, TFRecord builder, v1/v2 data and analysis
llm_judge/            vetting diagnostics, blind judge inputs, rule baseline, agreement analysis
figures/              every figure in the report
```

## Credits

The model code comes from [google-research/exoplanet-ml](https://github.com/google-research/exoplanet-ml) (Apache-2.0). Kepler DR24 labels are from Coughlin et al. (2016) and the light curves from [MAST](https://archive.stsci.edu/). This paper includes data collected by the Kepler mission, which is funded by the NASA Science Mission Directorate. Transit models use batman (Kreidberg 2015). This project was done for COMP8240 at Macquarie University, 2026.
