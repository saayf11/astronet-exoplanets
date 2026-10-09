# Finding exoplanets with deep learning: replicating and extending AstroNet

I replicated **AstroNet** ([Shallue & Vanderburg, 2018](https://iopscience.iop.org/article/10.3847/1538-3881/aa9e09)), the convolutional neural network from Google Brain that found the exoplanets Kepler-90i and Kepler-80g in NASA Kepler data. Then I tested it on two datasets it had never seen before.

**Headline:** my retrained model matches the paper on 1,574 held-out Kepler signals: **AUC 0.990** (the paper reports 0.988) and **96.3% accuracy**. It recovered 350 of the 360 real planet candidates.

## The problem

Kepler watched about 200,000 stars for four years. An automated pipeline flags periodic dips in brightness ("threshold crossing events"), but most of those dips are not planets. They come from eclipsing binary stars, instrumental artefacts or background stars. AstroNet treats vetting as binary classification: given a signal, it outputs P(planet).

## Model

Each signal is phase-folded on its period into two views:

- **Global view:** the whole orbit, 2,001 bins, fed through 5 convolutional blocks (16 → 256 filters)
- **Local view:** a zoom on the transit, 201 bins, fed through 2 convolutional blocks (16 → 32 filters)

The two towers are concatenated, passed through 4 dense layers of 512 units, and end in a sigmoid that outputs P(planet).

## Replication results (Kepler DR24 test set)

| Metric | My run | Published |
|---|---|---|
| AUC | **0.990** | 0.988 (paper) |
| Accuracy | **0.963** | 0.963 (repo) |
| Planets recovered | 350 / 360 | 350 / 360 (repo) |
| False alarms | 48 / 1,214 | 49 / 1,214 (repo) |

The model was trained from scratch on 12,589 expert-labelled signals: 625 steps, batch size 256, about 35 minutes on a CPU. I checked the result in two environments (Linux, and macOS via Docker) and got identical numbers.

## Extensions: two datasets AstroNet had never seen

**1. Kaggle "Exoplanet Hunting in Deep Space"** (5,657 Kepler stars, of which only 0.74% host planets). This dataset has raw flux, no timestamps and no candidate periods, so I built a pipeline to turn it into AstroNet's input: spline detrending, then a box-least-squares period search, then global and local views. Without retraining, the model reached **AUC 0.66** (0.76 on Kaggle's test split). It is clearly harder than DR24, because many stars show only 2–3 transits.

**2. Synthetic transit-injection set (built by me).** I simulated 1,000 two-year Kepler-like light curves with 29.4-minute cadence, quarter gaps, stellar variability and noise. Into them I injected planets, three kinds of eclipsing binary, and no-signal controls using [batman](https://github.com/lkreidberg/batman). Because I generated the data, every label and parameter (SNR, depth, impact parameter, dilution) is exact.

- **98%** of injected planets were recovered
- But **56% / 65%** of odd-even and grazing eclipsing binaries were called planets. These are the model's main blind spots.

## Reproduce it in one command

The evaluation runs the **unmodified** original code inside Google's official TensorFlow 1.15 Docker image.

```bash
docker pull --platform linux/amd64 tensorflow/tensorflow:1.15.5-py3
./run_demo.sh
```

The script prints accuracy, AUC and the confusion matrix in about a minute. On Apple Silicon, turn on Rosetta in Docker Desktop first (Settings → General) and use macOS 15 or later. Full setup notes are in `README.txt`.

## Repository layout

```
code/          astronet, light_curve, tf_util, third_party (from google-research/exoplanet-ml)
model/         my trained checkpoint (step 625)
data/          Kepler DR24 test TFRecord
run_demo.sh    one-command evaluation
```

## What's next

- v2 of the synthetic set: inject signals into real Kepler noise, with more faint planets
- Fine-tune on the Kaggle data and add signal strength as a feature
- An LLM-as-judge check on the vetting diagnostics

## Credits

The model code comes from [google-research/exoplanet-ml](https://github.com/google-research/exoplanet-ml) (Apache-2.0). Kepler DR24 labels are from Coughlin et al. (2016) and the light curves from [MAST](https://archive.stsci.edu/). This project was done for COMP8240 at Macquarie University, 2026.
