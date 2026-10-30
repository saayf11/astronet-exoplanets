"""Does adding the BLS detection SNR fix AstroNet's domain shift on the Kaggle data?

AstroNet normalises each view to depth -1, discarding absolute depth/SNR. On DR24 that is
harmless (every TCE passed Kepler's 7.1-sigma cut); on Kaggle the "TCE" is BLS's strongest
dip, which may be noise. We compare, on identical splits:

  A  AstroNet probability alone (DR24-trained, no retraining)
  B  BLS SNR alone
  C  gate: AstroNet probability, set to 0 when BLS SNR < threshold (threshold tuned on train)
  D  logistic regression on [logit(p_AstroNet), log(SNR), log(n_transits)] fit on train
  E  ablation: D without AstroNet (does the CNN add anything?)
  F  ablation: number of transits in the 65-day baseline alone

Evaluated (1) on the official Kaggle train -> test split (only 5 test hosts: noisy), and
(2) with repeated stratified 5-fold cross-validation over all 5,657 stars (42 hosts), which
gives a mean +/- std. Metrics: ROC AUC and average precision (AP; = area under the
precision-recall curve, the more informative metric at a 0.74% base rate).

  python snr_experiments.py data/kaggle_preds.csv results/snr_experiments.txt
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

preds = sys.argv[1] if len(sys.argv) > 1 else "data/kaggle_preds.csv"
bls = preds.replace("kaggle_preds", "kaggle_bls")
d = pd.read_csv(preds)
b = pd.read_csv(bls).rename(columns={"split": "kaggle_split", "row": "kaggle_row"})
d = d.merge(b[["kaggle_split", "kaggle_row", "n_transits"]], on=["kaggle_split", "kaggle_row"])
y = (d.kaggle_label == 2).astype(int).values
p = np.clip(d.pred_planet_prob.values, 1e-6, 1 - 1e-6)
snr = np.clip(d.bls_snr.values, 1e-3, None)
X = np.column_stack([np.log(p / (1 - p)), np.log(snr), np.log(d.n_transits.values)])


def fit_gate(tr):
  """Pick the SNR threshold (over train quantiles) that maximises train AP."""
  best = (-1, 0)
  for q in np.linspace(0, 0.95, 40):
    th = np.quantile(snr[tr], q)
    s = p[tr] * (snr[tr] >= th)
    ap = average_precision_score(y[tr], s)
    if ap > best[0]:
      best = (ap, th)
  return best[1]


def scores(tr, te):
  th = fit_gate(tr)
  lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr], y[tr])
  lr2 = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr][:, 1:], y[tr])
  return {"A AstroNet": p[te], "B BLS SNR": snr[te], "C SNR-gated AstroNet": p[te] * (snr[te] >= th),
          "D LogReg(AstroNet, SNR, n_tr)": lr.predict_proba(X[te])[:, 1],
          "E ablation: LogReg(SNR, n_tr)": lr2.predict_proba(X[te][:, 1:])[:, 1],
          "F ablation: n_transits alone": X[te][:, 2]}, th, lr


out = []
tr = np.where(d.kaggle_split == "train")[0]; te = np.where(d.kaggle_split == "test")[0]
s, th, lr = scores(tr, te)
out.append("== Official split: train %d stars (%d hosts) -> test %d stars (%d hosts)" % (
    len(tr), y[tr].sum(), len(te), y[te].sum()))
out.append("gate threshold (BLS SNR) = %.1f; LogReg coefs [logit p, log SNR, log n_tr] = %s" % (
    th, np.round(lr.coef_[0], 3)))
for k, v in s.items():
  out.append("  %-32s AUC %.3f  AP %.3f" % (k, roc_auc_score(y[te], v), average_precision_score(y[te], v)))

res = {}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=8240)
for tr, te in cv.split(X, y):
  s, _, _ = scores(tr, te)
  for k, v in s.items():
    res.setdefault(k, []).append((roc_auc_score(y[te], v), average_precision_score(y[te], v)))
out.append("\n== Repeated stratified 5-fold CV (10 repeats, all %d stars, %d hosts; base rate %.4f)" % (
    len(y), y.sum(), y.mean()))
for k, v in res.items():
  v = np.array(v)
  out.append("  %-32s AUC %.3f +/- %.3f   AP %.3f +/- %.3f" % (k, v[:, 0].mean(), v[:, 0].std(),
                                                             v[:, 1].mean(), v[:, 1].std()))
text = "\n".join(out)
print(text)
if len(sys.argv) > 2:
  open(sys.argv[2], "w").write(text + "\n")
