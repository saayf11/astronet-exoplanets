"""Light-curve simulator for the constructed (injection) dataset.

This is the simulation half of build_injection_dataset.py, split out so it can be used
without TensorFlow/AstroNet (e.g. to compute vetting diagnostics for the LLM-as-judge study).
Given the same (idx, type, label, seed) it returns exactly the light curve that
build_injection_dataset.py feeds into AstroNet's preprocessing.

Two generator versions:
  v1  (update presentation): target SNR log-uniform 5-300, noise clipped to 20-2000 ppm,
      so deep signals overshoot their target SNR (few planets near the 7.1-sigma threshold).
  v2  (final report): noise level drawn first (20-2000 ppm, log-uniform), then the signal is
      SCALED so its SNR hits the target exactly (target log-uniform 3-100). Planet radius ratio
      follows from the depth, EB dilution likewise. This resolves recovery near threshold.
"""
import batman
import numpy as np

CADENCE = 29.4 / 60 / 24  # Kepler long cadence, days
BASELINE = 730.0          # ~2 years
N_QUARTERS = 8
LD = [0.40, 0.26]         # quadratic limb darkening, roughly Sun-like in Kepler band

TYPES = [("planet", "PC"), ("eb_secondary", "AFP"), ("eb_oddeven", "AFP"),
         ("grazing_eb", "AFP"), ("none", "NTP")]


def a_over_rs(period, mstar, rstar):
  """Kepler's third law: semi-major axis in stellar radii (215 R_sun per AU)."""
  return 215.03 * (period / 365.25) ** (2 / 3) * mstar ** (1 / 3) / rstar


def t14(period, a, k, b):
  """Total transit duration (days) for a circular orbit."""
  inc = np.arccos(b / a)
  arg = np.sqrt(max((1 + k) ** 2 - b ** 2, 1e-6)) / (a * np.sin(inc))
  return period / np.pi * np.arcsin(min(arg, 1.0))


def transit(t, t0, period, k, a, b, supersample=7):
  p = batman.TransitParams()
  p.t0, p.per, p.rp, p.a = t0, period, k, a
  p.inc = np.degrees(np.arccos(b / a))
  p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", LD
  m = batman.TransitModel(p, t, supersample_factor=supersample, exp_time=CADENCE)
  return m.light_curve(p)


def base_light_curve(rng):
  """Time grid + multiplicative stellar variability (noise is added after injection)."""
  t = np.arange(0, BASELINE, CADENCE)
  var = np.zeros_like(t)
  for _ in range(2):
    amp = np.exp(rng.uniform(np.log(1e-4), np.log(5e-3)))
    per = rng.uniform(3, 40)
    var += amp * np.sin(2 * np.pi * t / per + rng.uniform(0, 2 * np.pi))
  return t, 1 + var


def _signal(t, sig_type, period, t0, mstar, rstar, rng, info):
  """Unscaled signal model (1 = no dip) and its geometry. Consumes rng in v1 order."""
  a = a_over_rs(period, mstar, rstar)
  if sig_type == "planet":
    k = np.exp(rng.uniform(np.log(0.01), np.log(0.2)))
    b = rng.uniform(0, 0.9)
    model = transit(t, t0, period, k, a, b)
  elif sig_type == "eb_secondary":
    k = rng.uniform(0.15, 0.5)
    b = rng.uniform(0, 0.7)
    ratio = rng.uniform(0.1, 0.8)  # secondary/primary depth
    model = transit(t, t0, period, k, a, b)
    model *= transit(t, t0 + period / 2, period, k * np.sqrt(ratio), a, b)
    info["depth_ratio"] = ratio
  elif sig_type == "eb_oddeven":
    k = rng.uniform(0.12, 0.45)
    b = rng.uniform(0, 0.7)
    ratio = rng.uniform(0.3, 0.8)  # odd/even depth ratio
    a = a_over_rs(2 * period, mstar, rstar)
    model = transit(t, t0, 2 * period, k, a, b)
    model *= transit(t, t0 + period, 2 * period, k * np.sqrt(ratio), a, b)
    info["depth_ratio"] = ratio
  elif sig_type == "grazing_eb":
    k = rng.uniform(0.2, 0.5)
    b = rng.uniform(1 - k + 0.05, 1 + k - 0.1)
    model = transit(t, t0, period, k, a, b)
  else:  # none
    k, b = 0.05, 0.5
    model = np.ones_like(t)
  return model, k, b, a


def simulate(idx, sig_type, label, seed, version=1):
  """Returns (list_of_time_arrays, list_of_flux_arrays, tce_dict, info_dict)."""
  rng = np.random.RandomState(seed)
  t, flux = base_light_curve(rng)
  mstar, rstar = rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3)
  period = np.exp(rng.uniform(np.log(0.8), np.log(60.0)))
  t0 = rng.uniform(0, period)
  info = dict(idx=idx, type=sig_type, label=label, period=period, t0=t0,
              mstar=mstar, rstar=rstar, version=version)
  model, k, b, a = _signal(t, sig_type, period, t0, mstar, rstar, rng, info)

  if label == "AFP":
    # Background-EB dilution: eclipse depth reduced by the blended flux ratio.
    dilution = np.exp(rng.uniform(np.log(0.003), np.log(0.5)))
    model = 1 - dilution * (1 - model)
    info["dilution"] = dilution

  duration = t14(period, a, k, min(b, 1 + k - 1e-3))
  n_tr = int(round(BASELINE / period))
  n_in = max(duration / CADENCE, 1) * n_tr

  if version == 1:
    depth = 1 - model.min()
    if depth > 0:
      target = np.exp(rng.uniform(np.log(5.0), np.log(300.0)))
      sigma = np.clip(depth * np.sqrt(n_in) / target, 20e-6, 2000e-6)
    else:
      sigma = np.exp(rng.uniform(np.log(30e-6), np.log(400e-6)))
  else:
    # v2: noise first, then scale the dip so the SNR equals the target exactly.
    sigma = np.exp(rng.uniform(np.log(20e-6), np.log(2000e-6)))
    depth0 = 1 - model.min()
    if depth0 > 0:
      target = np.exp(rng.uniform(np.log(3.0), np.log(100.0)))
      want = min(target * sigma / np.sqrt(n_in), 0.5)  # cap at 50% deep
      model = 1 - (1 - model) * (want / depth0)
      if label == "AFP":
        info["dilution"] *= want / depth0  # effective dilution after rescaling
      info["target_snr"] = target
  depth = 1 - model.min()
  flux = flux * model * (1 + rng.normal(0, sigma, t.size))
  info.update(k=k, b=b, depth_ppm=depth * 1e6, duration_days=duration, sigma_ppm=sigma * 1e6,
              snr=depth / sigma * np.sqrt(n_in) if depth > 0 else 0.0, n_transits=n_tr)

  keep = rng.uniform(size=t.size) > 0.03
  quarters = np.array_split(np.arange(t.size), N_QUARTERS)
  all_t, all_f = [], []
  for q in quarters:
    q = q[keep[q]][40:]  # drop start of quarter (gap)
    all_t.append(t[q])
    all_f.append(flux[q] * rng.uniform(0.98, 1.02))
  tce = {"tce_period": period, "tce_duration": max(duration, 0.02), "tce_time0bk": t0}
  return all_t, all_f, tce, info


def jobs(n_per_type, seed=8240, version=1):
  out, i = [], 0
  base = seed * 100000 if version == 1 else (seed + 1) * 100000
  for sig_type, label in TYPES:
    for _ in range(n_per_type):
      out.append((i, sig_type, label, base + i)); i += 1
  return out
