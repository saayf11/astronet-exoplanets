"""Robovetter-style vetting diagnostics for a TCE, computed from the raw light curve.

These are the diagnostics Kepler's Robovetter uses (and AstroNet's two views do NOT expose
explicitly): odd/even depth difference, secondary eclipse search, transit shape, implied
companion size and duration consistency. They are turned into a short structured text
summary that an LLM judge reads (see make_judge_inputs.py).

Only the TCE ephemeris (period, epoch, duration) and catalogue-level stellar parameters are
used -- never the label or the injection parameters.
"""
import numpy as np

R_SUN_IN_R_EARTH = 109.1


def detrend(times, fluxes, duration):
  """Per-segment normalisation + binned-median trend with iterative clipping of dips."""
  out_t, out_f = [], []
  w = max(0.75, 4 * duration)
  for t, f in zip(times, fluxes):
    f = f / np.median(f)
    mask = np.ones_like(f, bool)
    for _ in range(3):
      edges = np.arange(t[0], t[-1] + w, w)
      cen, med = [], []
      for lo, hi in zip(edges[:-1], edges[1:]):
        s = (t >= lo) & (t < hi) & mask
        if s.sum() > 5:
          cen.append(np.median(t[s])); med.append(np.median(f[s]))
      trend = np.interp(t, cen, med) if len(cen) > 1 else np.full_like(f, np.median(f[mask]))
      r = f / trend - 1
      sd = 1.4826 * np.median(np.abs(r[mask] - np.median(r[mask])))
      mask = r > -3 * sd
    out_t.append(t); out_f.append(r)
  return np.concatenate(out_t), np.concatenate(out_f)


def _depth(r, sel, oot):
  if sel.sum() < 3:
    return np.nan, np.nan
  d = np.median(r[oot]) - np.mean(r[sel])
  e = 1.4826 * np.median(np.abs(r[oot] - np.median(r[oot]))) / np.sqrt(sel.sum())
  return d, e


def diagnostics(times, fluxes, period, t0, duration, rstar=None, mstar=None):
  t, r = detrend(times, fluxes, duration)
  ph = ((t - t0) / period + 0.5) % 1 - 0.5          # phase in [-0.5, 0.5), transit at 0
  dt = ph * period
  epoch = np.round((t - t0) / period).astype(int)
  intr = np.abs(dt) < duration / 2
  near = np.abs(dt) < duration
  oot = ~near
  depth, derr = _depth(r, intr, oot)
  d = {"period_days": period, "duration_hours": duration * 24,
       "n_transits_observed": int(len(np.unique(epoch[intr]))),
       "depth_ppm": depth * 1e6, "depth_snr": depth / derr if derr > 0 else 0.0,
       "noise_ppm_per_point": 1.4826 * np.median(np.abs(r[oot] - np.median(r[oot]))) * 1e6}

  # Odd/even depths.
  do, eo = _depth(r, intr & (epoch % 2 == 1), oot)
  de, ee = _depth(r, intr & (epoch % 2 == 0), oot)
  d["odd_depth_ppm"], d["even_depth_ppm"] = do * 1e6, de * 1e6
  d["odd_even_sigma"] = abs(do - de) / np.hypot(eo, ee) if np.isfinite(do + de) else np.nan

  # Secondary eclipse: strongest dip of the same duration at phase 0.25-0.75 (from transit).
  best = (0.0, np.nan, np.nan)
  for p0 in np.arange(0.25, 0.75001, 0.005):
    off = (((t - t0) / period - p0 + 0.5) % 1 - 0.5) * period
    sel = np.abs(off) < duration / 2
    sd, se = _depth(r, sel & oot, oot)
    if np.isfinite(sd) and se > 0 and sd / se > best[0]:
      best = (sd / se, sd, p0)
  d["secondary_sigma"], d["secondary_depth_ppm"], d["secondary_phase"] = (
      best[0], best[1] * 1e6, best[2])

  # Shape: inner-half depth relative to full-window depth (~1.0-1.2 flat/U, larger = V).
  inner, _ = _depth(r, np.abs(dt) < duration / 4, oot)
  d["shape_inner_over_full"] = inner / depth if depth > 0 else np.nan

  if rstar is not None:
    k = np.sqrt(max(depth, 0))
    d["implied_radius_earth"] = k * rstar * R_SUN_IN_R_EARTH
    d["stellar_radius_rsun"] = rstar
  if rstar is not None and mstar is not None:
    a = 215.03 * (period / 365.25) ** (2 / 3) * mstar ** (1 / 3) / rstar
    d["duration_over_expected"] = duration / (period / np.pi / a)  # vs central, k->0 transit
  return d


def summary_text(tce_id, d):
  """Plain-language structured summary handed to the LLM judge."""
  f = lambda x, fmt: ("n/a" if x is None or not np.isfinite(x) else fmt % x)
  lines = [
      "TCE %s" % tce_id,
      "- Period: %s days; transit duration: %s hours; transits observed: %d" % (
          f(d["period_days"], "%.3f"), f(d["duration_hours"], "%.2f"), int(d["n_transits_observed"])),
      "- Transit depth: %s ppm; detection SNR: %s; noise per 29.4-min point: %s ppm" % (
          f(d["depth_ppm"], "%.0f"), f(d["depth_snr"], "%.1f"), f(d["noise_ppm_per_point"], "%.0f")),
      "- Odd transits depth: %s ppm; even transits depth: %s ppm; odd/even difference: %s sigma" % (
          f(d["odd_depth_ppm"], "%.0f"), f(d["even_depth_ppm"], "%.0f"), f(d["odd_even_sigma"], "%.1f")),
      "- Strongest secondary dip (phase 0.25-0.75): %s ppm at phase %s, significance %s sigma" % (
          f(d["secondary_depth_ppm"], "%.0f"), f(d["secondary_phase"], "%.3f"),
          f(d["secondary_sigma"], "%.1f")),
      "- Shape: depth in inner half of transit / depth over full transit = %s" % f(
          d["shape_inner_over_full"], "%.2f"),
  ]
  if "implied_radius_earth" in d and np.isfinite(d["implied_radius_earth"]):
    lines.append("- Host star radius: %s R_sun; implied companion radius if undiluted: %s R_earth" % (
        f(d["stellar_radius_rsun"], "%.2f"), f(d["implied_radius_earth"], "%.1f")))
  if "duration_over_expected" in d and np.isfinite(d["duration_over_expected"]):
    lines.append("- Duration / expected central-transit duration for this star: %s" % f(
        d["duration_over_expected"], "%.2f"))
  return "\n".join(lines)
