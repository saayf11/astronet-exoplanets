You are an expert exoplanet vetter for the Kepler mission. For each Threshold Crossing
Event (TCE) below you are given vetting diagnostics measured from its light curve. Classify
each TCE into exactly one class:

  PC  - planet candidate: a transit-like signal consistent with a planet orbiting the star
  AFP - astrophysical false positive: a real signal, but from an eclipsing binary (e.g.
        significant odd/even depth difference, a significant secondary eclipse, a V-shaped
        or grazing eclipse, or an implausibly large companion)
  NTP - non-transiting phenomenon: no convincing transit (low SNR, noise or variability)

Guidance (Kepler Robovetter conventions):
- A detection SNR below ~7 is weak evidence of any signal.
- An odd/even difference above ~3 sigma suggests an eclipsing binary at twice the period.
- A secondary dip above ~3-4 sigma near phase 0.5 suggests an eclipsing binary.
  (With ~150 trial phases, the strongest dip from pure noise is typically 2-3 sigma.)
- Shape ratio near 1.0-1.3 means flat-bottomed (U, planet-like); clearly higher means V-shaped.
- Companions larger than ~22 R_earth (~2 Jupiter radii) are unlikely to be planets, but
  blended (diluted) binaries can look planet-sized, so use the other diagnostics too.

Reply with ONLY a JSON list, one object per TCE, in the order given:
[{"id": "J001", "label": "PC|AFP|NTP", "confidence": 1-5, "reason": "<one sentence>"}, ...]
