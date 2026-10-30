# LLM-as-judge for TCE vetting

**Question.** Can an LLM, reading the same structured vetting diagnostics a human vetter uses, catch the eclipsing binaries that fool AstroNet? How well does it agree with the ground truth, with AstroNet, and with a human vetter?

## Protocol

1. **Diagnostics.** `compute_diagnostics.py` regenerates each constructed light curve exactly from its seed. It then measures Robovetter-style diagnostics:
   - depth and SNR;
   - odd/even depth difference;
   - strongest secondary dip at phase 0.25–0.75;
   - V/U shape ratio;
   - implied companion radius;
   - duration consistency.

   Only the TCE ephemeris and stellar radius and mass are used, never the label.
2. **Blind sample.** `make_judge_inputs.py` draws 20 TCEs per injected type (100 in total), shuffles them, and gives them anonymous IDs J001–J100. The answer key, `judge_inputs/key.csv`, is kept separate.
3. **Judge.** The system prompt is `judge_inputs/prompt.md`: rubric-based, following the Prometheus 2 style of an explicit rubric with a constrained output. Each batch of 25 is judged in a fresh context. Use either of these:
   - `run_judge_api.py`, at temperature 0, which saves the raw responses; or
   - paste `prompt.md` plus one `batch_N.txt` into a brand-new chat, with no project or memory attached. Save the JSON reply, then merge the four replies into `judge_outputs.json` as `{"model": "...", "verdicts": [...]}`.
4. **Human labels.** The student labels the same 100 TCEs in `hand_labels.csv` from the same summaries, without looking at `key.csv`.
5. **Agreement.** `analyze_judge.py` reports:
   - accuracy, confusion matrices and Cohen's κ for judge vs truth, human vs truth and judge vs human (3-class);
   - planet-vs-not agreement, including against AstroNet.

   `rule_baseline.py` provides a fixed-threshold reference on the same inputs.

## Caveats (state in the report)

- The diagnostics use the true injected ephemeris and duration, and the EBs are idealised. Real TCEs have noisier ephemerides and more kinds of false positive, such as centroid offsets and ephemeris matches. The numbers are therefore an upper bound.
- The judge sees engineered features, not pixels. This tests reasoning over diagnostics, not perception.
- The LLM's behaviour depends on the model version and the prompt. Both are recorded in `judge_outputs.json`.
