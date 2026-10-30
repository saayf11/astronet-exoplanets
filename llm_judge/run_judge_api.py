"""Run the LLM judge through the Anthropic API (optional; the batches can also be pasted
into a fresh chat by hand -- the prompt and batches are identical either way).

  pip install anthropic
  export ANTHROPIC_API_KEY=...
  python run_judge_api.py --inputs judge_inputs --model claude-opus-5-5 --out judge_outputs.json

temperature=0 for repeatability; every raw response is saved alongside the parsed labels.
"""
import argparse
import glob
import json
import os
import re

import anthropic


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--inputs", default="judge_inputs")
  ap.add_argument("--model", default="claude-opus-5-5")
  ap.add_argument("--out", default="judge_outputs.json")
  f = ap.parse_args()
  client = anthropic.Anthropic()
  prompt = open(os.path.join(f.inputs, "prompt.md")).read()
  verdicts, raw = [], []
  for path in sorted(glob.glob(os.path.join(f.inputs, "batch_*.txt")),
                     key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
    msg = client.messages.create(model=f.model, max_tokens=8000, temperature=0, system=prompt,
                                 messages=[{"role": "user", "content": open(path).read()}])
    text = msg.content[0].text
    raw.append({"batch": os.path.basename(path), "response": text})
    verdicts += json.loads(text[text.index("["):text.rindex("]") + 1])
    print(path, "->", len(verdicts), "verdicts so far")
  json.dump({"model": f.model, "verdicts": verdicts, "raw": raw}, open(f.out, "w"), indent=1)


if __name__ == "__main__":
  main()
