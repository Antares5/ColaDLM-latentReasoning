# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rebuild ``generate_task_data/*.jsonl`` inputs from the reference outputs.

The raw benchmark inputs are not shipped in the repository (gitignored),
but ``eval_output/tasks_default/<task>.jsonl`` contains, for every sample,
the exact final prompt string (``few_shot_prefix`` + ``prompt``) together
with ``ground_truth`` and ``choices``. This script inverts
``cola_dlm.inference.apply_prompt_template`` to recover the raw
``question`` / ``context`` fields and writes input files that reproduce
the reference prompts exactly (verified by a round-trip assertion).

Usage:
    python scripts/rebuild_task_data.py \
        --ref_dir eval_output/tasks_default \
        --out_dir generate_task_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cola_dlm.inference import apply_prompt_template

TASKS = ["lambada", "mmlu", "obqa", "hellaswag", "race", "siqa", "squad", "story_cloze"]


def choices_text(choices: list[str]) -> str:
    return "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))


def strip_tail(body: str, tail: str) -> str:
    assert body.endswith(tail), f"tail mismatch: ...{body[-120:]!r} vs {tail[:120]!r}"
    return body[: -len(tail)]


def invert_generic(task: str, full_prompt: str, rec: dict) -> dict:
    choices = rec.get("choices") or []
    ct = choices_text(choices)
    ctx, question = "", ""

    if task == "lambada":
        question = full_prompt
    elif task in ("mmlu", "obqa", "hellaswag"):
        # FS prefix ends with "Question: " (mmlu/obqa) or "Context: " (hellaswag);
        # body = f"{question}\n{choices}\nAnswer:"
        body = full_prompt
        body = strip_tail(body, f"\n{ct}\nAnswer:")
        # remove the few-shot prefix; the label ("Question: "/"Context: ") is
        # part of the common prefix that was stripped into few_shot_prefix.
        body = body[len(rec.get("few_shot_prefix", "")) :]
        question = body
    elif task == "race":
        # body = f"{context}\nQuestion: {question}\n" or f"{question}\n"
        #      + f"Options:\n{choices}\nAnswer:"
        body = strip_tail(full_prompt, f"\nOptions:\n{ct}\nAnswer:")
        body = body[len(rec.get("few_shot_prefix", "")) :]
        if "\nQuestion: " in body:
            ctx, question = body.rsplit("\nQuestion: ", 1)
        else:
            ctx, question = "", body
    elif task == "siqa":
        # body = f"{context}\nQuestion: {question}\n{choices}\nAnswer:"
        body = strip_tail(full_prompt, f"\n{ct}\nAnswer:")
        body = body[len(rec.get("few_shot_prefix", "")) :]
        ctx, question = body.rsplit("\nQuestion: ", 1)
    elif task == "squad":
        # body = f"{context}\nQuestion: {question}\nAnswer:"
        body = strip_tail(full_prompt, "\nAnswer:")
        body = body[len(rec.get("few_shot_prefix", "")) :]
        ctx, question = body.rsplit("\nQuestion: ", 1)
    elif task == "story_cloze":
        # body = f"{question}\n(A) {c0}\n(B) {c1}\nEnd:"
        c0, c1 = (choices + ["", ""])[:2]
        body = strip_tail(full_prompt, f"\n(A) {c0}\n(B) {c1}\nEnd:")
        body = body[len(rec.get("few_shot_prefix", "")) :]
        question = body
    else:
        raise ValueError(f"unknown task {task}")

    item = {
        "id": rec.get("id"),
        "context": ctx,
        "question": question,
        "ground_truth": rec.get("ground_truth", ""),
    }
    if choices:
        item["choices"] = choices

    # round-trip verification: the rebuilt input must reproduce the prompt
    rebuilt = apply_prompt_template(task, ctx, question, item["ground_truth"], choices or None)
    assert rebuilt == full_prompt, f"{task} id={rec.get('id')}: round-trip mismatch"
    return item


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", default="eval_output/tasks_default")
    ap.add_argument("--out_dir", default="generate_task_data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for task in TASKS:
        src = os.path.join(args.ref_dir, f"{task}.jsonl")
        if not os.path.isfile(src):
            print(f"[skip] {src} not found")
            continue
        n_ok = n_fail = 0
        out_path = os.path.join(args.out_dir, f"{task}.jsonl")
        with open(src, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                full_prompt = rec.get("few_shot_prefix", "") + rec["prompt"]
                try:
                    item = invert_generic(task, full_prompt, rec)
                except AssertionError as e:
                    n_fail += 1
                    if n_fail <= 3:
                        print(f"  [warn] {e}")
                    continue
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_ok += 1
        print(f"[{task}] rebuilt {n_ok} samples -> {out_path} (failed: {n_fail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
