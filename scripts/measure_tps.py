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
"""Measure Cola DLM generation throughput (tokens per second).

Runs ``generate_task_repaint_inference`` on real benchmark prompts
(default: LAMBADA) with the reference decoding settings and reports
wall-clock throughput for one or more batch sizes. Model loading is
excluded; each configuration gets one warmup batch before timing.

Usage:
    python scripts/measure_tps.py \
        --dit_path hf_models/cola_dlm/cola_dit \
        --vae_path hf_models/cola_dlm/cola_vae \
        --tokenizer_path hf_models/tokenizer.json \
        --input_jsonl generate_task_data/lambada.jsonl \
        --batch_sizes 1 20 --num_samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cola_dlm import ColaDiTModel, ColaTextVAEModel, generate_task_repaint_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit_path", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--task_name", default="lambada")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 20])
    p.add_argument("--num_samples", type=int, default=100, help="Samples per batch-size configuration")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--timestep_num", type=int, default=16)
    p.add_argument("--guidance_scale", type=float, default=7.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--pad_token_id", type=int, default=100277)
    p.add_argument("--eos_token_id", type=int, default=100257)
    p.add_argument("--im_end_token_id", type=int, default=100265)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("COLA_INFER_PER_SAMPLE_NOISE_SEED", "66")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tps] device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    dit = ColaDiTModel.from_pretrained(args.dit_path).to(device).eval()
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    data = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    data = data[: args.num_samples]
    print(f"[tps] {len(data)} prompts from {args.input_jsonl} (task={args.task_name})")
    print(
        f"[tps] settings: max_new_tokens={args.max_new_tokens} timestep_num={args.timestep_num} "
        f"guidance_scale={args.guidance_scale} temperature={args.temperature}"
    )

    def run_batch(batch):
        return generate_task_repaint_inference(
            dit=dit,
            vae=vae,
            tokenizer=tokenizer,
            prompts=batch,
            task_name=args.task_name,
            device=device,
            T=1000.0,
            timestep_num=args.timestep_num,
            guidance_scale=args.guidance_scale,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            pad_token_id=args.pad_token_id,
            eos_token_id=args.eos_token_id,
            im_end_token_id=args.im_end_token_id,
        )

    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    summary = []
    for bs in args.batch_sizes:
        batches = [data[i : i + bs] for i in range(0, len(data), bs)]
        # warmup (excluded from timing)
        run_batch(batches[0])
        sync()

        n_tok, n_samples, t0 = 0, 0, time.perf_counter()
        for batch in batches:
            results = run_batch(batch)
            sync()
            n_samples += len(results)
            for r in results:
                n_tok += len(tokenizer.encode(r["generate"]).ids)
        elapsed = time.perf_counter() - t0
        tps = n_tok / elapsed
        summary.append((bs, n_samples, n_tok, elapsed, tps, elapsed / n_samples))
        print(
            f"[tps] batch_size={bs:>3} | samples={n_samples:>4} | gen_tokens={n_tok:>5} | "
            f"time={elapsed:7.2f}s | TPS={tps:8.2f} tok/s | latency={elapsed / n_samples:6.3f}s/sample"
        )

    print("\n===== TPS summary =====")
    print(f"{'batch_size':>10} {'samples':>8} {'gen_tokens':>10} {'time_s':>9} {'tok/s':>9} {'s/sample':>9}")
    for bs, n_samples, n_tok, elapsed, tps, per in summary:
        print(f"{bs:>10} {n_samples:>8} {n_tok:>10} {elapsed:>9.2f} {tps:>9.2f} {per:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
