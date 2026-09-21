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
"""Diagnose the looped drift field of a depth-loop checkpoint.

For each loop count R in ``--loops``:
  * FM loss stratified by flow-matching time bucket (same two-pass,
    cache-consistent layout as train_depth_loop.py, all under no_grad);
  * cosine similarity and relative RMS between the R-loop drift and the
    1-loop drift on identical (z_t, t, history) inputs — measures how
    much the loop actually changes the vector field.

Example:
    python scripts/diag_loop_field.py \
        --dit_path ckpts/depth_loop_r124_v2/step1000 \
        --vae_path hf_models/cola_dlm/cola_vae \
        --tokenizer_path hf_models/tokenizer.json \
        --loops 1 2 4 --n_samples 64
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cola_dlm import ColaDiTModel, ColaTextVAEModel
from scripts.train_depth_loop import build_windows

T_BUCKETS = [(0, 100), (100, 300), (300, 500), (500, 700), (700, 900), (900, 1000)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit_path", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    p.add_argument("--dataset_lines", type=int, default=3000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_samples", type=int, default=64, help="Windows per (R, t-bucket) cell")
    p.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--T", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


@torch.no_grad()
def drift_for(dit, z_t, hist, t, device, block_size):
    """Two-pass cache-consistent drift of one sample; returns (block, d) fp32."""
    for blk in dit.blocks:
        blk.set_kv_cache(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if hist is not None:
            h = hist.to(torch.bfloat16)
            h_shape = torch.tensor([[h.shape[0]]], device=device, dtype=torch.long)
            dit(txt=h, txt_shape=h_shape, txt_q_shape=h_shape,
                timestep=torch.zeros(h.shape[0], device=device, dtype=torch.bfloat16),
                update_kv=True, use_kv_cache=True)
            k_shape = h_shape + block_size
        else:
            k_shape = torch.tensor([[block_size]], device=device, dtype=torch.long)
        q_shape = torch.tensor([[block_size]], device=device, dtype=torch.long)
        ts = torch.full((block_size,), t, device=device, dtype=torch.bfloat16)
        out = dit(txt=z_t.to(torch.bfloat16), txt_shape=k_shape, txt_q_shape=q_shape,
                  timestep=ts, update_kv=False, use_kv_cache=hist is not None).txt_sample
    for blk in dit.blocks:
        blk.set_kv_cache(False)
    return out.float()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    block_size = 16
    n_blocks = args.seq_len // block_size

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    windows = build_windows(tokenizer, args)[: args.n_samples]
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
    scale, shift = vae.scaling_factor, vae.shifting_factor
    z0_all: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, len(windows), 16):
            enc = vae.encode([w.to(device) for w in windows[i : i + 16]])
            z0_all.extend([((lat - shift) * scale).float() for lat in enc.latents_list])
    del vae, windows
    torch.cuda.empty_cache()

    dit = ColaDiTModel.from_pretrained(args.dit_path).to(device).eval()

    # Fixed evaluation grid: same (z0, block, t) triples for every R.
    cells = []
    for lo, hi in T_BUCKETS:
        for i in range(len(z0_all)):
            z0 = z0_all[i]
            b_idx = random.randint(1, n_blocks - 1)  # always conditional
            t = random.uniform(lo, hi)
            z0_blk = z0[b_idx * block_size : (b_idx + 1) * block_size]
            g = torch.Generator(device=device).manual_seed(args.seed * 100003 + i * 97 + lo)
            z1 = torch.randn(z0_blk.shape, device=device, generator=g)
            z_t = (1.0 - t / args.T) * z0_blk + (t / args.T) * z1
            cells.append((z0[: b_idx * block_size], z_t, t, z1 - z0_blk))

    print(f"[diag] {len(z0_all)} windows x {len(T_BUCKETS)} t-buckets = {len(cells)} cells per R")

    ref_drifts: list[torch.Tensor] | None = None
    for r in args.loops:
        dit.depth_loops = r
        drifts: list[torch.Tensor] = []
        bucket_loss: dict[tuple[int, int], list[float]] = {b: [] for b in T_BUCKETS}
        for idx, (hist, z_t, t, target) in enumerate(cells):
            drift = drift_for(dit, z_t, hist, t, device, block_size)
            drifts.append(drift)
            lo = T_BUCKETS[idx // len(z0_all)][0]
            for b in T_BUCKETS:
                if b[0] == lo:
                    bucket_loss[b].append(F.mse_loss(drift, target).item())
        line = " ".join(
            f"[{lo:>3},{hi:>4}): {sum(v) / len(v):.3f}" for (lo, hi), v in bucket_loss.items()
        )
        print(f"[diag] R={r} loss by t-bucket: {line}")
        if ref_drifts is None:
            ref_drifts = drifts
        else:
            cos_vals, rms_vals = [], []
            for d, d0 in zip(drifts, ref_drifts):
                cos_vals.append(F.cosine_similarity(d.flatten(), d0.flatten(), dim=0).item())
                rms_vals.append(((d - d0).pow(2).mean().sqrt() / d0.pow(2).mean().sqrt()).item())
            print(
                f"[diag] R={r} vs R=1: cos={sum(cos_vals) / len(cos_vals):.4f} "
                f"rel_rms_diff={sum(rms_vals) / len(rms_vals):.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
