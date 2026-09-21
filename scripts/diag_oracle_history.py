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
"""Oracle-history generation diagnostic for depth-loop checkpoints.

Isolates *why* R>1 generation collapses even though the per-cell FM loss
looks close to R=1. Blocks are denoised exactly as at inference (16-step
Euler, two-pass cache-consistent forward, conditional drift only), but
the KV history is always committed from the ground-truth clean latent
``z_0`` instead of the model's own output:

  * If R>1 reconstruction error explodes here, the looped drift field /
    its Euler trajectory is broken even with perfect history.
  * If R>1 reconstructs fine here, the collapse must come from the
    self-history feedback loop at inference (committing the model's own
    imperfect blocks), i.e. the looped t=0 commit forward amplifying
    small errors.

Example:
    python scripts/diag_oracle_history.py \
        --dit_path ckpts/depth_loop_r124_v2/step1000 \
        --vae_path hf_models/cola_dlm/cola_vae \
        --tokenizer_path hf_models/tokenizer.json \
        --loops 1 2 4 --n_samples 32
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit_path", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    p.add_argument("--dataset_lines", type=int, default=3000)
    p.add_argument("--seq_len", type=int, default=64, help="4 blocks: 1 history + 3 generated")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--timestep_num", type=int, default=16)
    p.add_argument("--T", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--history", choices=["oracle", "self"], default="oracle",
                   help="oracle: commit ground-truth z0 blocks; self: commit the model's own outputs (true autoregression)")
    p.add_argument("--mix_pin", type=int, default=0,
                   help="Pin this many leading rows of the FIRST generated block to their z0 "
                        "values at t=0 (lambada-style clean guidance: a mixed-t block that is "
                        "never seen in FM training). 0 disables.")
    p.add_argument("--cfg", type=float, default=0.0,
                   help="Enable the inference CFG branch with this guidance scale "
                        "(0 = conditional drift only).")
    p.add_argument("--decode_samples", type=int, default=3, help="VAE-decode this many samples for a text check")
    return p.parse_args()


@torch.no_grad()
def commit_history(dit, hist, device):
    """Commit clean history rows to the DiT KV cache at t=0 (bf16, like inference)."""
    for blk in dit.blocks:
        blk.set_kv_cache(True)
    h = hist.to(torch.bfloat16)
    h_shape = torch.tensor([[h.shape[0]]], device=device, dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dit(txt=h, txt_shape=h_shape, txt_q_shape=h_shape,
            timestep=torch.zeros(h.shape[0], device=device, dtype=torch.bfloat16),
            update_kv=True, use_kv_cache=True)


@torch.no_grad()
def cond_drift(dit, z_t, t, k_len, block_size, device, cfg: float = 0.0):
    q_shape = torch.tensor([[block_size]], device=device, dtype=torch.long)
    k_shape = torch.tensor([[k_len]], device=device, dtype=torch.long)
    if torch.is_tensor(t):
        ts = t.to(device=device, dtype=torch.bfloat16)
    else:
        ts = torch.full((block_size,), t, device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cond = dit(txt=z_t.to(torch.bfloat16), txt_shape=k_shape, txt_q_shape=q_shape,
                   timestep=ts, update_kv=False, use_kv_cache=True).txt_sample
        if cfg > 0:
            # Unconditional branch: K = Q = own block, no cache — exactly
            # the inference.py CFG path, combined in bf16 like inference.
            uncond = dit(txt=z_t.to(torch.bfloat16), txt_shape=q_shape, txt_q_shape=q_shape,
                         timestep=ts, update_kv=False, use_kv_cache=False).txt_sample
            s = torch.tensor(cfg, device=device, dtype=torch.bfloat16)
            drift = s * (cond - uncond) + uncond
            if torch.isnan(drift).any() or torch.isinf(drift).any():
                print(f"[warn] non-finite drift: cond_nan={torch.isnan(cond).any().item()} "
                      f"uncond_nan={torch.isnan(uncond).any().item()} "
                      f"cond_absmax={cond.abs().max().item():.2f} uncond_absmax={uncond.abs().max().item():.2f}")
            return drift.float()
    return cond.float()


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

    dit = ColaDiTModel.from_pretrained(args.dit_path).to(device).eval()
    timesteps = torch.linspace(int(args.T), 0, args.timestep_num + 1)

    # Fixed per-sample noise for the whole experiment (same across R).
    gen = torch.Generator(device=device).manual_seed(args.seed)
    noise_all = [torch.randn(args.seq_len, z0_all[0].shape[-1], device=device, generator=gen)
                 for _ in z0_all]

    for r in args.loops:
        dit.depth_loops = r
        # block_mse[b] = list over samples of MSE(denoised, z0_block b)
        block_mse: dict[int, list[float]] = {b: [] for b in range(1, n_blocks)}
        recon: list[torch.Tensor] = []
        for si, z0 in enumerate(z0_all):
            z_hat = z0.clone()
            for b in range(1, n_blocks):
                hist = z_hat[: b * block_size] if args.history == "self" else z0[: b * block_size]
                commit_history(dit, hist, device)
                z = noise_all[si][b * block_size : (b + 1) * block_size].clone()
                z0_blk = z0[b * block_size : (b + 1) * block_size]
                pin = args.mix_pin if b == 1 else 0
                if pin:
                    z[:pin] = z0_blk[:pin]
                for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
                    dt = (float(t_curr) - float(t_next)) / args.T
                    if pin:
                        ts_vec = torch.full((block_size,), float(t_curr), device=device)
                        ts_vec[:pin] = 0.0
                    else:
                        ts_vec = float(t_curr)
                    drift = cond_drift(dit, z, ts_vec, (b + 1) * block_size, block_size, device, cfg=args.cfg)
                    z = z - drift * dt
                    if pin:
                        z[:pin] = z0_blk[:pin]
                block_mse[b].append(F.mse_loss(z, z0_blk).item())
                z_hat[b * block_size : (b + 1) * block_size] = z
            recon.append(z_hat)
        for blk in dit.blocks:
            blk.set_kv_cache(False)
        msg = " ".join(f"b{b}: {sum(v) / len(v):.3f}" for b, v in block_mse.items())
        print(f"[{args.history}] R={r} recon MSE by block: {msg}")

        if args.decode_samples > 0:
            for si in range(min(args.decode_samples, len(recon))):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dec = vae.decode(
                        z=recon[si],
                        txt_shape=torch.tensor([[args.seq_len]], device=device, dtype=torch.long),
                        txt_q_shape=torch.tensor([[args.seq_len]], device=device, dtype=torch.long),
                        update_kv=False,
                    )
                ids = dec.view(-1, dec.shape[-1]).argmax(dim=-1).tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True)
                print(f"[{args.history}] R={r} sample{si} text: {text[:160]!r}")
    del vae
    return 0


if __name__ == "__main__":
    sys.exit(main())
