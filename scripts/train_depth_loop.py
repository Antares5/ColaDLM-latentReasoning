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
"""Flow Matching fine-tune of the Cola DiT prior with depth looping.

Trains the released checkpoint under the paper's conditional Flow
Matching objective (Eq. 2.1.7 / 2.2.3) with one addition: the number of
depth loops R is sampled per step from ``--loop_choices`` and the loop
index is injected through the zero-initialised ``loop_emb`` channel
(see ``ColaDiTModel``). One checkpoint then serves every R at inference.

Training layout mirrors the inference-time conditional forward exactly:

* VAE is frozen; text windows are encoded once per batch to ``z_0``.
* Per sample, a random target block ``b`` is picked; the NA sequence is
  ``[z_0^(<b), z_t^(b)]`` with ``txt_q_shape = block_size``.
* ``z_t = (1 - t/T) z_0 + (t/T) z_1``, target ``v = (z_1 - z_0) / T``
  — the same parameterisation as the Euler update in inference.py.
* With probability ``--cfg_dropout`` the history is emptied (uncond).

Example:
    python scripts/train_depth_loop.py \
        --dit_path hf_models/cola_dlm/cola_dit \
        --vae_path hf_models/cola_dlm/cola_vae \
        --tokenizer_path hf_models/tokenizer.json \
        --out_dir ckpts/depth_loop_r124 \
        --max_steps 500 --batch_size 8 --seq_len 128 \
        --loop_choices 1 2 4 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cola_dlm import ColaDiTModel, ColaTextVAEModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit_path", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    p.add_argument("--dataset_lines", type=int, default=30000, help="Lines of the train split to tokenise")
    p.add_argument("--seq_len", type=int, default=128, help="Token window length (multiple of 16)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loop_choices", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--cfg_dropout", type=float, default=0.1)
    p.add_argument("--T", type=float, default=1000.0)
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--optimizer", default="adamw8bit", choices=["adamw8bit", "adamw"])
    p.add_argument("--grad_ckpt", type=int, default=1, help="Checkpoint the DiT forward (recompute in backward)")
    return p.parse_args()


def lr_at(step: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cos)


def build_windows(tokenizer: Tokenizer, args) -> list[torch.Tensor]:
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split="train")
    windows: list[torch.Tensor] = []
    for line in ds.select(range(min(args.dataset_lines, len(ds))))["text"]:
        line = line.strip()
        if len(line) < 200:
            continue
        ids = tokenizer.encode(line).ids
        for start in range(0, len(ids) - args.seq_len + 1, args.seq_len):
            windows.append(torch.tensor(ids[start : start + args.seq_len], dtype=torch.long))
    return windows


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    block_size = 16
    n_blocks = args.seq_len // block_size
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[train] loading tokenizer: {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    print(f"[train] tokenising dataset {args.dataset}/{args.dataset_config} ...")
    windows = build_windows(tokenizer, args)
    print(f"[train] {len(windows)} training windows of {args.seq_len} tokens")

    print(f"[train] loading VAE (frozen): {args.vae_path}")
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device)
    vae.requires_grad_(False)
    vae.eval()
    scale, shift = vae.scaling_factor, vae.shifting_factor

    # Pre-encode every training window once, then drop the VAE entirely:
    # one z0 (128 x 16 fp32) is only 8KB, so the whole latent dataset is
    # a few dozen MB and the ~2GB VAE never touches the training loop.
    print(f"[train] pre-encoding {len(windows)} windows with the VAE ...")
    z0_all: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, len(windows), 16):
            chunk = [w.to(device) for w in windows[i : i + 16]]
            enc = vae.encode(chunk)
            z0_all.extend([((lat - shift) * scale).float() for lat in enc.latents_list])
    del vae, windows
    torch.cuda.empty_cache()
    n_samples = len(z0_all)

    print(f"[train] loading DiT: {args.dit_path}")
    dit = ColaDiTModel.from_pretrained(args.dit_path).to(device)
    dit.train()
    max_loops = max(args.loop_choices)
    assert max_loops <= dit.max_depth_loops, (
        f"loop choice {max_loops} exceeds COLA_DIT_MAX_LOOPS={dit.max_depth_loops}"
    )

    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.PagedAdamW8bit(dit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(dit.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_hist: dict[int, list[float]] = {r: [] for r in args.loop_choices}
    t_start = time.time()
    for step in range(args.max_steps):
        r = random.choice(args.loop_choices)
        dit.depth_loops = r
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)

        batch_idx = random.sample(range(n_samples), args.batch_size)
        z0_list = [z0_all[i] for i in batch_idx]

        # ---- build the NA FM batch ----------------------------------------
        # Full clean sequence per sample with the target block replaced by
        # its noisy interpolation z_t. Q covers every position (the
        # block-causal mask restricts visibility to <= its own block, so
        # the target block sees exactly {clean history, noisy block} —
        # the training visible set V_b). Loss is taken on the target block
        # rows only. CFG dropout: feed the noisy block with empty history.
        txt_parts, ts_parts, targets = [], [], []
        row_spans = []  # (start, end) of each sample's target block in the flat batch
        offset = 0
        for z0 in z0_list:
            b_idx = random.randint(0, n_blocks - 1)
            drop_hist = random.random() < args.cfg_dropout
            z0_blk = z0[b_idx * block_size : (b_idx + 1) * block_size]
            z1 = torch.randn_like(z0_blk)
            t = random.uniform(0.0, args.T)
            z_t = (1.0 - t / args.T) * z0_blk + (t / args.T) * z1
            if drop_hist:
                seq, tgt_lo = z_t, 0
            else:
                seq = z0.clone()
                seq[b_idx * block_size : (b_idx + 1) * block_size] = z_t
                tgt_lo = b_idx * block_size
            txt_parts.append(seq)
            ts_parts.append(torch.full((seq.shape[0],), t, device=device))
            targets.append((z1 - z0_blk) / args.T)
            row_spans.append((offset + tgt_lo, offset + tgt_lo + block_size))
            offset += seq.shape[0]

        txt = torch.cat(txt_parts, dim=0)
        target = torch.cat(targets, dim=0)
        ts = torch.cat(ts_parts, dim=0)
        txt_shape = torch.tensor([p.shape[0] for p in txt_parts], device=device).unsqueeze(1)
        txt_q_shape = txt_shape.clone()

        def _dit_fwd(t_):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return dit(
                    txt=t_,
                    txt_shape=txt_shape,
                    txt_q_shape=txt_q_shape,
                    timestep=ts,
                    update_kv=False,
                    use_kv_cache=False,
                ).txt_sample

        if args.grad_ckpt:
            pred = torch.utils.checkpoint.checkpoint(_dit_fwd, txt.to(torch.bfloat16), use_reentrant=False)
        else:
            pred = _dit_fwd(txt.to(torch.bfloat16))
        pred_rows = torch.cat([pred[s:e] for s, e in row_spans], dim=0)
        loss = F.mse_loss(pred_rows.float(), target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(dit.parameters(), args.grad_clip)
        opt.step()

        loss_hist[r].append(loss.item())
        if (step + 1) % args.log_every == 0:
            msg = " | ".join(
                f"R{k}: {sum(v[-args.log_every:]) / max(1, len(v[-args.log_every:])):.4f}"
                for k, v in loss_hist.items()
            )
            el = time.time() - t_start
            print(f"[train] step {step + 1}/{args.max_steps} lr={lr_at(step, args):.2e} {msg} ({el:.0f}s)")

        if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
            ckpt = os.path.join(args.out_dir, f"step{step + 1}")
            dit.depth_loops = 1  # persist with the default single-pass setting
            dit.save_pretrained(ckpt)
            print(f"[train] saved {ckpt}")

    summary = {
        "args": vars(args),
        "final_loss": {str(k): (sum(v[-20:]) / max(1, len(v[-20:]))) for k, v in loss_hist.items()},
        "wall_time_s": time.time() - t_start,
    }
    with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] done in {summary['wall_time_s']:.0f}s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
