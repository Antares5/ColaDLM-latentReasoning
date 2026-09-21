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

Training layout mirrors the inference-time conditional forward exactly
(two-pass, cache-consistent):

* VAE is frozen; text windows are encoded once with the VAE to ``z_0``.
* Per sample, a random target block ``b`` is picked. Pass 1 (no_grad)
  commits the clean history ``z_0^(<b)`` to the DiT KV cache at ``t=0``
  — identical to the inference-time cache write and implementing the
  paper's stop-gradient ``sg(z_0^(<b))``. Pass 2 denoises only the noisy
  block ``z_t^(b)`` with ``txt_q_shape = block_size``, reading the cache
  for conditional samples and running cache-free for unconditional ones
  (``b == 0`` or CFG dropout), matching the two CFG branch forwards.
* ``z_t = (1 - t/T) z_0 + (t/T) z_1``, target drift ``v = z_1 - z_0``
  — the same parameterisation as the Euler update in inference.py.

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
    p.add_argument("--loop_reinject", type=int, default=0,
                   help="Enable loop-transformer-style input re-injection (x_emb added back "
                        "to the hidden state at every loop iteration r>=1)")
    p.add_argument("--loop_cond", choices=["global", "layer", "film"], default="global",
                   help="Per-loop conditioning mode: global = shared loop_emb vector; "
                        "layer = per-(loop, layer) AdaLN embedding; film = per-(loop, layer) "
                        "FiLM on the residual stream")
    p.add_argument("--cond_lr_mult", type=float, default=1.0,
                   help="Learning-rate multiplier for the loop-conditioning parameters "
                        "(loop_emb / loop_emb_layers / loop_film) relative to --lr")
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
    if args.loop_cond != "global":
        # The conditioning parameters are created at construction time, so
        # the mode must be visible via env when fine-tuning a checkpoint
        # whose config predates them.
        os.environ["COLA_DIT_LOOP_COND"] = args.loop_cond
    dit = ColaDiTModel.from_pretrained(args.dit_path).to(device)
    if args.loop_reinject:
        dit.loop_reinject = True
        dit.config.loop_reinject = True  # persist into the saved checkpoint
    if args.loop_cond != "global":
        dit.config.loop_cond = args.loop_cond  # persist into the saved checkpoint
    dit.train()
    max_loops = max(args.loop_choices)
    assert max_loops <= dit.max_depth_loops, (
        f"loop choice {max_loops} exceeds COLA_DIT_MAX_LOOPS={dit.max_depth_loops}"
    )

    cond_param_names = {"loop_emb.weight", "loop_emb_layers.weight", "loop_film"}
    cond_params = [p for n, p in dit.named_parameters() if n in cond_param_names]
    base_params = [p for n, p in dit.named_parameters() if n not in cond_param_names]
    param_groups = [
        {"params": base_params, "lr_mult": 1.0},
        {"params": cond_params, "lr_mult": args.cond_lr_mult},
    ]
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.PagedAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    loss_hist: dict[int, list[float]] = {r: [] for r in args.loop_choices}
    tgt_rms_hist: list[float] = []
    t_start = time.time()
    for step in range(args.max_steps):
        r = random.choice(args.loop_choices)
        dit.depth_loops = r
        lr_now = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr_now * g["lr_mult"]

        batch_idx = random.sample(range(n_samples), args.batch_size)
        z0_list = [z0_all[i] for i in batch_idx]

        # ---- build the cache-consistent NA FM batch ----------------------
        # Two-pass layout that mirrors inference.py byte-for-byte:
        #   pass 1 (no_grad): commit the clean history z_0^(<b) to the KV
        #     cache at t=0. This both matches the inference-time cache
        #     write (history K/V are produced by a separate t=0 forward,
        #     not by the same forward that denoises the block) and
        #     implements the paper's stop-gradient sg(z_0^(<b)) on the
        #     visible set V_b.
        #   pass 2: denoise the noisy target block z_t^(b) only
        #     (txt_q_shape = block_size). Conditional samples read the
        #     cache (use_kv_cache=True); unconditional ones (b==0 or CFG
        #     dropout) run cache-free — the same pair of forwards the CFG
        #     branch performs at inference time.
        cond_txt, cond_ts, cond_targets = [], [], []
        cond_hist, cond_hist_lens = [], []
        unc_txt, unc_ts, unc_targets = [], [], []
        for z0 in z0_list:
            b_idx = random.randint(0, n_blocks - 1)
            drop_hist = random.random() < args.cfg_dropout
            z0_blk = z0[b_idx * block_size : (b_idx + 1) * block_size]
            z1 = torch.randn_like(z0_blk)
            t = random.uniform(0.0, args.T)
            z_t = (1.0 - t / args.T) * z0_blk + (t / args.T) * z1
            target = z1 - z0_blk
            if drop_hist or b_idx == 0:
                unc_txt.append(z_t)
                unc_ts.append(t)
                unc_targets.append(target)
            else:
                cond_txt.append(z_t)
                cond_hist.append(z0[: b_idx * block_size])
                cond_hist_lens.append(b_idx * block_size)
                cond_ts.append(t)
                cond_targets.append(target)

        def _dit_fwd(t_, k_shape, q_shape, ts_, use_cache):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return dit(
                    txt=t_,
                    txt_shape=k_shape,
                    txt_q_shape=q_shape,
                    timestep=ts_,
                    update_kv=False,
                    use_kv_cache=use_cache,
                ).txt_sample

        def _block_fwd(*fwd_args):
            if args.grad_ckpt:
                return torch.utils.checkpoint.checkpoint(_dit_fwd, *fwd_args, use_reentrant=False)
            return _dit_fwd(*fwd_args)

        preds, targets = [], []
        if cond_txt:
            for blk in dit.blocks:
                blk.set_kv_cache(True)  # enable + reset the per-sample cache
            hist_cat = torch.cat(cond_hist, dim=0).to(torch.bfloat16)
            hist_shape = torch.tensor(cond_hist_lens, device=device, dtype=torch.long).unsqueeze(1)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                dit(
                    txt=hist_cat,
                    txt_shape=hist_shape,
                    txt_q_shape=hist_shape,
                    timestep=torch.zeros(hist_cat.shape[0], device=device, dtype=torch.bfloat16),
                    update_kv=True,
                    use_kv_cache=True,
                )
            k_shape_c = hist_shape + block_size
            q_shape_c = torch.full((len(cond_txt), 1), block_size, device=device, dtype=torch.long)
            txt_c = torch.cat(cond_txt, dim=0).to(torch.bfloat16)
            ts_c = torch.tensor(cond_ts, device=device).repeat_interleave(block_size).to(torch.bfloat16)
            preds.append(_block_fwd(txt_c, k_shape_c, q_shape_c, ts_c, True))
            targets.append(torch.cat(cond_targets, dim=0))
        if unc_txt:
            q_shape_u = torch.full((len(unc_txt), 1), block_size, device=device, dtype=torch.long)
            txt_u = torch.cat(unc_txt, dim=0).to(torch.bfloat16)
            ts_u = torch.tensor(unc_ts, device=device).repeat_interleave(block_size).to(torch.bfloat16)
            preds.append(_block_fwd(txt_u, q_shape_u, q_shape_u, ts_u, False))
            targets.append(torch.cat(unc_targets, dim=0))

        target = torch.cat(targets, dim=0)
        pred_rows = torch.cat(preds, dim=0)
        loss = F.mse_loss(pred_rows.float(), target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(dit.parameters(), args.grad_clip)
        opt.step()
        # The cache must stay populated through backward() above (gradient
        # checkpointing recomputes the conditional forward, which reads it);
        # it is safe to drop only now.
        for blk in dit.blocks:
            blk.set_kv_cache(False)

        loss_hist[r].append(loss.item())
        tgt_rms_hist.append(target.pow(2).mean().sqrt().item())
        if (step + 1) % args.log_every == 0:
            msg = " | ".join(
                f"R{k}: {sum(v[-args.log_every:]) / max(1, len(v[-args.log_every:])):.4f}"
                for k, v in loss_hist.items()
            )
            tgt_rms = sum(tgt_rms_hist[-args.log_every :]) / len(tgt_rms_hist[-args.log_every :])
            el = time.time() - t_start
            print(f"[train] step {step + 1}/{args.max_steps} lr={lr_at(step, args):.2e} {msg} tgt_rms={tgt_rms:.3f} ({el:.0f}s)")

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
