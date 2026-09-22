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
"""Continuous pretraining of the Cola DiT prior with depth recurrence.

Replicates the Huginn (Geiping et al. 2025, arXiv:2502.05171) training
scheme on top of the released ColaDLM checkpoint, keeping the native
conditional Flow Matching objective (cache-consistent two-pass layout
from train_depth_loop.py):

* random recurrence count r per training step from a log-normal Poisson
  distribution (Huginn Eq. 1-2), one shared r per step across DDP ranks
  ("locked-step sampling");
* truncated backprop through the loop: only the last ``--tbptt_k``
  iterations carry gradients (COLA_DIT_TBPTT_K);
* concat + learned adapter input injection (h <- A([h; x_emb]), A init
  to [I | 0]) instead of additive re-injection;
* warmup -> constant learning rate;
* packed token stream (documents concatenated, chunked into seq_len
  windows), VAE latents pre-encoded once and cached to disk in bf16;
* per-step validation FM loss at R in {1, 2, 4, 8} on held-out windows
  (the Huginn Fig. 6 analogue: all recurrence depths should improve
  together) plus a per-loop hidden-state probe (RMS / token correlation)
  to watch for Huginn-style state collapse.

Launch (4 GPUs):
    torchrun --nproc_per_node=4 scripts/pretrain_loop.py \
        --dit_path hf_models/cola_dlm/cola_dit \
        --vae_path hf_models/cola_dlm/cola_vae \
        --tokenizer_path hf_models/tokenizer.json \
        --out_dir ckpts/pretrain_v8 --max_steps 30000
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

BLOCK_SIZE = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit_path", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    p.add_argument("--dataset_lines", type=int, default=0, help="0 = full split")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cfg_dropout", type=float, default=0.1)
    p.add_argument("--T", type=float, default=1000.0)
    p.add_argument("--r_mean", type=float, default=3.0, help="Mean recurrence of the log-normal Poisson sampler")
    p.add_argument("--r_sigma", type=float, default=0.5)
    p.add_argument("--r_max", type=int, default=8)
    p.add_argument("--r_dist", choices=["lognormal", "uniform"], default="lognormal")
    p.add_argument("--r_choices", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--tbptt_k", type=int, default=2)
    p.add_argument("--adapter", type=int, default=1, help="Enable the concat loop adapter (0 = off)")
    p.add_argument("--bf16", type=int, default=0, help="Load/train the model in pure bf16 (halves weight+grad memory)")
    p.add_argument("--loop_renorm", type=int, default=0,
                   help="RMS-renormalise the hidden state at every loop boundary (anti-collapse)")
    p.add_argument("--cond_lr_mult", type=float, default=1.0)
    p.add_argument("--val_windows", type=int, default=64)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--probe_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--keep_states", type=int, default=2, help="Optimizer-state checkpoints to keep")
    p.add_argument("--cache_dir", default="pretrain_cache")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--optimizer", default="adamw8bit", choices=["adamw8bit", "adamw"])
    p.add_argument("--resume", default="", help="Path to a state checkpoint dir to resume from")
    return p.parse_args()


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank_info() -> tuple[int, int, int]:
    if is_dist():
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    return 0, 1, 0


def lr_at(step: int, args) -> float:
    """Warmup then constant (Huginn schedule)."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    return args.lr


def sample_r(step: int, args) -> int:
    """Recurrence count for this step. Lognormal: log-normal Poisson
    (Huginn Eq. 1-2). Uniform: uniform over ``args.r_choices``. Seeded
    per step so every DDP rank draws the same r (locked-step sampling)."""
    g = torch.Generator().manual_seed(args.seed * 100003 + step)
    if args.r_dist == "uniform":
        i = int(torch.randint(0, len(args.r_choices), (1,), generator=g).item())
        return int(args.r_choices[i])
    tau = torch.randn(1, generator=g).item() * args.r_sigma + (math.log(args.r_mean) - 0.5 * args.r_sigma**2)
    lam = math.exp(tau)
    r = torch.poisson(torch.tensor([lam]), generator=g).item() + 1
    return int(min(args.r_max, max(1, round(r))))


def build_packed_windows(tokenizer: Tokenizer, args, rank: int) -> list[torch.Tensor]:
    """Tokenise the split, concatenate the id stream, chunk into seq_len windows."""
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split="train")
    n_lines = len(ds) if args.dataset_lines <= 0 else min(args.dataset_lines, len(ds))
    ids: list[int] = []
    for line in ds.select(range(n_lines))["text"]:
        ids.extend(tokenizer.encode(line).ids)
    if rank == 0:
        print(f"[data] token stream: {len(ids)} tokens", flush=True)
    n_win = len(ids) // args.seq_len
    return [
        torch.tensor(ids[i * args.seq_len : (i + 1) * args.seq_len], dtype=torch.long)
        for i in range(n_win)
    ]


def encode_latents(vae, windows, device, scale, shift, rank: int) -> torch.Tensor:
    """VAE-encode all windows -> (N, seq_len, latent_dim) bf16 CPU tensor."""
    out = torch.empty(len(windows), windows[0].shape[0], vae.latent_dim, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, len(windows), 4):
            chunk = [w.to(device) for w in windows[i : i + 4]]
            enc = vae.encode(chunk)
            for j, lat in enumerate(enc.latents_list):
                out[i + j] = ((lat - shift) * scale).to(torch.bfloat16).cpu()
            if rank == 0 and (i // 4) % 2000 == 0:
                print(f"[data] encoded {i + len(chunk)}/{len(windows)} windows", flush=True)
    return out


@torch.no_grad()
def _commit_history(dit, hist, device):
    """Commit clean history to the KV cache at t=0 (no_grad, own autocast
    context so its weight casts are never reused by the grad pass)."""
    h = hist.to(torch.bfloat16)
    h_shape = torch.tensor([[h.shape[0]]], device=device, dtype=torch.long)
    # cache_enabled=False: the per-context bf16 weight-cast cache would
    # otherwise hold a full second copy of the model (~3.6GB) for the
    # whole commit pass.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
        dit(txt=h, txt_shape=h_shape, txt_q_shape=h_shape,
            timestep=torch.zeros(h.shape[0], device=device, dtype=torch.bfloat16),
            update_kv=True, use_kv_cache=True)
    return h_shape


def drift_twopass(dit, z_t, hist, t, device):
    """Cache-consistent conditional drift of one (block, d) cell.

    ``dit`` may be a DDP wrapper; the forward that will carry gradients
    must go through it. History commit and the Q forward use separate
    autocast contexts (autocast caches bf16 weight casts per context;
    sharing a context with the no_grad commit would sever weight grads).
    """
    raw = getattr(dit, "module", dit)
    for blk in raw.blocks:
        blk.set_kv_cache(True)
    if hist is not None and hist.shape[0] > 0:
        h_shape = _commit_history(dit, hist, device)
        k_shape = h_shape + BLOCK_SIZE
    else:
        k_shape = torch.tensor([[BLOCK_SIZE]], device=device, dtype=torch.long)
    q_shape = torch.tensor([[BLOCK_SIZE]], device=device, dtype=torch.long)
    ts = torch.full((BLOCK_SIZE,), t, device=device, dtype=torch.bfloat16)
    # cache_enabled=True here (grad pass): the per-context cast cache holds
    # ONE bf16 cast per weight, reused across all loop iterations. With the
    # cache disabled, every loop re-casts and autograd saves a separate
    # ~3.6GB set of casts per loop (14GB at r=4).
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = dit(txt=z_t.to(torch.bfloat16), txt_shape=k_shape, txt_q_shape=q_shape,
                  timestep=ts, update_kv=False, use_kv_cache=hist is not None and hist.shape[0] > 0).txt_sample
    for blk in raw.blocks:
        blk.set_kv_cache(False)
    return out.float()


@torch.no_grad()
def validate(dit, val_z0, args, device) -> dict[int, float]:
    """Mean conditional FM loss at R in {1,2,4,8} on fixed (window, block, t) cells."""
    n_blocks = args.seq_len // BLOCK_SIZE
    losses: dict[int, list[float]] = {r: [] for r in (1, 2, 4, 8) if r <= dit.max_depth_loops}
    cells = []
    for i in range(len(val_z0)):
        g = torch.Generator().manual_seed(args.seed + i)
        b_idx = 1 + int(torch.randint(0, n_blocks - 1, (1,), generator=g).item())
        for t in (200.0, 500.0, 800.0):
            cells.append((i, b_idx, t))
    for r in losses:
        dit.depth_loops = r
        for i, b_idx, t in cells:
            z0 = val_z0[i].to(device).float()
            z0_blk = z0[b_idx * BLOCK_SIZE : (b_idx + 1) * BLOCK_SIZE]
            g = torch.Generator(device=device).manual_seed(args.seed + i * 13 + int(t))
            z1 = torch.randn(z0_blk.shape, device=device, generator=g)
            z_t = (1.0 - t / args.T) * z0_blk + (t / args.T) * z1
            pred = drift_twopass(dit, z_t, z0[: b_idx * BLOCK_SIZE], t, device)
            losses[r].append(F.mse_loss(pred, z1 - z0_blk).item())
    return {r: sum(v) / len(v) for r, v in losses.items()}


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    rank, world, local = rank_info()
    # NOTE: init_process_group is deferred until after the (potentially
    # ~1h) VAE encode: ranks 1..N would otherwise hit the default 600s
    # store timeout waiting for rank 0. The encode decision uses the
    # torchrun-provided LOCAL_RANK before any NCCL setup.
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    n_blocks = args.seq_len // BLOCK_SIZE
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Recurrence machinery via env (construction-time switches).
    os.environ["COLA_DIT_TBPTT_K"] = str(args.tbptt_k)
    os.environ["COLA_DIT_LOOP_ADAPTER"] = "1" if args.adapter else "0"
    if args.loop_renorm:
        os.environ["COLA_DIT_LOOP_RENORM"] = "1"

    # ---------------- data: pack, encode, cache ----------------
    tag = f"{args.dataset_config}_seq{args.seq_len}_lines{args.dataset_lines}"
    cache_path = os.path.join(args.cache_dir, f"{tag}_z0.pt")
    meta_path = os.path.join(args.cache_dir, f"{tag}_meta.json")
    if rank == 0 and not os.path.exists(cache_path):
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        windows = build_packed_windows(tokenizer, args, rank)
        print(f"[data] {len(windows)} packed windows of {args.seq_len} tokens", flush=True)
        vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
        z0_stream = encode_latents(vae, windows, device, vae.scaling_factor, vae.shifting_factor, rank)
        torch.save(z0_stream, cache_path)
        with open(meta_path, "w") as f:
            json.dump({"n_windows": len(windows), "seq_len": args.seq_len}, f)
        print(f"[data] latent stream cached -> {cache_path} ({z0_stream.shape})", flush=True)
        del vae, windows, z0_stream
        torch.cuda.empty_cache()
    if is_dist():
        import datetime

        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=3))
        torch.distributed.barrier()
    z0_all = torch.load(cache_path)  # (N, seq_len, d) bf16, CPU
    n_train = len(z0_all) - args.val_windows
    train_z0, val_z0 = z0_all[:n_train], z0_all[n_train:]
    # rank-sharded window order (reshuffled every epoch by seeded RNG)
    shard = list(range(rank, n_train, world))
    if rank == 0:
        print(f"[data] train windows {n_train}, val windows {len(val_z0)}, shard/rank {len(shard)}", flush=True)

    # ---------------- model ----------------
    if rank == 0:
        print(f"[train] loading DiT: {args.dit_path}", flush=True)
    load_dtype = torch.bfloat16 if args.bf16 else torch.float32
    dit = ColaDiTModel.from_pretrained(args.dit_path, torch_dtype=load_dtype).to(device)
    if args.adapter:
        dit.config.loop_adapter = True  # persist into saved checkpoints
    if args.loop_renorm:
        dit.config.loop_renorm = True  # persist into saved checkpoints
    dit.train()
    assert args.r_max <= dit.max_depth_loops

    cond_names = ("loop_emb", "loop_emb_layers", "loop_film", "lora_A", "lora_B", "loop_adapter")
    cond_params = [p for n, p in dit.named_parameters() if any(k in n for k in cond_names)]
    base_params = [p for n, p in dit.named_parameters() if not any(k in n for k in cond_names)]
    param_groups = [
        {"params": base_params, "lr_mult": 1.0},
        {"params": cond_params, "lr_mult": args.cond_lr_mult},
    ]
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.PagedAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        state = torch.load(os.path.join(args.resume, "trainer_state.pt"), map_location="cpu")
        start_step = state["step"]
        opt.load_state_dict(state["opt"])
        if rank == 0:
            print(f"[train] resumed from {args.resume} at step {start_step}", flush=True)

    if is_dist():
        dit = torch.nn.parallel.DistributedDataParallel(
            dit, device_ids=[local], find_unused_parameters=True
        )
    model = dit.module if is_dist() else dit

    loss_hist: dict[int, list[float]] = {r: [] for r in range(1, args.r_max + 1)}
    state_ckpts: list[str] = []
    t_start = time.time()
    for step in range(start_step, args.max_steps):
        epoch = step // max(1, len(shard))
        g = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(shard), generator=g).tolist()
        win_idx = shard[order[step % len(shard)]]
        r = sample_r(step, args)
        model.depth_loops = r
        lr_now = lr_at(step, args)
        for gparam in opt.param_groups:
            gparam["lr"] = lr_now * gparam["lr_mult"]

        z0 = train_z0[win_idx].to(device, non_blocking=True).float()
        b_idx = random.randint(0, n_blocks - 1)
        drop_hist = random.random() < args.cfg_dropout
        z0_blk = z0[b_idx * BLOCK_SIZE : (b_idx + 1) * BLOCK_SIZE]
        z1 = torch.randn_like(z0_blk)
        t = random.uniform(0.0, args.T)
        z_t = (1.0 - t / args.T) * z0_blk + (t / args.T) * z1
        target = z1 - z0_blk
        hist = None if (drop_hist or b_idx == 0) else z0[: b_idx * BLOCK_SIZE]

        pred = drift_twopass(dit, z_t, hist, t, device)
        loss = F.mse_loss(pred.float(), target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(dit.parameters(), args.grad_clip)
        opt.step()
        loss_hist[r].append(loss.item())

        # ---- per-loop hidden-state probe (rank 0 only) ----
        if rank == 0 and (step + 1) % args.probe_every == 0:
            probe: list = []
            model._probe = probe
            model.depth_loops = min(args.r_max, 8)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                drift_twopass(model, z_t, hist, t, device)
            model._probe = None
            stats = " ".join(f"L{li}:rms{rm:.1f}/c{co:+.2f}" for li, rm, co in probe[:7])
            print(f"[probe] step {step + 1} per-loop hidden rms/token-corr: {stats}", flush=True)

        if (step + 1) % args.log_every == 0 and rank == 0:
            msg = " ".join(
                f"R{k}:{sum(v[-args.log_every:]) / max(1, len(v[-args.log_every:])):.3f}"
                for k, v in loss_hist.items() if v
            )
            el = time.time() - t_start
            print(f"[train] step {step + 1}/{args.max_steps} lr={lr_now:.2e} {msg} ({el:.0f}s)", flush=True)

        # ---- validation by recurrence depth ----
        if (step + 1) % args.val_every == 0:
            model_eval = model
            model_eval.eval()
            val_losses = validate(model_eval, val_z0, args, device)
            model_eval.train()
            if rank == 0:
                vmsg = " ".join(f"R{k}:{v:.3f}" for k, v in val_losses.items())
                print(f"[val] step {step + 1} {vmsg}", flush=True)

        # ---- checkpointing (rank 0) ----
        if rank == 0 and ((step + 1) % args.save_every == 0 or step + 1 == args.max_steps):
            ckpt = os.path.join(args.out_dir, f"step{step + 1}")
            model.depth_loops = 1
            model.save_pretrained(ckpt)
            torch.save({"step": step + 1, "opt": opt.state_dict()}, os.path.join(ckpt, "trainer_state.pt"))
            state_ckpts.append(ckpt)
            while len(state_ckpts) > args.keep_states:
                old = state_ckpts.pop(0)
                old_state = os.path.join(old, "trainer_state.pt")
                if os.path.exists(old_state):
                    os.remove(old_state)
            print(f"[train] saved {ckpt}", flush=True)

    if rank == 0:
        summary = {
            "args": vars(args),
            "final_loss": {str(k): (sum(v[-50:]) / max(1, len(v[-50:]))) for k, v in loss_hist.items()},
            "wall_time_s": time.time() - t_start,
        }
        with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[train] done in {summary['wall_time_s']:.0f}s -> {args.out_dir}", flush=True)
    if is_dist():
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
