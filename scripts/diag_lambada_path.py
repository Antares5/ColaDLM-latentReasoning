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
"""Lambada-path diagnostic: real task data + verified two-pass generation.

Replicates inference.py's lambada preprocessing exactly (tokenise, pad to
block alignment with pad_token_id, VAE encode, latent labels, prefix /
first-gen-block split with label-1 pinning), but generates with the
standalone two-pass mechanics from diag_oracle_history.py — the same
mechanics that produce coherent text at R>1 on wikitext. Scores the
lambada first word like acc_calc.

This bisects the R>1 benchmark collapse:
  * fails here  -> the trigger is in the task data pipeline
                   (template / pad / encode / strip) interacting with loops;
  * works here  -> the trigger is inside inference.py's generation loop
                   itself, not the data and not the DiT loop mechanics.

Example:
    python scripts/diag_lambada_path.py \
        --dit_path ckpts/depth_loop_r124_v2/step1000 \
        --vae_path hf_models/cola_dlm/cola_vae \
        --tokenizer_path hf_models/tokenizer.json \
        --input_jsonl generate_task_data/lambada.jsonl \
        --loops 1 2 --n_samples 40 --cfg 7.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cola_dlm import ColaDiTModel, ColaTextVAEModel
from cola_dlm.inference import apply_prompt_template
from scripts.diag_oracle_history import commit_history, cond_drift

PAD_TOKEN_ID = 100277


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit_path", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--n_samples", type=int, default=40)
    p.add_argument("--loops", type=int, nargs="+", default=[1, 2])
    p.add_argument("--cfg", type=float, default=7.0)
    p.add_argument("--timestep_num", type=int, default=16)
    p.add_argument("--n_gen_blocks", type=int, default=2)
    p.add_argument("--T", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=66)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    block_size = 16

    items = [json.loads(l) for l in open(args.input_jsonl)][: args.n_samples]
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
    scale, shift = vae.scaling_factor, vae.shifting_factor
    dit = ColaDiTModel.from_pretrained(args.dit_path).to(device).eval()
    timesteps = torch.linspace(int(args.T), 0, args.timestep_num + 1)

    # ---- replicate inference.py Step 1-3 (lambada: template is identity) --
    samples = []
    for item in items:
        prompt_str = apply_prompt_template(
            task="lambada", context=item.get("context", ""),
            question=item.get("question", ""),
            answer=item.get("ground_truth", ""), choices=None,
        )
        ids = tokenizer.encode(prompt_str).ids
        p_pad = (block_size - len(ids) % block_size) % block_size
        t_labels = [1] * len(ids) + [3] * p_pad
        ids = ids + [PAD_TOKEN_ID] * p_pad
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            enc = vae.encode([torch.tensor(ids, dtype=torch.long, device=device)])
        lat = ((enc.latents_list[0] - shift) * scale).float()
        labels = torch.tensor(t_labels, dtype=torch.long, device=device)  # patch_size=1
        num_ones = int((labels == 1).sum().item())
        if num_ones % block_size != 0:
            start_idx = (num_ones // block_size) * block_size
            prefix = lat[:start_idx].clone()
            blk = lat[start_idx : start_idx + block_size].clone()
            blab = labels[start_idx : start_idx + block_size].clone()
            blab[blab == 3] = 2
            prompt_tok_in_blk = int((blab == 1).sum().item())
        else:
            prefix = lat[:num_ones].clone()
            blk = torch.randn(block_size, lat.shape[-1], device=device)
            blab = torch.full((block_size,), 2, dtype=torch.long, device=device)
            prompt_tok_in_blk = 0
        samples.append({
            "id": item.get("id"), "gt": item.get("ground_truth", ""),
            "prefix": prefix, "blk0": blk, "blab0": blab,
            "prompt_tokens": len(t_labels) - p_pad,
            "prompt_tok_in_blk": prompt_tok_in_blk,
        })

    gen = torch.Generator(device=device).manual_seed(args.seed)
    for r in args.loops:
        dit.depth_loops = r
        n_correct = 0
        for s in samples:
            hist_blocks: list[torch.Tensor] = [s["prefix"]] if s["prefix"].shape[0] > 0 else []
            pin = int((s["blab0"] == 1).sum().item())
            gen_text_ids: list[int] = []
            for b in range(args.n_gen_blocks):
                hist = torch.cat(hist_blocks, dim=0) if hist_blocks else None
                if hist is not None:
                    commit_history(dit, hist, device)
                z = torch.randn(block_size, s["blk0"].shape[-1], device=device, generator=gen)
                if b == 0 and pin > 0:
                    z[:pin] = s["blk0"][:pin]
                for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
                    dt = (float(t_curr) - float(t_next)) / args.T
                    if b == 0 and pin > 0:
                        ts_vec = torch.full((block_size,), float(t_curr), device=device)
                        ts_vec[:pin] = 0.0
                    else:
                        ts_vec = float(t_curr)
                    k_len = (hist.shape[0] if hist is not None else 0) + block_size
                    if hist is None:
                        # empty prefix: cache-free forward like the uncond branch
                        for blk_ in dit.blocks:
                            blk_.set_kv_cache(True)
                        drift = cond_drift(dit, z, ts_vec, block_size, block_size, device, cfg=0.0)
                    else:
                        drift = cond_drift(dit, z, ts_vec, k_len, block_size, device, cfg=args.cfg)
                    z = z - drift * dt
                    if b == 0 and pin > 0:
                        z[:pin] = s["blk0"][:pin]
                hist_blocks.append(z)
                full = torch.cat(hist_blocks, dim=0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dec = vae.decode(
                        z=full,
                        txt_shape=torch.tensor([[full.shape[0]]], device=device, dtype=torch.long),
                        txt_q_shape=torch.tensor([[full.shape[0]]], device=device, dtype=torch.long),
                        update_kv=False,
                    )
                ids = dec.view(-1, dec.shape[-1]).argmax(dim=-1).tolist()
                gen_text_ids = ids  # decode the whole sequence each block (cheap enough)
            for blk_ in dit.blocks:
                blk_.set_kv_cache(False)
            cont_ids = gen_text_ids[s["prompt_tokens"] :]
            cont_text = tokenizer.decode(cont_ids, skip_special_tokens=True).strip()
            first_word = cont_text.split()[0].strip(".,!?;:\"'()") if cont_text.split() else ""
            gt = s["gt"].strip().strip(".,!?;:\"'()")
            ok = first_word.lower() == gt.lower()
            n_correct += int(ok)
            if s["id"] in (1, 9) or len(samples) <= 4:
                print(f"[lambada-path] R={r} id={s['id']} gen={cont_text[:80]!r} gt={gt!r} ok={ok}")
        print(f"[lambada-path] R={r} cfg={args.cfg} first-word acc: "
              f"{n_correct}/{len(samples)} = {100.0 * n_correct / len(samples):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
