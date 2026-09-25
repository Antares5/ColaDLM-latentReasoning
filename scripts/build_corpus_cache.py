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
"""Build a VAE-latent training stream from raw text corpora.

Reads parquet files (fineweb rows, column ``text``), tokenises with the
ColaDLM tokenizer, packs into seq_len windows, VAE-encodes to latents
(bf16) on the assigned GPU shard, and writes one ``.pt`` shard per GPU.
``--merge_with`` additionally concatenates an existing stream (e.g. the
wikitext cache) on rank 0 after encoding.

Launch one process per GPU:
    CUDA_VISIBLE_DEVICES=$G python scripts/build_corpus_cache.py \
        --parquet_glob "corpus/fw_*.parquet" --shard_id $G --num_shards 4 ...
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cola_dlm import ColaTextVAEModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet_glob", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--out_dir", default="pretrain_cache")
    p.add_argument("--tag", default="fineweb")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_rows", type=int, default=0, help="0 = all")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import pyarrow.parquet as pq

    device = torch.device("cuda")
    files = sorted(glob.glob(args.parquet_glob))
    assert files, f"no files match {args.parquet_glob}"
    print(f"[shard {args.shard_id}] {len(files)} parquet files", flush=True)

    # ---- tokenise + pack (each shard reads its strided slice of rows) ----
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    ids: list[int] = []
    row_i = 0
    t0 = time.time()
    for f in files:
        table = pq.read_table(f, columns=["text"])
        texts = table.column("text").to_pylist()
        for text in texts:
            if row_i % args.num_shards == args.shard_id:
                if text and len(text) > 200:
                    ids.extend(tokenizer.encode(text).ids)
            row_i += 1
            if args.max_rows and row_i >= args.max_rows * args.num_shards:
                break
        if args.max_rows and row_i >= args.max_rows * args.num_shards:
            break
        print(f"[shard {args.shard_id}] {f}: cum tokens {len(ids) / 1e6:.0f}M ({time.time() - t0:.0f}s)", flush=True)
    n_win = len(ids) // args.seq_len
    windows = torch.tensor(ids[: n_win * args.seq_len], dtype=torch.long).view(n_win, args.seq_len)
    print(f"[shard {args.shard_id}] packed {n_win} windows ({len(ids) / 1e6:.0f}M tokens)", flush=True)
    del ids

    # ---- VAE encode on this shard's GPU ----
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
    scale, shift = vae.scaling_factor, vae.shifting_factor
    out = torch.empty(n_win, args.seq_len, vae.latent_dim, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, n_win, 4):
            chunk = [w.to(device) for w in windows[i : i + 4]]
            enc = vae.encode(chunk)
            for j, lat in enumerate(enc.latents_list):
                out[i + j] = ((lat - shift) * scale).to(torch.bfloat16).cpu()
            if (i // 4) % 500 == 0:
                print(f"[shard {args.shard_id}] encoded {i}/{n_win} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.tag}_shard{args.shard_id}_z0.pt")
    torch.save(out, path)
    print(f"[shard {args.shard_id}] saved {path} {tuple(out.shape)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
