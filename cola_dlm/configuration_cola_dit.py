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

from transformers import PretrainedConfig


class ColaDiTConfig(PretrainedConfig):
    """Configuration for :class:`ColaDiTModel`.

    Parameterizes the **block-causal Diffusion Transformer prior**
    ``p_psi(z_0)`` of Cola DLM (Eq. 2.1.4 of the paper *Continuous
    Latent Diffusion Language Model*, arXiv:2605.06548). The DiT learns
    the vector field ``v_psi(z_t, t; z_0^(<b))`` of the continuous-flow
    prior (Eq. 2.1.2 / 2.1.7) under the visible set
    ``V_b = {sg(z_0^(<b)), z_t^(b)}`` (Eq. 2.2.3).

    Key knobs (in paper notation):

    * ``txt_in_channels`` / ``txt_out_channels``: the latent dimension
      ``d`` of ``z_0 ∈ R^d`` (Eq. 2.1.1). Must agree with the VAE
      ``latent_dim``.
    * ``txt_dim`` / ``emb_dim``: hidden width of the DiT trunk and the
      AdaLN timestep embedding.
    * ``heads`` / ``head_dim`` / ``expand_ratio``: attention and SwiGLU
      FFN shape.
    * ``num_layers``: depth of the DiT trunk.
    * ``patch_size``: 1-D patchification factor over the latent
      sequence axis (1 in the released checkpoint).
    * ``rope_dim``: number of channels per head that receive RoPE
      positional encoding.
    * ``block_size``: per-sample factor that defines how the latent
      sequence is split into ``B`` generation blocks
      ``z_0 = (z_0^(1), ..., z_0^(B))`` (Eq. 2.1.4). Concretely, both
      the block-causal attention pattern (visible set ``V_b``) and the
      inference loop in :func:`generate_task_repaint_inference` advance
      one ``block_size`` chunk at a time.

    Defaults match the released ~1.8B-parameter DiT prior from the
    paper's RQ4 scaling curve (24 layers × 16 heads × 128 head_dim).
    """

    model_type = "cola_dit"

    def __init__(
        self,
        txt_in_channels: int = 16,
        txt_out_channels: int = 16,
        txt_dim: int = 2048,
        emb_dim: int = 2048,
        heads: int = 16,
        head_dim: int = 128,
        expand_ratio: int = 4,
        num_layers: int = 24,
        norm_eps: float = 1e-5,
        qk_bias: bool = False,
        patch_size: int = 1,
        rope_dim: int = 96,
        block_size: int = 4,
        loop_reinject: bool = False,
        loop_cond: str = "global",
        loop_lora_rank: int = 0,
        loop_adapter: bool = False,
        loop_renorm: bool = False,
        **kwargs,
    ):
        self.txt_in_channels = txt_in_channels
        self.txt_out_channels = txt_out_channels
        self.txt_dim = txt_dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = head_dim
        self.expand_ratio = expand_ratio
        self.num_layers = num_layers
        self.norm_eps = norm_eps
        self.qk_bias = qk_bias
        self.patch_size = patch_size
        self.rope_dim = rope_dim
        self.block_size = block_size
        # Loop-transformer-style input re-injection: at each depth-loop
        # iteration r >= 1, add the initial txt_in representation back to
        # the hidden state (h <- Stack(h + x_emb)). Persisted in the
        # checkpoint config; env COLA_DIT_LOOP_REINJECT overrides at load.
        self.loop_reinject = loop_reinject
        # Per-loop conditioning mode: "global" (one loop_emb vector shared
        # by all layers), "layer" (per-layer loop embedding, the AdaLN
        # native form of per-loop LayerNorm), "film" (per-loop per-layer
        # FiLM on the residual stream). Persisted in the checkpoint config;
        # env COLA_DIT_LOOP_COND overrides at load.
        self.loop_cond = loop_cond
        # Per-loop LoRA rank on attention/MLP projections (0 = off).
        # Persisted in the checkpoint config; env COLA_DIT_LOOP_LORA_RANK
        # overrides at load.
        self.loop_lora_rank = loop_lora_rank
        # Huginn-style concat adapter for input injection at loop
        # iterations r >= 1 (h <- A([h; x_emb]), A init [I | 0]).
        # Persisted in the checkpoint config; env COLA_DIT_LOOP_ADAPTER
        # overrides at load.
        self.loop_adapter = loop_adapter
        # RMS renormalisation of the hidden state at loop boundaries
        # (anti-collapse); env COLA_DIT_LOOP_RENORM overrides at load.
        self.loop_renorm = loop_renorm
        super().__init__(**kwargs)
