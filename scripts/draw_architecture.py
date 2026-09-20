# Draw the Cola DLM architecture diagram with tensor shapes (data formats).
# Output: architecture_diagram.png in the repo root.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for f in [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
]:
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Noto Serif CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_INPUT = "#FFF3D6"   # data / tensors
C_VAE = "#D6E9FF"     # VAE modules
C_DIT = "#FFE0E0"     # DiT modules
C_OP = "#E3F3E3"      # ops / sampling
C_EDGE = "#444444"


def box(ax, x, y, w, h, text, fc, fontsize=10, title=None, title_fs=11, ec="#333333"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        fc=fc, ec=ec, lw=1.4, zorder=2,
    )
    ax.add_patch(p)
    if title:
        ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + (h - 5.2) / 2, text, ha="center", va="center",
                fontsize=fontsize, zorder=3, linespacing=1.55)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, zorder=3, linespacing=1.55)
    return (x, y, w, h)


def arrow(ax, x1, y1, x2, y2, label=None, fs=9.5, color=C_EDGE, style="-|>",
          lw=1.8, label_dx=0, label_dy=2.2, connectionstyle=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                        color=color, lw=lw, zorder=4,
                        connectionstyle=connectionstyle or "arc3,rad=0")
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy, label,
                ha="center", va="center", fontsize=fs, color="#8a4b00", zorder=5,
                fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))


fig, ax = plt.subplots(figsize=(20, 13.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 98.2, "Cola DLM 架构与数据形式（Text VAE 0.5B + 分块因果 DiT 先验 1.8B）",
        ha="center", va="center", fontsize=17, fontweight="bold")
ax.text(50, 95.4,
        "层次化隐变量模型:  p(x, z₀) = p_θ(x | z₀) · p_ψ(z₀)，"
        "z₀ 为连续隐序列（latent_dim = 16），推理 = 前缀编码 → 分块先验传输 → 条件解码",
        ha="center", va="center", fontsize=11, color="#555555")

# ================= Row 1: end-to-end pipeline =================
b1 = box(ax, 1.5, 79, 13, 12,
         "token ids  x^pre\n(L_pre,)  int64\n词表 100278\npad 到 16 的倍数",
         C_INPUT, fontsize=9.5, title="① 输入 prompt")

b2 = box(ax, 17.5, 79, 15.5, 12,
         "4 × Transformer\n(d=1536, 12 heads)\nRoPE(θ=5e5), QK-norm\n"
         "head → (n_pre, 32)\n= mean ‖ logvar",
         C_VAE, fontsize=9.5, title="② VAE Encoder  q_φ(z|x)")

b3 = box(ax, 36, 76.5, 27, 14.5,
         "for b = 1..B（每块 16 个 latent）:\n"
         "  ε^(b) ~ N(0, I)        (16, 16)\n"
         "  16 步 Euler:  z_{t-Δ} = z_t − Δ/T · v_ψ\n"
         "  t: 1000 → 0，CFG = 7.0（cond+uncond 两次前向）\n"
         "  输出 ẑ₀^(b)          (16, 16)\n"
         "  提交进 DiT / Decoder 的 KV cache",
         C_DIT, fontsize=9.5, title="③ 分块先验传输  Φ^ψ_{0←1}（DiT 1.8B）")

b4 = box(ax, 66, 79, 15.5, 12,
         "4 × Transformer\n(d=1536, 12 heads)\n"
         "z → Linear → vocab\n"
         "logits (16B, 100278)",
         C_VAE, fontsize=9.5, title="④ VAE Decoder  p_θ(x|z₀)")

b5 = box(ax, 84.5, 79, 14, 12,
         "greedy / top-k / top-p\nrepetition penalty\n"
         "x̂  (L_new,) int64\n遇 EOS 或 32 tokens 停止",
         C_OP, fontsize=9.5, title="⑤ 采样输出")

arrow(ax, 14.5, 85, 17.5, 85)
arrow(ax, 33, 85, 36, 85)
arrow(ax, 63, 85, 66, 85)
arrow(ax, 81.5, 85, 84.5, 85)
ax.text(34.5, 92.6, "z^pre  (n_pre, 16)  fp32", ha="center", va="center", fontsize=9.5,
        color="#8a4b00", fontweight="bold",
        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
ax.text(64.5, 92.6, "[z^pre, ẑ₀^(1:B)]  (n_pre+16B, 16)", ha="center", va="center",
        fontsize=9.5, color="#8a4b00", fontweight="bold",
        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))

# ================= Row 2 left: DiT detail =================
panel = FancyBboxPatch((1.5, 7), 55, 64, boxstyle="round,pad=0.4,rounding_size=1.5",
                       fc="#FFF9F9", ec="#CC8888", lw=1.6, zorder=1)
ax.add_patch(panel)
ax.text(29, 68.2, "ColaDiTModel —— 分块因果隐先验 p_ψ(z₀)（~1.8B）",
        ha="center", fontsize=13, fontweight="bold", color="#8a2e2e")

d_in = box(ax, 3.5, 58.5, 24, 6.5,
           "输入: 噪声块 z_t^(b)  (L_q, 16)\n+ txt_shape (B,1) + timestep t (B,)",
           C_INPUT, fontsize=9.5)
d_patch = box(ax, 3.5, 49.5, 24, 6.5,
              "patchify (patch_size=1)\nLinear 16 → 2048\ntimestep: sin(t) embedding (B,)",
              C_DIT, fontsize=9.5)
d_blk = box(ax, 3.5, 24.5, 24, 22.5,
            "× 24 层\n\n"
            "RMSNorm (pre-norm)\n"
            "AdaLN(t): scale/shift 条件化\n"
            "分块因果 MSA: 16 heads × 128\n"
            "  RoPE 作用前 96 通道\n"
            "  块内双向 / 块间因果（可见集 V_b）\n"
            "  KV cache: (L_kv, 16, 128) /层\n"
            "SwiGLU FFN: 2048 → 8192 → 2048",
            C_DIT, fontsize=9.5, title="Transformer block")
d_out = box(ax, 3.5, 15.5, 24, 6.5,
            "RMSNorm + Linear 2048 → 16\n输出漂移 v_ψ  (L_q, 16)",
            C_DIT, fontsize=9.5)
arrow(ax, 15.5, 58.5, 15.5, 56.2)
arrow(ax, 15.5, 49.5, 15.5, 47.2)
arrow(ax, 15.5, 24.5, 15.5, 22.2)

# CFG / mask note
box(ax, 30.5, 40, 24.5, 25,
    "每一步两次前向（CFG）:\n"
    "  cond:   K/V = [z^pre, ẑ₀^(<b), z_t^(b)]\n"
    "  uncond: K/V = [z_t^(b)]（空前缀）\n"
    "  v = uncond + 7.0 × (cond − uncond)\n\n"
    "分块因果 mask（式 2.2.3 可见集 V_b）:\n"
    "  Q 块 b_q 只见 K 块 b_k ≤ b_q\n"
    "  样本间完全阻断\n"
    "  加性 mask: 0 / dtype.min",
    "#FFFFFF", fontsize=9.5, title="CFG 与可见性约束", title_fs=10.5, ec="#CC8888")

box(ax, 30.5, 24.5, 24.5, 13,
    "NA flatten-concat 布局（无 padding）:\n"
    "  txt:        (L_total, c)，L_total = Σ n_i\n"
    "  txt_shape:  (B, 1)  各样本 K 侧长度\n"
    "  txt_q_shape:(B, 1)  Q 侧长度（=16/样本）",
    "#FFFFFF", fontsize=9.5, title="数据排布", title_fs=10.5, ec="#CC8888")

box(ax, 30.5, 15.5, 24.5, 6.5,
    "隐先验的块因果分解:\np_ψ(z₀) = p_ψ(z₀⁽¹⁾) · Π p_ψ(z₀⁽ᵇ⁾ | z₀⁽<ᵇ⁾)",
    "#FFFFFF", fontsize=9.5, ec="#CC8888")

# ================= Row 2 right: VAE detail =================
panel2 = FancyBboxPatch((59.5, 7), 39, 64, boxstyle="round,pad=0.4,rounding_size=1.5",
                        fc="#F7FBFF", ec="#7799CC", lw=1.6, zorder=1)
ax.add_patch(panel2)
ax.text(79, 68.2, "ColaTextVAEModel —— q_φ 编码器 + p_θ 解码器（~0.5B）",
        ha="center", fontsize=13, fontweight="bold", color="#2e4e8a")

box(ax, 61.5, 55, 35, 10,
    "tokens (L,) → Embedding (100278 × 1536)\n"
    "4 × Transformer: d=1536, 12 heads, ffn 6144\n"
    "RoPE(θ=5e5) + QK-norm, SwiGLU, 因果注意力\n"
    "LN → Linear 1536 → 32（mean ‖ logvar）",
    C_VAE, fontsize=9.5, title="Encoder  q_φ(z₀|x)", title_fs=10.5)

box(ax, 61.5, 45.5, 35, 6.5,
    "后验采样 / mode → z: (n, 16)，n = L / patch_size（=1，不压缩）\n"
    "归一化: z = (z − shift) × scale（本 checkpoint: 0 / 1）",
    C_INPUT, fontsize=9.5)

box(ax, 61.5, 28.5, 35, 14,
    "z (L_q, 16) → Linear 16 → 1536\n"
    "4 × Transformer（同构, KV cache 同 DiT 约定）\n"
    "以 [z^pre, ẑ₀^(1:B)] 为条件做分块因果注意力\n"
    "Linear 1536 → 100278\n"
    "logits: (L_q, 100278)",
    C_VAE, fontsize=9.5, title="Decoder  p_θ(x|z₀)", title_fs=10.5)

box(ax, 61.5, 15.5, 35, 10,
    "latent 位置标签:\n"
    "  1 = prompt（来自 q_φ 前缀编码）\n"
    "  2 = 待生成（DiT 先验传输得到）\n"
    "  3 = 硬 pad（尾部裁剪，不进入计算）",
    C_OP, fontsize=9.5, title="标签约定", title_fs=10.5)

arrow(ax, 79, 55, 79, 52.2)
arrow(ax, 79, 45.5, 79, 42.7)

# connect row1 modules to detail panels
arrow(ax, 25, 79, 25, 71.6, color="#7799CC", style="-|>", lw=1.4)
arrow(ax, 49.5, 76.5, 49.5, 71.6, color="#CC8888", style="-|>", lw=1.4)
arrow(ax, 73.5, 79, 85, 71.6, color="#7799CC", style="-|>", lw=1.4)

# bottom footnote
ax.text(50, 3.2,
        "checkpoint: 2000 EFLOPs（RQ4 最大节点） | DiT: block_size=16, 24 层, txt_dim=2048 | "
        "VAE: latent_dim=16, patch_size=1（序列不压缩） | 全程 bf16 autocast, NA 无填充布局",
        ha="center", va="center", fontsize=10, color="#666666")

fig.savefig("architecture_diagram.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved architecture_diagram.png")
