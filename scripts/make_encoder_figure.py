"""Generate frozen-vs-trained parameter accounting chart for the VLA encoder pipeline."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/figures/architecture"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

# Architecture as used in M6 / M8 (configs/m6_vla_aux_v0.yaml, m8b_bc_per_task_head_v0.yaml)
# hidden_dims = [256, 256, 128]
TRAINED = {
    "lang_proj\n(512→64)":   32_832,
    "image_proj\n(768→128)": 98_432,
    "task_head\n(64→3)":     195,
    "MLP trunk\n258→256→256→128→8": 166_024,
}
FROZEN = {
    "CLIP text tower\n(ViT-B/32)":   63_700_000,   # HuggingFace model card
    "CLIP vision tower\n(ViT-B/32)": 87_800_000,
}

trained_total = sum(TRAINED.values())
frozen_total = sum(FROZEN.values())
total = trained_total + frozen_total

# -----------------------------------------------------------------------------
# Figure: stacked-bar comparison + breakdown
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                gridspec_kw={"width_ratios": [1.0, 1.6]})

# (a) frozen vs trained totals on log scale
labels = ["Frozen\n(CLIP ViT-B/32)", "Trained\n(VLA head)"]
vals = [frozen_total, trained_total]
colors = ["#5b9bd5", "#ed7d31"]
bars = ax1.bar(labels, vals, color=colors, width=0.6)
ax1.set_yscale("log")
ax1.set_ylabel("# parameters (log)")
ax1.set_title("Where the parameters live")
for rect, v in zip(bars, vals):
    ax1.text(rect.get_x() + rect.get_width() / 2, v * 1.3,
             f"{v/1e6:.2f} M" if v >= 1e6 else f"{v/1e3:.0f} k",
             ha="center", fontsize=11, fontweight="bold")
ax1.text(0.5, 0.04, f"trained / total = {trained_total/total*100:.2f}%",
         transform=ax1.transAxes, ha="center", fontsize=9, color="#444")

# (b) breakdown of trained components — what's actually learned
items = [("CLIP text tower (frozen)",   63_700_000, "#5b9bd5"),
         ("CLIP vision tower (frozen)", 87_800_000, "#5b9bd5"),
         ("image_proj (768→128)",       98_432,     "#ed7d31"),
         ("MLP trunk + action head",    166_024,    "#ed7d31"),
         ("lang_proj (512→64)",         32_832,     "#ed7d31"),
         ("task_head (64→3, aux CE)",   195,        "#ed7d31")]
names = [n for n, _, _ in items]
sizes = [v for _, v, _ in items]
cols = [c for _, _, c in items]

y = np.arange(len(items))[::-1]
bars = ax2.barh(y, sizes, color=cols, edgecolor="none")
ax2.set_xscale("log")
ax2.set_yticks(y)
ax2.set_yticklabels(names)
ax2.set_xlabel("# parameters (log)")
ax2.set_title("Per-component parameter count")
for rect, v in zip(bars, sizes):
    label = f"{v/1e6:.1f} M" if v >= 1e6 else (f"{v/1e3:.1f} k" if v >= 1e3 else f"{v}")
    ax2.text(v * 1.3, rect.get_y() + rect.get_height() / 2, label,
             va="center", fontsize=9)

# legend
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#5b9bd5", label="Frozen (pretrained CLIP)"),
                    Patch(color="#ed7d31", label="Trained (this project)")],
           loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("VLA encoder + head · parameter accounting (hidden_dims=[256,256,128])",
             y=1.02, fontsize=11)
fig.tight_layout()

out = OUT / "frozen_vs_trained_params.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out.relative_to(ROOT)}")
print(f"trained = {trained_total:,}   frozen = {frozen_total:,}   ratio = {trained_total/total*100:.3f}%")
