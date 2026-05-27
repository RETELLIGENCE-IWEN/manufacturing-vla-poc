"""Generate portfolio figures from runs/ artifacts.

Outputs to docs/figures/{curves,comparison}/. No pandas dependency — csv stdlib.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT_CURVES = ROOT / "docs/figures/curves"
OUT_CMP = ROOT / "docs/figures/comparison"
OUT_CURVES.mkdir(parents=True, exist_ok=True)
OUT_CMP.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def read_curve(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                if v in (None, "", "nan"):
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                out.setdefault(k, []).append(fv)
    return out


def read_summary(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def policy_section(d: dict) -> dict:
    for k in ("phase_bc", "bc", "vla", "policy", "diffusion"):
        if k in d and isinstance(d[k], dict):
            return d[k]
    return {}


# -----------------------------------------------------------------------------
# Figure 1: language conditioning open-loop — M4 → M5 → M5.1 val MSE
# -----------------------------------------------------------------------------
def fig_lang_conditioning_curve():
    runs = [
        ("M4 single-task lang BC", RUNS / "m4_bc_lang_v0/training_curve.csv", "val_mse_norm"),
        ("M5 multi-task lang BC", RUNS / "m5_bc_lang_multitask_v0/training_curve.csv", "val_mse_norm"),
        ("M5.1 + aux task_id CE", RUNS / "m5_1_bc_lang_multitask_aux_v0/training_curve.csv", "val_mse_norm"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, path, key in runs:
        c = read_curve(path)
        y = c.get(key)
        if not y:
            continue
        ax.plot(c["epoch"][: len(y)], y, label=label, linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val MSE (normalized)")
    ax.set_yscale("log")
    ax.set_title("M4 → M5 → M5.1 · open-loop validation MSE")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_CURVES / "fig1_lang_conditioning_val_mse.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# Figure 2: M5.1 aux loss — val_task_acc converges to 1.0
# -----------------------------------------------------------------------------
def fig_task_acc():
    c = read_curve(RUNS / "m5_1_bc_lang_multitask_aux_v0/training_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(c["epoch"][: len(c["val_task_acc"])], c["val_task_acc"], linewidth=1.6, color="#286")
    ax.axhline(1.0, color="#888", linestyle=":", linewidth=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val task_acc (aux head)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("M5.1 · auxiliary task classification reaches 1.0")
    fig.tight_layout()
    out = OUT_CURVES / "fig2_m5_1_task_acc.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# Figure 3: M7 diffusion v0 vs v1 — eps-MSE
# -----------------------------------------------------------------------------
def fig_diffusion_curve():
    runs = [
        ("M7 v0 (200 ep, K=4, T=16)", RUNS / "m7_diffusion_v0/training_curve.csv"),
        ("M7.1 v1 (500 ep, K=2, T=24)", RUNS / "m7_diffusion_v1/training_curve.csv"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, path in runs:
        c = read_curve(path)
        y = c.get("val_diff_mse")
        if not y:
            continue
        ax.plot(c["epoch"][: len(y)], y, label=label, linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val eps-MSE")
    ax.set_yscale("log")
    ax.set_title("M7 → M7.1 · Diffusion Policy validation eps-MSE")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_CURVES / "fig3_diffusion_val_eps_mse.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# Figure 4: M3 line — closed-loop progress (grasped_once, mean_return)
# -----------------------------------------------------------------------------
def fig_m3_progression():
    bars = [
        ("State-only BC",           "runs/m3_bc_state/closedloop_eval/closedloop_summary.json"),
        ("Phase-aware 100",         "runs/m3_bc_phase_aware/closedloop_eval_final_hold_120step/closedloop_summary.json"),
        ("Phase-aware 5000",        "runs/m3_bc_phase_aware_5000/closedloop_eval_safe_120step/closedloop_summary.json"),
        ("DAgger v1",               "runs/m3_bc_dagger_v1/closedloop_eval_safe/closedloop_summary.json"),
        ("DAgger v1 + force_grip",  "runs/m3_bc_dagger_v1/closedloop_eval_safe_forcegrip/closedloop_summary.json"),
    ]
    names, grasped, final_grasped, returns = [], [], [], []
    for n, p in bars:
        d = read_summary(ROOT / p)
        s = policy_section(d)
        names.append(n)
        grasped.append((s.get("grasped_once_rate") or 0.0) * 100)
        final_grasped.append((s.get("final_grasped_rate") or 0.0) * 100)
        returns.append(s.get("mean_return") or 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(names))
    w = 0.38
    ax1.bar(x - w / 2, grasped, w, label="grasped_once (%)", color="#5b9bd5")
    ax1.bar(x + w / 2, final_grasped, w, label="final_grasped (%)", color="#ed7d31")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("rate (%)")
    ax1.set_title("M3 line · PickCube grasp metrics (30 ep, seed=3000)")
    ax1.legend(frameon=False)
    for i, v in enumerate(grasped):
        ax1.text(i - w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for i, v in enumerate(final_grasped):
        ax1.text(i + w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)

    ax2.bar(x, returns, color="#70ad47")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("mean_return")
    ax2.set_title("M3 line · mean_return progression")
    for i, v in enumerate(returns):
        ax2.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    out = OUT_CMP / "fig4_m3_progression.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# Figure 5: M8 four-way comparison — matched-instruction success
# -----------------------------------------------------------------------------
M8_RUNS = {
    "BC v2 (M6.2)":      "m6_vla_aux_v2",
    "Diffusion (M7.1)":  "m7_diffusion_v1",
    "PCGrad (M8a)":      "m8a_bc_pcgrad_v0",
    "PerTaskHead (M8b)": "m8b_bc_per_task_head_v0",
}
CELLS = {
    "PickCube / Pick (grasp)":   ("eval_pickcube_pick",   "grasped_once_rate"),
    "PushCube / Push (success)": ("eval_pushcube_push",   "success_rate_once"),
    "PullCube / Pull (success)": ("eval_pullcube_pull",   "success_rate_once"),
}

def fig_m8_bars():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    n_models = len(M8_RUNS)
    n_cells = len(CELLS)
    w = 0.8 / n_models
    x = np.arange(n_cells)
    colors = ["#5b9bd5", "#ed7d31", "#70ad47", "#a55a9c"]
    for j, (mname, mdir) in enumerate(M8_RUNS.items()):
        vals = []
        for cell, (eval_sub, key) in CELLS.items():
            p = RUNS / mdir / eval_sub / "closedloop_summary.json"
            if not p.exists():
                vals.append(0.0)
                continue
            d = read_summary(p)
            s = policy_section(d)
            vals.append((s.get(key) or 0.0) * 100)
        bars = ax.bar(x + (j - (n_models - 1) / 2) * w, vals, w, label=mname, color=colors[j])
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(CELLS.keys()))
    ax.set_ylabel("rate (%)")
    ax.set_ylim(0, max(60, ax.get_ylim()[1]))
    ax.set_title("M8 · 4-way matched-instruction comparison (30 ep / cell, seed=3000)")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    out = OUT_CMP / "fig5_m8_4way_matched.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# Figure 6: Swap matrix heatmap — M5.1 vs M6.2 (instruction following)
# -----------------------------------------------------------------------------
SWAP_GRID = {
    # row = env, col = instruction
    "PickCube": {
        "Pick (matched)": "eval_pickcube_pick",
        "Push (swap)":    "eval_pickcube_pushswap",
        "Pull (swap)":    "eval_pickcube_pullswap",
    },
    "PushCube": {
        "Push (matched)": "eval_pushcube_push",
        "Pick (swap)":    "eval_pushcube_pickswap",
        "Pull (swap)":    None,  # not collected
    },
    "PullCube": {
        "Pull (matched)": "eval_pullcube_pull",
        "Pick (swap)":    "eval_pullcube_pickswap",
        "Push (swap)":    None,
    },
}

def swap_grid(run_dir: Path, metric: str) -> np.ndarray:
    envs = list(SWAP_GRID.keys())
    cols = ["matched", "swap A", "swap B"]
    grid = np.full((len(envs), 3), np.nan)
    for i, env in enumerate(envs):
        for j, (label, sub) in enumerate(SWAP_GRID[env].items()):
            if sub is None:
                continue
            p = run_dir / sub / "closedloop_summary.json"
            if not p.exists():
                continue
            d = read_summary(p)
            s = policy_section(d)
            grid[i, j] = (s.get(metric) or 0.0) * 100
    return grid, envs, [list(SWAP_GRID[e].keys())[j] for e in envs for j in range(3)]


def fig_swap_heatmaps():
    runs = [
        ("M5.1 (state + text, no vision)", RUNS / "m5_1_bc_lang_multitask_aux_v0"),
        ("M6.2 BC v2 (state + text + vision)", RUNS / "m6_vla_aux_v2"),
        ("M8b Per-task heads", RUNS / "m8b_bc_per_task_head_v0"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, rd) in zip(axes, runs):
        envs = list(SWAP_GRID.keys())
        # For each env, build a 3-column row using per-cell labels (matched, swap1, swap2)
        col_labels_per_env = [list(SWAP_GRID[e].keys()) for e in envs]
        rows = []
        for i, env in enumerate(envs):
            row = []
            for label in col_labels_per_env[i]:
                sub = SWAP_GRID[env][label]
                if sub is None:
                    row.append(np.nan)
                    continue
                p = rd / sub / "closedloop_summary.json"
                if not p.exists():
                    row.append(np.nan)
                    continue
                d = read_summary(p)
                s = policy_section(d)
                # Pick grasped for PickCube, success for Push/Pull
                key = "grasped_once_rate" if env == "PickCube" else "success_rate_once"
                row.append((s.get(key) or 0.0) * 100)
            rows.append(row)
        mat = np.array(rows)
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=50, aspect="auto")
        ax.set_xticks(range(3))
        # Build column labels per env (different per row), so use generic header
        ax.set_xticklabels(["matched", "swap A", "swap B"])
        ax.set_yticks(range(len(envs)))
        ax.set_yticklabels(envs)
        ax.set_title(title, fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", color="#888", fontsize=10)
                else:
                    col = "white" if v > 25 else "black"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", color=col, fontsize=11)
        # annotate per-cell column meaning below tick
        per_env_labels = [list(SWAP_GRID[e].keys()) for e in envs]
        for i, labs in enumerate(per_env_labels):
            for j, lab in enumerate(labs):
                tag = lab.split(" (")[0]
                ax.text(j, i + 0.32, tag, ha="center", va="center", fontsize=7, color="#444")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("matched: success_once (Push/Pull) or grasped_once (Pick), %")
    fig.suptitle("Instruction-swap matrix · matched cell uses task-native success metric", y=1.02, fontsize=11)
    out = OUT_CMP / "fig6_swap_matrix_heatmaps.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# Figure 7: Capacity-sharing radar — best-at / worst-at pattern
# -----------------------------------------------------------------------------
def fig_radar():
    axes_labels = [
        "PickCube\ngrasp",
        "PushCube\nsuccess",
        "PullCube\nsuccess",
        "Swap\nobedience\n(1−swap-grasp)",
    ]
    # For each model, build the four metrics (in %).
    def get(rd, sub, key):
        p = RUNS / rd / sub / "closedloop_summary.json"
        if not p.exists():
            return 0.0
        d = read_summary(p)
        s = policy_section(d)
        return (s.get(key) or 0.0) * 100

    def obedience(rd):
        # higher = better. defined as 100 - (PickCube/Push-swap grasped + PullCube/Pick-swap success), capped to >=0
        a = get(rd, "eval_pickcube_pushswap", "grasped_once_rate")
        b = get(rd, "eval_pullcube_pickswap", "success_rate_once")
        return max(0.0, 100.0 - (a + b))

    models = {
        "BC v2 (M6.2)":      "m6_vla_aux_v2",
        "Diffusion (M7.1)":  "m7_diffusion_v1",
        "PCGrad (M8a)":      "m8a_bc_pcgrad_v0",
        "PerTaskHead (M8b)": "m8b_bc_per_task_head_v0",
    }

    angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    colors = ["#5b9bd5", "#ed7d31", "#70ad47", "#a55a9c"]
    for (name, rd), col in zip(models.items(), colors):
        vals = [
            get(rd, "eval_pickcube_pick", "grasped_once_rate"),
            get(rd, "eval_pushcube_push", "success_rate_once"),
            get(rd, "eval_pullcube_pull", "success_rate_once"),
            obedience(rd),
        ]
        vals += [vals[0]]
        ax.plot(angles, vals, label=name, color=col, linewidth=1.7)
        ax.fill(angles, vals, color=col, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=9)
    ax.set_rlim(0, 100)
    ax.set_rticks([20, 40, 60, 80])
    ax.set_rlabel_position(225)
    ax.tick_params(axis="y", labelsize=7, colors="#666")
    ax.set_title("Multi-task capacity-sharing · each model wins on a different axis", pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), frameon=False, fontsize=9)
    out = OUT_CMP / "fig7_capacity_radar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# Figure 8: M6.x evolution — PickCube grasp vs PushCube success vs PullCube success
# -----------------------------------------------------------------------------
def fig_m6_evolution():
    runs = [
        ("M5.1 (no vision)", "m5_1_bc_lang_multitask_aux_v0"),
        ("M6 v0 (+ vision)", "m6_vla_aux_v0"),
        ("M6.1 v1 (Push settle, lw=8)", "m6_vla_aux_v1"),
        ("M6.2 v2 (Pull settle)", "m6_vla_aux_v2"),
    ]
    cells = {
        "PickCube grasp":   ("eval_pickcube_pick", "grasped_once_rate"),
        "PushCube success": ("eval_pushcube_push", "success_rate_once"),
        "PullCube success": ("eval_pullcube_pull", "success_rate_once"),
    }
    n = len(runs)
    m = len(cells)
    x = np.arange(n)
    w = 0.27
    colors = ["#5b9bd5", "#ed7d31", "#70ad47"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for j, (label, (sub, key)) in enumerate(cells.items()):
        vals = []
        for _, rd in runs:
            p = RUNS / rd / sub / "closedloop_summary.json"
            if not p.exists():
                vals.append(0.0)
                continue
            d = read_summary(p)
            s = policy_section(d)
            vals.append((s.get(key) or 0.0) * 100)
        bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=colors[j])
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in runs], rotation=15, ha="right")
    ax.set_ylabel("rate (%)")
    ax.set_title("M5.1 → M6 → M6.1 → M6.2 · matched-instruction trade-off")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_CMP / "fig8_m6_evolution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    print("[1/8] Language conditioning val MSE")
    fig_lang_conditioning_curve()
    print("[2/8] M5.1 task accuracy")
    fig_task_acc()
    print("[3/8] Diffusion val eps-MSE")
    fig_diffusion_curve()
    print("[4/8] M3 progression bars")
    fig_m3_progression()
    print("[5/8] M8 4-way comparison")
    fig_m8_bars()
    print("[6/8] Swap matrix heatmaps")
    fig_swap_heatmaps()
    print("[7/8] Capacity radar")
    fig_radar()
    print("[8/8] M6.x evolution")
    fig_m6_evolution()
    print("done.")
