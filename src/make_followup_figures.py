"""Figures for the follow-up V2 experiment (re-run as more blocks finish; missing data is skipped).

fu_classifier_search_val.png      mean Val accuracy of every searched classifier config (Val only)
fu_a0_v1_vs_v2_by_speaker.png     Original-FP32 Test accuracy per speaker, V1 (main) vs V2 (seed 0)
fu_accuracy_vs_bytes_v2.png       Accuracy vs packet bytes on top of V2 (seed 0)
fu_task_gain_by_speaker_v2.png    Task-AE minus AE-MSE per speaker on top of V2 (seed 0)
fu_retention_v1_vs_v2.png         R_acc of codecs at each d, main vs follow-up (seed 0)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotstyle
from plotstyle import METHOD_COLORS, SERIES, TEXT2
from utils import FIG_DIR, SUMMARY_DIR

plotstyle.apply()
ORDER = ["Task-AE", "AE-MSE", "DCT", "Low-Mel"]


def search_fig():
    df = pd.read_csv(SUMMARY_DIR / "classifier_v2_search_val.csv")
    g = df.groupby("config").val_acc.agg(["mean", "std"]).sort_values("mean")
    import json
    from utils import LOG_DIR
    try:
        locked = json.load(open(LOG_DIR / "followup_v2_locked_config.json"))["locked_config"]
    except FileNotFoundError:
        locked = None
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = [SERIES[0] if c == locked else ("#52514e" if c.startswith("A1") else "#9fc3ee") for c in g.index]
    ax.barh(g.index, g["mean"], xerr=g["std"], color=colors, height=0.65, error_kw=dict(ecolor=TEXT2, lw=1))
    for y, v in enumerate(g["mean"]):
        ax.text(0.01, y, f"{v:.3f}", va="center", fontsize=8, color="white", fontweight="bold")
    ax.set(xlabel="Validation accuracy (mean ± std over 6 LOSO folds, seed 0)", xlim=(0, 1),
           title="Classifier V2 search (Validation only; dark blue = locked, grey = spec V1)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "fu_classifier_search_val.png")
    plt.close(fig)


def load():
    m = pd.read_csv(SUMMARY_DIR / "results_summary.csv")
    v = pd.read_csv(SUMMARY_DIR / "results_summary_v2.csv")
    return m[m.seed == 0], v[v.seed == 0], v


def a0_fig(m_all, v_all):
    """Bars = mean of seeds 0-2, dots = individual seeds (V2 is seed-sensitive)."""
    fig, ax = plt.subplots(figsize=(8.5, 4))
    w = 0.38
    spk = sorted(m_all.test_speaker.unique())
    for i, (df, col, lab) in enumerate([(m_all, "#52514e", "V1 (main, spec)"), (v_all, SERIES[0], "V2 (follow-up)")]):
        o = df[df.method == "Original"].pivot_table(index="test_speaker", columns="seed", values="Accuracy").reindex(spk)
        xs = np.arange(len(spk)) + (i - 0.5) * w
        ax.bar(xs, o.mean(1), w - 0.02, color=col, label=f"{lab}: mean of {o.shape[1]} seeds = {o.mean(1).mean():.3f}")
        for c in o.columns:
            ax.scatter(xs, o[c], s=14, color="#0b0b0b", zorder=3, edgecolor="white", linewidth=0.5)
    ax.scatter([], [], s=14, color="#0b0b0b", label="individual seeds")
    ax.axhline(0.1, color=TEXT2, ls=":", lw=1)
    ax.set_xticks(range(len(spk)), spk)
    ax.set(xlabel="Held-out test speaker", ylabel="Test accuracy (Original FP32)", ylim=(0, 1),
           title="Reference classifier A0: V1 vs V2 (seeds 0-2)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(FIG_DIR / "fu_a0_v1_vs_v2_by_speaker.png")
    plt.close(fig)


def bytes_fig(v0):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, m in enumerate(ORDER):
        g = v0[v0.method == m].groupby("bytes").Accuracy.agg(["mean", "std"])
        if g.empty:
            continue
        xs = g.index * (1 + 0.025 * (k - 1.5))
        ax.errorbar(xs, g["mean"], yerr=g["std"], marker="o", capsize=3, color=METHOD_COLORS[m], label=f"{m} (mean ± std)")
    for m, ls in [("Original", "--"), ("INT8", ":")]:
        val = v0[v0.method == m].Accuracy.mean()
        ax.axhline(val, color=METHOD_COLORS[m], ls=ls, lw=1.3, label=f"{m} ({int(v0[v0.method == m].bytes.iloc[0])} B)")
    ticks = sorted(v0[v0.method == "Task-AE"].bytes.unique())
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set(xlabel="Packet size (bytes, payload + 4-byte header)", ylabel="Test accuracy (mean of 6 LOSO folds)",
           title="Follow-up V2: accuracy vs transmitted bytes")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.14), fontsize=8)
    fig.savefig(FIG_DIR / "fu_accuracy_vs_bytes_v2.png")
    plt.close(fig)


def gain_fig(v0):
    p = v0[v0.method.isin(["Task-AE", "AE-MSE"])].pivot_table(index=["test_speaker", "d"], columns="method", values="Accuracy")
    gain = (100 * (p["Task-AE"] - p["AE-MSE"])).unstack("d")
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(gain.columns)
    for i, d in enumerate(gain.columns):
        ax.bar(np.arange(len(gain)) + (i - (len(gain.columns) - 1) / 2) * w, gain[d], width=w - 0.02, color=SERIES[i], label=f"d={d}")
    ax.axhline(0, color=TEXT2, lw=1)
    ax.set_xticks(range(len(gain)), gain.index)
    ax.set(xlabel="Test speaker (fold)", ylabel="Accuracy gain (pp)", title="Follow-up V2: Task-AE minus AE-MSE per held-out speaker")
    ax.legend()
    fig.savefig(FIG_DIR / "fu_task_gain_by_speaker_v2.png")
    plt.close(fig)


def retention_fig(m0, v0):
    rows = []
    for name, df in (("main (V1)", m0), ("follow-up (V2)", v0)):
        g = df[df.method.isin(["Task-AE", "AE-MSE", "DCT"])].groupby(["method", "d"]).R_acc.mean()
        for (meth, d), r in g.items():
            rows.append(dict(exp=name, method=meth, d=d, R_acc=r))
    t = pd.DataFrame(rows)
    ds = sorted(t.d.unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(4 * len(ds), 3.8), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], ds):
        sub = t[t.d == d].pivot(index="method", columns="exp", values="R_acc").reindex(["Task-AE", "AE-MSE", "DCT"])
        w = 0.38
        for i, (exp, col) in enumerate([("main (V1)", "#52514e"), ("follow-up (V2)", SERIES[0])]):
            if exp in sub:
                ax.bar(np.arange(len(sub)) + (i - 0.5) * w, sub[exp], w - 0.02, color=col, label=exp)
        ax.axhline(100, color=TEXT2, ls="--", lw=1)
        ax.set_xticks(range(len(sub)), sub.index)
        ax.set(title=f"d = {d} ({d + 4} B)", ylim=(60, 115))
    axes[0][0].set_ylabel("R_acc = 100·Acomp/A0 (%)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#52514e"), plt.Rectangle((0, 0), 1, 1, color=SERIES[0])],
               labels=["main (V1 classifier)", "follow-up (V2 classifier)"], loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Accuracy retention R_acc, seed 0 (missing bar = not yet run)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG_DIR / "fu_retention_v1_vs_v2.png")
    plt.close(fig)


def lambda_sel_fig():
    path = SUMMARY_DIR / "followup_v2_lambda_val_selection.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.38
    x = np.arange(len(d))
    ax.bar(x - w / 2, d.test_acc_lambda1, w - 0.02, color="#52514e", label=f"λ = 1 fixed (spec)  mean {d.test_acc_lambda1.mean():.3f}")
    ax.bar(x + w / 2, d.test_acc_selected, w - 0.02, color=SERIES[0], label=f"λ chosen on Val per fold  mean {d.test_acc_selected.mean():.3f}")
    for xi, lam in zip(x, d.selected_lambda):
        ax.text(xi + w / 2, 0.02, f"λ={lam:g}", ha="center", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x, d.test_speaker)
    ax.set(xlabel="Held-out test speaker", ylabel="Test accuracy", ylim=(0, 1),
           title="Follow-up V2, Task-AE d=128: per-fold λ selected on Validation")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(FIG_DIR / "fu_lambda_val_selection_v2.png")
    plt.close(fig)


def schedule_fig():
    path = SUMMARY_DIR / "followup_v2_schedule_ablation.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    g = d.groupby("schedule")[["val_acc", "test_acc", "val_mse", "test_mse"]].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (metric, lab) in zip(axes, [("acc", "accuracy"), ("mse", "MSE")]):
        for i, (split, col) in enumerate([("val", "#9fc3ee"), ("test", SERIES[0])]):
            m = g[f"{split}_{metric}"]
            ax.bar(np.arange(len(g)) + (i - 0.5) * 0.38, m["mean"], 0.36, yerr=m["std"], color=col,
                   label=split.capitalize(), error_kw=dict(ecolor=TEXT2, lw=1))
        ax.set_xticks(range(len(g)), g.index)
        ax.set(ylabel=lab, title=f"λ schedule vs {lab} (mean ± std, 6 folds)")
        ax.legend(fontsize=8, loc="lower center", ncol=2, framealpha=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fu_lambda_schedule_v2.png")
    plt.close(fig)


if __name__ == "__main__":
    search_fig()
    m0, v0, v = load()
    a0_fig(pd.read_csv(SUMMARY_DIR / "results_summary.csv"), v)
    bytes_fig(v0)
    gain_fig(v0)
    retention_fig(m0, v0)
    lambda_sel_fig()
    schedule_fig()
    print("follow-up figures written")
