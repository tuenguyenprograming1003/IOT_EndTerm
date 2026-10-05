"""Generate report/PHU_LUC_SO_LIEU.md — every number is computed from the project's CSV/log/checkpoint files.

Run from src/:  ../.venv/bin/python make_report_appendix.py
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import confusion_matrix, f1_score

from audio import load_fixed
from classifier import DigitCNN, build_classifier, load_classifier, predict
from codec import Codec
from dataset import load_fold, load_logmel_cache
from features import logmel_raw, mel_power
from utils import CKPT_DIR, LOG_DIR, MANIFEST_PATH, PRED_DIR, RESULTS_DIR, ROOT, SUMMARY_DIR

warnings.filterwarnings("ignore")
OUT = ROOT / "report" / "PHU_LUC_SO_LIEU.md"
HW = RESULTS_DIR / "hardware"
SPK = ["george", "jackson", "lucas", "nicolas", "theo", "yweweler"]
lines: list[str] = []


def w(s=""):
    lines.append(s)


def table(df: pd.DataFrame, fmt: dict | None = None, index=True, floatfmt="{:.3f}"):
    fmt = fmt or {}
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    w("| " + " | ".join(cols) + " |")
    w("|" + "---|" * len(cols))
    for r in d.itertuples(index=False):  # itertuples keeps per-column dtypes (iterrows upcasts ints)
        cells = []
        for c, v in zip(d.columns, r):
            f = fmt.get(c, floatfmt)
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                cells.append(fmt[c].format(v) if c in fmt and "," in fmt[c] else (f"{v:,}" if abs(v) >= 100000 else str(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("–" if np.isnan(v) else f.format(v))
            else:
                cells.append(str(v))
        w("| " + " | ".join(cells) + " |")
    w()


def pct(x):
    return 100 * x


# ---------------------------------------------------------------- A. data / features
def section_data():
    man = pd.read_csv(MANIFEST_PATH)
    w("## A. Dữ liệu và đặc trưng")
    w()
    w("### A1. Thống kê theo người nói")
    w()
    g = man.groupby("speaker")
    t = pd.DataFrame({
        "số file": g.size(),
        "thời lượng TB (s)": g.duration_sec.mean(), "std (s)": g.duration_sec.std(),
        "min (s)": g.duration_sec.min(), "median (s)": g.duration_sec.median(), "max (s)": g.duration_sec.max(),
        "PAD": g.num_samples.apply(lambda s: int((s < 8000).sum())),
        "CROP": g.num_samples.apply(lambda s: int((s > 8000).sum())),
    })
    table(t, {"số file": "{:.0f}", "PAD": "{:.0f}", "CROP": "{:.0f}"})
    w("### A2. Thống kê theo chữ số")
    w()
    g = man.groupby("label")
    t = pd.DataFrame({
        "số file": g.size(), "thời lượng TB (s)": g.duration_sec.mean(), "std (s)": g.duration_sec.std(),
        "min (s)": g.duration_sec.min(), "max (s)": g.duration_sec.max(),
        "PAD": g.num_samples.apply(lambda s: int((s < 8000).sum())),
        "CROP": g.num_samples.apply(lambda s: int((s > 8000).sum())),
    })
    t.index.name = "digit"
    table(t, {"số file": "{:.0f}", "PAD": "{:.0f}", "CROP": "{:.0f}"})
    n = man.num_samples
    w(f"Số mẫu/file: min {n.min()}, median {int(n.median())}, mean {n.mean():.0f}, max {n.max()}; "
      f"tổng 3000 file → PAD {int((n < 8000).sum())}, KEEP {int((n == 8000).sum())}, CROP {int((n > 8000).sum())}. "
      f"Tỷ lệ mẫu là zero-padding sau chuẩn hoá 1 s: {100 * (1 - np.minimum(n, 8000).sum() / (8000 * len(n))):.1f} % "
      f"tổng số mẫu.")
    w()

    w("### A3. Thang giá trị đặc trưng theo người nói (lý do bỏ `log1p`)")
    w()
    rows = []
    S_all, idx = load_logmel_cache()
    for spk, gg in man.groupby("speaker"):
        L1, LR, peaks = [], [], []
        for p in gg.path:
            y = load_fixed(ROOT / p)
            P = mel_power(y)
            L1.append(np.log1p(P))
            LR.append(np.log(P + 1e-6))
            peaks.append(np.abs(y).max())
        L1, LR = np.stack(L1), np.stack(LR)
        S = S_all[[idx[f] for f in gg.filename]]
        rows.append(dict(speaker=spk, **{"đỉnh |y| median": np.median(peaks), "log1p(P) TB": L1.mean(),
                                         "log1p(P) p99": np.percentile(L1, 99), "% log1p(P)<0.01": 100 * (L1 < 0.01).mean(),
                                         "log(P+1e-6) TB": LR.mean(), "log(P+1e-6) std": LR.std(),
                                         "CMVN S TB": S.mean(), "CMVN S std": S.std()}))
    t = pd.DataFrame(rows).set_index("speaker")
    table(t, {"log1p(P) TB": "{:.4f}", "log1p(P) p99": "{:.4f}", "% log1p(P)<0.01": "{:.1f}",
              "CMVN S TB": "{:.1e}", "đỉnh |y| median": "{:.3f}"})
    w("Với `log1p`, trung bình đặc trưng giữa speaker chênh tới hai bậc độ lớn; sau CMVN mọi speaker có "
      "trung bình 0 và độ lệch chuẩn 1.")
    w()

    w("### A4. Normalization theo Train từng fold (μ_f, σ_f trên 64 mel-bin)")
    w()
    rows = []
    for fold in range(1, 7):
        sp, mu, sigma = load_fold(fold)
        r = dict(fold=fold, test=sp["test"].meta.speaker.iloc[0], val=sp["val"].meta.speaker.iloc[0],
                 **{"μ min": mu.min(), "μ max": mu.max(), "σ min": sigma.min(), "σ max": sigma.max()})
        for s in ("train", "val", "test"):
            r[f"{s} mean"] = sp[s].x.mean()
            r[f"{s} std"] = sp[s].x.std()
        rows.append(r)
    table(pd.DataFrame(rows), index=False, fmt={"fold": "{}", "train mean": "{:.1e}"})
    w("Train sau chuẩn hoá có trung bình ≈ 0, std = 1 (theo định nghĩa); Val/Test dùng cùng μ, σ nên lệch nhẹ "
      "(std 0.92–1.10) — đây là lệch miền giữa người nói, không phải rò rỉ.")
    w()
    diag = (LOG_DIR / "diagnostics" / "feature_variant_diagnostic.md").read_text()
    w("### A5. Chẩn đoán biến thể đặc trưng (classifier V1, Test, chỉ để chọn đặc trưng ban đầu)")
    w()
    for ln in diag.splitlines():
        if ln.startswith("|"):
            w(ln)
    w()


# ---------------------------------------------------------------- complexity
def macs(model: torch.nn.Module, x: torch.Tensor) -> int:
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, torch.nn.Conv2d):
            total[0] += out.numel() // out.shape[0] * m.in_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups
        elif isinstance(m, torch.nn.ConvTranspose2d):
            i = inp[0]
            total[0] += i.numel() // i.shape[0] * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, torch.nn.Linear):
            total[0] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return total[0]


def section_complexity():
    w("## B. Độ phức tạp mô hình")
    w()
    x = torch.zeros(1, 1, 64, 32)
    rows = []
    for name, m in [("Classifier V1 (đề cương)", DigitCNN()),
                    ("Classifier V2 (t1, dropout 0.3)", build_classifier(dict(name="t1", dropout=0.3)))]:
        rows.append(dict(**{"mô hình": name, "tham số": sum(p.numel() for p in m.parameters()),
                            "MAC / mẫu": macs(m.eval(), x)}))
    for d in (64, 128, 256, 512):
        c = Codec(d).eval()
        z = c.encoder(x)
        rows.append({"mô hình": f"Encoder d={d} (r={c.r})", "tham số": sum(p.numel() for p in c.encoder.parameters()),
                     "MAC / mẫu": macs(c.encoder, x)})
        rows.append({"mô hình": f"Decoder d={d}", "tham số": sum(p.numel() for p in c.decoder.parameters()),
                     "MAC / mẫu": macs(c.decoder, z)})
    t = pd.DataFrame(rows)
    t["kích thước FP32 (KB)"] = t["tham số"] * 4 / 1000
    table(t, {"tham số": "{:,}", "MAC / mẫu": "{:,}", "kích thước FP32 (KB)": "{:.1f}"}, index=False)
    w("Byte và tỷ số nén theo byte thực (header 4 byte):")
    w()
    rows = [dict(**{"cấu hình": "Original FP32", "payload (B)": 8192, "packet (B)": 8196})]
    rows.append({"cấu hình": "Direct INT8", "payload (B)": 2048, "packet (B)": 2052})
    for d in (64, 128, 256, 512):
        rows.append({"cấu hình": f"d = {d}", "payload (B)": d, "packet (B)": d + 4})
    t = pd.DataFrame(rows)
    t["CR_FP32 = 8196/B"] = 8196 / t["packet (B)"]
    t["CR_INT8 = 2052/B"] = 2052 / t["packet (B)"]
    t["giảm số phần tử 2048/d (không phải CR)"] = [np.nan, np.nan] + [2048 / d for d in (64, 128, 256, 512)]
    t["bit / giá trị log-mel"] = t["payload (B)"] * 8 / 2048
    table(t, {"payload (B)": "{:.0f}", "packet (B)": "{:.0f}", "CR_FP32 = 8196/B": "{:.2f}",
              "CR_INT8 = 2052/B": "{:.2f}", "giảm số phần tử 2048/d (không phải CR)": "{:.0f}",
              "bit / giá trị log-mel": "{:.3f}"}, index=False)


# ---------------------------------------------------------------- classifiers
def clf_rows(tag: str):
    sub = "classifiers" if not tag else f"classifiers_{tag}"
    preds = pd.read_csv(PRED_DIR / f"classifier_original_predictions{'_' + tag if tag else ''}.csv")
    rows = []
    for seed in (0, 1, 2):
        for fold in range(1, 7):
            ck = CKPT_DIR / sub / f"fold_{fold}_seed_{seed}.pt"
            if not ck.exists():
                continue
            meta = torch.load(ck, weights_only=False)
            clf = load_classifier(ck)
            sp, _, _ = load_fold(fold)
            tr = (predict(clf, sp["train"].x).numpy() == sp["train"].y).mean()
            p = preds[(preds.fold == fold) & (preds.seed == seed)]
            rows.append(dict(seed=seed, fold=fold, test=sp["test"].meta.speaker.iloc[0],
                             **{"best epoch": meta["best_epoch"], "train acc": tr, "val acc": meta["val_acc"],
                                "test acc": p.correct.mean(),
                                "test Macro-F1": f1_score(p.label, p.prediction, average="macro"),
                                "train−test (pp)": 100 * (tr - p.correct.mean())}))
    return pd.DataFrame(rows), preds


def per_digit(preds: pd.DataFrame, label: str) -> pd.DataFrame:
    rec = preds.groupby("label").correct.mean()
    f1 = f1_score(preds.label, preds.prediction, average=None, labels=range(10))
    return pd.DataFrame({f"{label} recall": rec.values, f"{label} F1": f1}, index=range(10))


def top_confusions(preds: pd.DataFrame, k=8) -> str:
    cm = confusion_matrix(preds.label, preds.prediction, labels=range(10))
    np.fill_diagonal(cm, 0)
    order = np.dstack(np.unravel_index(np.argsort(-cm.ravel()), cm.shape))[0][:k]
    tot = preds.groupby("label").size()
    return "; ".join(f"{a}→{b}: {cm[a, b]} ({100 * cm[a, b] / tot[a]:.0f} %)" for a, b in order)


def section_classifiers():
    w("## C. Classifier tham chiếu")
    w()
    for tag, title in (("", "V1 (thí nghiệm chính)"), ("v2", "V2 (follow-up)")):
        df, preds = clf_rows(tag)
        w(f"### C{1 if not tag else 2}. {title}: từng fold × seed")
        w()
        table(df, {"seed": "{}", "fold": "{}", "best epoch": "{}", "train acc": "{:.3f}"}, index=False)
        s = df.groupby("seed")[["train acc", "val acc", "test acc", "test Macro-F1"]].agg(["mean", "std"])
        s.columns = [f"{a} {b}" for a, b in s.columns]
        table(s, {"seed": "{}"})
        allm = df[["train acc", "val acc", "test acc", "test Macro-F1"]].agg(["mean", "std"]).T
        w(f"Trung bình 18 run: train {allm.loc['train acc', 'mean']:.3f}, val {allm.loc['val acc', 'mean']:.3f}, "
          f"test {allm.loc['test acc', 'mean']:.3f} ± {allm.loc['test acc', 'std']:.3f}, "
          f"Macro-F1 {allm.loc['test Macro-F1', 'mean']:.3f}. Khoảng cách train − test trung bình "
          f"{df['train−test (pp)'].mean():.1f} pp.")
        w()
        sp = df.groupby("test")["test acc"].agg(["mean", "std", "min", "max"])
        sp.index.name = "test speaker"
        w("Theo test speaker (qua 3 seed):")
        w()
        table(sp)
    v1 = pd.read_csv(PRED_DIR / "classifier_original_predictions.csv")
    v2 = pd.read_csv(PRED_DIR / "classifier_original_predictions_v2.csv")
    w("### C3. Theo chữ số (Test, gộp 6 fold × 3 seed = 9000 dự đoán mỗi classifier)")
    w()
    t = per_digit(v1, "V1").join(per_digit(v2, "V2"))
    t.index.name = "digit"
    table(t)
    w(f"Cặp nhầm nhiều nhất V1: {top_confusions(v1)}.")
    w()
    w(f"Cặp nhầm nhiều nhất V2: {top_confusions(v2)}.")
    w()
    w("### C4. Tìm cấu hình V2 (Validation, seed 0) — đầy đủ 14 cấu hình")
    w()
    s = pd.read_csv(SUMMARY_DIR / "classifier_v2_search_val.csv")
    piv = s.pivot_table(index="config", columns="fold", values="val_acc")
    piv.columns = [f"f{c}" for c in piv.columns]
    piv["mean"] = piv.mean(1)
    piv["std"] = piv.iloc[:, :6].std(1)
    be = s.groupby("config").best_epoch.apply(lambda v: "/".join(map(str, v)))
    piv["best epochs f1..f6"] = be
    meta = s.groupby("config")[["arch", "dropout", "aug", "epochs", "lr", "wd"]].first()
    piv = meta.join(piv).sort_values("mean", ascending=False)
    table(piv, {"dropout": "{:.1f}", "epochs": "{:.0f}", "lr": "{:g}", "wd": "{:g}"})


# ---------------------------------------------------------------- codecs
def speaker_tables(summary: pd.DataFrame, seed=0, label=""):
    s0 = summary[summary.seed == seed]
    for metric, f in (("Accuracy", 100), ("Macro_F1", 1), ("MSE", 1), ("PRD_spec", 1)):
        p = s0.pivot_table(index=["method", "d"], columns="test_speaker", values=metric) * f
        p = p.reindex(columns=SPK)
        p["mean"] = p.mean(1)
        p["std"] = p[SPK].std(1)
        p["min"] = p[SPK].min(1)
        p["max"] = p[SPK].max(1)
        unit = " (%)" if metric == "Accuracy" else ""
        w(f"**{metric}{unit} theo test speaker {label}, seed {seed}:**")
        w()
        table(p, {"d": "{}"}, floatfmt="{:.1f}" if metric in ("Accuracy", "PRD_spec") else "{:.3f}")


def paired_tests(summary: pd.DataFrame, label: str):
    rows = []
    for d in sorted(summary[summary.method == "Task-AE"].d.unique()):
        for scope, sub in (("seed 0", summary[summary.seed == 0]), ("trung bình 3 seed", summary)):
            m = sub[(sub.d == d) & sub.method.isin(["Task-AE", "AE-MSE"])]
            if scope != "seed 0" and m.seed.nunique() < 3:
                continue
            spk = m.groupby(["test_speaker", "method"]).Accuracy.mean().unstack()
            g = 100 * (spk["Task-AE"] - spk["AE-MSE"])
            wp = wilcoxon(spk["Task-AE"], spk["AE-MSE"]).pvalue if (g != 0).any() else np.nan
            tp = ttest_rel(spk["Task-AE"], spk["AE-MSE"]).pvalue
            rows.append(dict(d=d, **{"phạm vi": scope, "gain TB (pp)": g.mean(), "std (pp)": g.std(),
                                     "min (pp)": g.min(), "max (pp)": g.max(), "speaker dương": f"{(g > 0).sum()}/6",
                                     "Wilcoxon p": wp, "paired t p": tp}))
    w(f"**Task-AE − AE-MSE, đơn vị = speaker (n = 6), {label}:**")
    w()
    table(pd.DataFrame(rows), {"d": "{}", "Wilcoxon p": "{:.4f}", "paired t p": "{:.4f}"}, index=False, floatfmt="{:.2f}")
    s0 = summary[summary.seed == 0]
    p = s0[s0.method.isin(["Task-AE", "AE-MSE"])].pivot_table(index=["fold", "d"], columns="method",
                                                               values=["MSE", "Accuracy"])
    hm = p["MSE"]["Task-AE"] > p["MSE"]["AE-MSE"]
    ha = p["Accuracy"]["Task-AE"] > p["Accuracy"]["AE-MSE"]
    w(f"Cặp (fold, d) seed 0: Task-AE MSE cao hơn {hm.sum()}/{len(hm)}; accuracy cao hơn {ha.sum()}/{len(ha)}; "
      f"vừa MSE cao hơn vừa accuracy cao hơn {(hm & ha).sum()}/{len(hm)}.")
    w()
    corr = s0[s0.method.isin(["Task-AE", "AE-MSE", "DCT"])][["MSE", "Accuracy"]].corr(method="spearman").iloc[0, 1]
    w(f"Tương quan Spearman giữa MSE và Accuracy trên mọi (method ∈ {{Task-AE, AE-MSE, DCT}}, d, fold) seed 0: "
      f"ρ = {corr:.3f}.")
    w()


def multiseed(summary: pd.DataFrame, label: str):
    m = summary[(summary.d.isin([128, 2048])) & summary.method.isin(["Original", "INT8", "Task-AE", "AE-MSE", "DCT"])]
    p = m.pivot_table(index=["seed", "test_speaker"], columns="method", values="Accuracy") * 100
    p = p[["Original", "INT8", "DCT", "AE-MSE", "Task-AE"]]
    p["Task−AE-MSE"] = p["Task-AE"] - p["AE-MSE"]
    w(f"**Accuracy (%) từng seed × speaker, d = 128, {label}:**")
    w()
    table(p, {"seed": "{}"}, floatfmt="{:.1f}")
    t = m.groupby(["method", "seed"]).Accuracy.mean().unstack() * 100
    t["TB 3 seed"] = t.mean(1)
    t["std giữa seed"] = t.iloc[:, :3].std(1)
    w(f"**Trung bình 6 fold theo seed (%), {label}:**")
    w()
    table(t, floatfmt="{:.2f}")


def codec_training(tag: str, label: str):
    rows = []
    for p in sorted((LOG_DIR / "codecs").glob("*.json")):
        name = p.stem
        is_v2 = name.endswith("_v2") or "_v2_" in name
        if (tag == "v2") != is_v2:
            continue
        j = json.loads(p.read_text())
        rows.append(dict(run=name, method=j["method"], d=j["d"], fold=j["fold"], seed=j["seed"], lam=j.get("lam_max"),
                         sched="slow" if name.endswith("_slow") else "main",
                         best_epoch=j["best_epoch"], val_mse=j["val_mse"], val_acc=j["val_acc"],
                         rec_first=j["first_train_rec"], rec_last=j["last_train_rec"], sec=j["seconds"],
                         enc_grad=(j.get("gradient_check") or {}).get("encoder_grad_norm"),
                         clf_grad=(j.get("gradient_check") or {}).get("classifier_params_received_grad")))
    df = pd.DataFrame(rows)
    main = df[(df.sched == "main") & ((df.method == "AE-MSE") | (df.lam == 1.0))]
    g = main.groupby(["method", "d"]).agg(runs=("run", "size"), best_epoch=("best_epoch", "mean"),
                                          val_mse=("val_mse", "mean"), val_acc=("val_acc", "mean"),
                                          train_rec_ep1=("rec_first", "mean"), train_rec_last=("rec_last", "mean"),
                                          sec_per_run=("sec", "mean"))
    w(f"**Huấn luyện codec ({label}), trung bình theo method × d (mọi seed có sẵn):**")
    w()
    table(g, {"d": "{}", "runs": "{:.0f}", "best_epoch": "{:.1f}", "sec_per_run": "{:.0f}"})
    gc = df[df.enc_grad.notna()]
    w(f"Kiểm tra gradient (chỉ lan truyền CE qua classifier đóng băng) trên {len(gc)} run Task-AE: chuẩn gradient "
      f"tại encoder {gc.enc_grad.min():.3f} – {gc.enc_grad.max():.3f} (luôn > 0); classifier nhận gradient: "
      f"{int(gc.clf_grad.sum())}/{len(gc)} run. Tổng thời gian train codec ({label}): {df.sec.sum() / 3600:.1f} giờ CPU-worker "
      f"({len(df)} run).")
    w()


def section_codecs():
    s = pd.read_csv(SUMMARY_DIR / "results_summary.csv")
    v = pd.read_csv(SUMMARY_DIR / "results_summary_v2.csv")
    w("## D. Codec — thí nghiệm chính (classifier V1)")
    w()
    speaker_tables(s, 0, "(main)")
    paired_tests(s, "main")
    multiseed(s, "main")
    codec_training("", "main")
    w("**Ablation λ (main, d = 128, seed 0) theo fold:**")
    w()
    va = pd.read_csv(SUMMARY_DIR / "lambda_ablation_val.csv")
    te = pd.read_csv(SUMMARY_DIR / "lambda_ablation_test_posthoc.csv")
    pv = va.pivot_table(index="fold", columns="lambda_max", values="Val_Accuracy") * 100
    pv.columns = [f"Val λ={c:g}" for c in pv.columns]
    pt = te.pivot_table(index="fold", columns="lambda_max", values="Test_Accuracy", aggfunc="first") * 100
    pt.columns = ["Test AE-MSE" if c == "AE-MSE" else f"Test λ={float(c):g}" for c in pt.columns]
    t = pv.join(pt)
    t.loc["mean"] = t.mean()
    table(t, {"fold": "{}"}, floatfmt="{:.1f}")
    pm = va.groupby("lambda_max")[["Val_MSE", "Val_PRD_spec", "Val_Macro_F1"]].mean()
    pm = pm.join(te[te.lambda_max != "AE-MSE"].assign(lambda_max=lambda d: d.lambda_max.astype(float))
                 .groupby("lambda_max").Test_MSE.mean())
    table(pm, {"lambda_max": "{:g}"})

    w("## E. Codec — follow-up (classifier V2)")
    w()
    speaker_tables(v, 0, "(V2)")
    paired_tests(v, "V2")
    multiseed(v, "V2")
    codec_training("v2", "V2")
    sel = pd.read_csv(SUMMARY_DIR / "followup_v2_lambda_val_selection.csv")
    w("**Chọn λ theo Val (V2, d = 128, seed 0) — Val accuracy của từng λ:**")
    w()
    vv = pd.DataFrame([{float(k): v for k, v in eval(x).items()} for x in sel.val_acc_by_lambda]) * 100
    vv.columns = [f"Val λ={c:g}" for c in vv.columns]
    vv.insert(0, "speaker", sel.test_speaker)
    vv["λ chọn"] = sel.selected_lambda
    vv["Test λ chọn"] = sel.test_acc_selected * 100
    vv["Test λ=1"] = sel.test_acc_lambda1 * 100
    vv["chênh Val max−min (pp)"] = vv.filter(like="Val λ").max(1) - vv.filter(like="Val λ").min(1)
    table(vv, {"λ chọn": "{:g}"}, index=False, floatfmt="{:.1f}")
    sch = pd.read_csv(SUMMARY_DIR / "followup_v2_schedule_ablation.csv")
    p = sch.pivot_table(index="fold", columns="schedule", values=["val_acc", "test_acc", "test_mse"])
    p.columns = [f"{a} {b}" for a, b in p.columns]
    p.loc["mean"] = p.mean()
    w("**Lịch λ (V2, d = 128, seed 0) theo fold:**")
    w()
    table(p, {"fold": "{}"})


# ---------------------------------------------------------------- hardware
def section_hardware():
    w("## F. ESP32-S3 (đo trên board thật)")
    w()
    b = pd.read_csv(HW / "hardware_benchmark.csv")
    rows = []
    for d, g in b.groupby("d"):
        for c in ("preprocess_ms", "encoder_ms", "quantization_ms", "packetization_ms", "total_node_ms", "end_to_end_node_ms"):
            s = g[c]
            rows.append(dict(d=d, stage=c, n=len(s), mean=s.mean(), median=s.median(), std=s.std(), min=s.min(),
                             p95=s.quantile(0.95), max=s.max()))
    w("**Latency (ms), 1000 lần đo sau 50 warm-up, batch = 1, 240 MHz:**")
    w()
    table(pd.DataFrame(rows), {"d": "{}", "n": "{}", "std": "{:.4f}"}, index=False, floatfmt="{:.3f}")
    rows = []
    for d, g in b.groupby("d"):
        enc = g.encoder_ms.median() / 1000
        m = macs(Codec(d).encoder, torch.zeros(1, 1, 64, 32))
        rows.append(dict(d=d, **{"MAC encoder": m, "MMAC/s": m / enc / 1e6, "cycle / MAC (240 MHz)": 240e6 * enc / m,
                                 "% thời gian node cho encoder": 100 * g.encoder_ms.median() / g.end_to_end_node_ms.median(),
                                 "% cho tiền xử lý": 100 * g.preprocess_ms.median() / g.end_to_end_node_ms.median(),
                                 "mẫu / giây (end-to-end)": 1000 / g.end_to_end_node_ms.median(),
                                 "byte/s phát ra ở 1 mẫu/s": d + 4}))
    w("**Thông lượng suy ra:**")
    w()
    table(pd.DataFrame(rows), {"d": "{}", "MAC encoder": "{:,}", "byte/s phát ra ở 1 mẫu/s": "{}"}, index=False, floatfmt="{:.2f}")
    w("**Bộ nhớ theo giai đoạn (log serial, byte):**")
    w()
    rows = []
    for log in sorted((HW / "serial_logs").glob("d*_2026*.log")):
        if "qemu" in log.name:
            continue
        d = int(re.match(r"d(\d+)_", log.name)[1])
        for ln in log.read_text(errors="replace").splitlines():
            if ln.startswith("[MEM] stage="):
                kv = dict(re.findall(r"(\w+)=(\S+)", ln[6:]))
                rows.append(dict(d=d, log=log.stem, **kv))
    mem = pd.DataFrame(rows).drop_duplicates(subset=["d", "stage"], keep="last").sort_values("d", kind="stable")
    for c in mem.columns:
        if c not in ("stage", "log"):
            mem[c] = mem[c].astype(int)
    table(mem.drop(columns=["log"]), index=False)
    hc = pd.read_csv(HW / "host_c_consistency.csv")
    w("**Kiểm tra trước khi nạp (host build cùng mã C, 500 recording Test fold 1):**")
    w()
    table(hc.set_index("d").T.reset_index().rename(columns={"index": "chỉ số"}), index=False, floatfmt="{:.6g}")
    cons = pd.read_csv(HW / "hardware_consistency.csv")
    w("**Round-trip qua board (500 recording × chế độ × d):**")
    w()
    table(cons, {"d": "{}", "n": "{}", "max_abs_q_diff": "{}", "packet_bytes": "{}"}, index=False, floatfmt="{:.6f}")
    p = pd.read_csv(HW / "hardware_predictions.csv")
    p["ok"] = (p.label == p.prediction_esp32_packet).astype(float)
    acc = p.groupby(["d", "mode", "label"]).ok.mean().unstack("label") * 100
    w("**Accuracy (%) theo chữ số từ packet ESP32 (fold 1 = george):**")
    w()
    table(acc, {"d": "{}"}, floatfmt="{:.0f}")
    rt = json.loads((LOG_DIR / "packet_roundtrip_Task-AE_d128_fold1_seed0.json").read_text())
    w(f"Round-trip file nhị phân hai tiến trình (PC): {rt['n_packets']} packet, kích thước {rt['packet_bytes_min']}–"
      f"{rt['packet_bytes_max']} B (kỳ vọng {rt['expected_bytes']}), accuracy {rt['accuracy_from_packets']:.3f}, "
      f"prediction trùng tham chiếu {100 * rt['prediction_agreement_with_reference']:.0f} %.")
    w()


def main():
    w("# Phụ lục số liệu chi tiết — Đề tài C3")
    w()
    w("*Sinh tự động bởi `src/make_report_appendix.py` từ các file trong `results/`, `data/` và `checkpoints/`. "
      "Không chỉnh tay. Accuracy là tỷ lệ (0–1) trừ khi ghi %. \"main\" = thí nghiệm chính (classifier V1, đã khoá); "
      "\"V2\" = follow-up hậu kiểm.*")
    w()
    section_data()
    section_complexity()
    section_classifiers()
    section_codecs()
    section_hardware()
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT, len(lines), "lines")


if __name__ == "__main__":
    main()
