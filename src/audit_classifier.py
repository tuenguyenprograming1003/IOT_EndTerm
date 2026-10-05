"""Audit of the locked reference classifier (seed 0): is the low LOSO A0 a bug or speaker shift?

Checks
 1. Test predictions come from the Val-best checkpoint (reloaded checkpoint reproduces stored val_acc
    and the saved Test predictions).
 2. Test normalisation uses the Train mu/sigma of that fold (recomputed from Train only).
 3. label <-> filename consistency in splits and prediction CSV.
 4. classifier input shape is [B, 1, 64, 32].
 5. classifier is in eval mode at evaluation.
 6. Per-speaker accuracy, train accuracy (fit), and a diagnostic speaker-DEPENDENT split
    (random recordings, all speakers) — NOT part of the protocol, only to show the pipeline can learn.
Output: results/logs/classifier_audit.log, results/figures/confusion_original_per_speaker.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from classifier import DigitCNN, load_classifier, predict
from dataset import load_fold, load_logmel_cache
from features import fit_normalization, normalize
from utils import CKPT_DIR, FIG_DIR, LOG_DIR, MANIFEST_PATH, PRED_DIR, SPLITS_DIR, set_seed

lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


def main():
    preds = pd.read_csv(PRED_DIR / "classifier_original_predictions.csv")
    preds = preds[preds.seed == 0]
    S_all, index = load_logmel_cache()
    per_spk = []
    for fold in range(1, 7):
        ck_path = CKPT_DIR / "classifiers" / f"fold_{fold}_seed_0.pt"
        ck = torch.load(ck_path, weights_only=False)
        clf = load_classifier(ck_path)
        sp, mu, sigma = load_fold(fold)
        # 1. checkpoint reproduces stored Val accuracy and saved Test predictions
        val_acc = (predict(clf, sp["val"].x).numpy() == sp["val"].y).mean()
        test_pred = predict(clf, sp["test"].x).numpy()
        saved = preds[preds.fold == fold].set_index("filename").loc[sp["test"].meta.filename, "prediction"].to_numpy()
        # 2. normalisation recomputed from Train only
        tr = pd.read_csv(SPLITS_DIR / f"fold_{fold}_train.csv")
        mu2, sg2 = fit_normalization(S_all[[index[f] for f in tr.filename]])
        x_te2 = normalize(S_all[[index[f] for f in sp["test"].meta.filename]], mu2, sg2)
        # 3. labels vs filenames
        lab_ok = all(int(f.split("_")[0]) == y for f, y in zip(sp["test"].meta.filename, sp["test"].y))
        # 4./5. shape + eval mode
        x_t = torch.from_numpy(sp["test"].x).unsqueeze(1)
        train_acc = (predict(clf, sp["train"].x).numpy() == sp["train"].y).mean()
        test_acc = (test_pred == sp["test"].y).mean()
        out(f"fold {fold} ({sp['test'].meta.speaker.iloc[0]}): best_epoch={ck['best_epoch']} "
            f"val_acc stored={ck['val_acc']:.4f} reloaded={val_acc:.4f} | test preds == saved CSV: {np.array_equal(test_pred, saved)} | "
            f"mu/sigma == Train-recomputed: {np.allclose(mu, mu2) and np.allclose(sigma, sg2)} | "
            f"test x identical: {np.allclose(x_te2, sp['test'].x)} | labels==filename digit: {lab_ok} | "
            f"input shape {tuple(x_t.shape)} | eval mode: {not clf.training} | train acc {train_acc:.3f} test acc {test_acc:.3f}")
        per_spk.append(dict(fold=fold, test_speaker=sp["test"].meta.speaker.iloc[0], train_acc=train_acc,
                            val_acc=val_acc, test_acc=test_acc))
    csv_labels = preds.apply(lambda r: int(r.filename.split("_")[0]) == r.label, axis=1).all()
    out(f"prediction CSV label == filename digit for all rows: {csv_labels}")
    out("\nPer-speaker summary:\n" + pd.DataFrame(per_spk).round(3).to_string(index=False))

    # 6. diagnostic speaker-dependent split (random 70/15/15 of recordings across all speakers)
    man = pd.read_csv(MANIFEST_PATH)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(man))
    tr_i, va_i, te_i = perm[:2100], perm[2100:2550], perm[2550:]
    X = S_all[[index[f] for f in man.filename]]
    y = man.label.to_numpy()
    mu, sg = fit_normalization(X[tr_i])
    Xn = normalize(X, mu, sg)
    set_seed(0)
    m = DigitCNN()
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(Xn[tr_i]).unsqueeze(1), torch.from_numpy(y[tr_i])
    best = (0, None)
    for ep in range(40):
        m.train()
        p = torch.randperm(len(yt))
        for i in range(0, len(yt), 64):
            b = p[i:i + 64]
            loss = F.cross_entropy(m(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        va = (predict(m, Xn[va_i]).numpy() == y[va_i]).mean()
        if va > best[0]:
            best = (va, {k: v.clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    te = (predict(m, Xn[te_i]).numpy() == y[te_i]).mean()
    out(f"\nDiagnostic speaker-DEPENDENT random split (same features/classifier, 40 ep): "
        f"val {best[0]:.3f} test {te:.3f}  (not part of the protocol)")
    (LOG_DIR / "classifier_audit.log").write_text("\n".join(lines) + "\n")

    # per-speaker confusion matrices
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    import plotstyle

    plotstyle.apply()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for ax, (spk, g) in zip(axes.ravel(), preds.groupby("speaker")):
        cm = confusion_matrix(g.label, g.prediction, labels=range(10))
        ax.imshow(cm / 50, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 30 else "#0b0b0b")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set(title=f"{spk}: acc {g.correct.mean():.3f}", xlabel="Predicted", ylabel="True")
    fig.suptitle("Original classifier (seed 0) — confusion per held-out speaker")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_original_per_speaker.png")


if __name__ == "__main__":
    main()
