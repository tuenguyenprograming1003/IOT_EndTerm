"""Lock the V2 classifier configuration from the Val-only search (pre-specified rule:
highest mean Val accuracy over the 6 LOSO folds, seed 0; Test is not used).
Writes results/logs/followup_v2_locked_config.json and prints the train_classifier.py arguments."""
import json

import pandas as pd

from utils import LOG_DIR, SUMMARY_DIR

df = pd.read_csv(SUMMARY_DIR / "classifier_v2_search_val.csv")
complete = df.groupby("config").fold.nunique() == 6
df = df[df.config.isin(complete[complete].index)]
summ = df.groupby("config").agg(mean_val_acc=("val_acc", "mean"), std_val_acc=("val_acc", "std"),
                                 arch=("arch", "first"), dropout=("dropout", "first"), aug=("aug", "first"),
                                 epochs=("epochs", "first"), patience=("patience", "first"),
                                 lr=("lr", "first"), wd=("wd", "first")).sort_values("mean_val_acc", ascending=False)
best = summ.iloc[0]
args = f"--arch {best.arch} --dropout {best.dropout:g} --aug {best.aug} --epochs {int(best.epochs)} --lr {best.lr:g} --wd {best.wd:g}"
if pd.notna(best.patience):
    args += f" --patience {int(best.patience)}"
rec = {"rule": "max mean Val accuracy over 6 folds (seed 0); Test not used",
       "locked_config": best.name, "clf_args": args, "ranking": summ.reset_index().round(4).to_dict("records")}
(LOG_DIR / "followup_v2_locked_config.json").write_text(json.dumps(rec, indent=2, default=str))
print(summ.round(4).to_string())
print("LOCKED:", best.name)
print("CLF_ARGS=" + args)
