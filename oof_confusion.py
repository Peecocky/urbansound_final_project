import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

NUM_CLASSES = 10

# =========================
# 配置
# =========================
BASE_DIR = Path("outputs/logits/val")
MODELS = ["crnn", "resnet18", "resnet34", "resnet50"]
SEED = 350234
FOLDS = [1,2,3,4,5,6,7,8]

OUT_DIR = Path("analysis/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_oof_logits(model_name):
    """
    合并 8 个 fold 的 OOF logits
    """
    all_logits = []
    all_labels = []

    for f in FOLDS:
        p = BASE_DIR / model_name / f"seed{SEED}" / f"fold{f}.npz"
        assert p.exists(), f"Missing {p}"

        data = np.load(p)
        logits = data["logits"]
        labels = data["labels"]

        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    preds = all_logits.argmax(axis=1)
    return all_labels, preds


def analyze_model(model_name):
    y_true, y_pred = load_oof_logits(model_name)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)]
    )

    cm_df.to_csv(OUT_DIR / f"{model_name}_confusion_matrix.csv")

    # classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(OUT_DIR / f"{model_name}_class_report.csv")

    print(f"\n===== {model_name.upper()} =====")
    print(report_df[["precision", "recall", "f1-score"]].iloc[:NUM_CLASSES])

    return report_df


def main():
    summaries = []

    for model in MODELS:
        rep = analyze_model(model)
        macro_f1 = rep.loc["macro avg", "f1-score"]
        acc = rep.loc["accuracy", "precision"]
        summaries.append({
            "model": model,
            "accuracy": acc,
            "macro_f1": macro_f1
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "model_summary.csv", index=False)

    print("\n=== Model Summary ===")
    print(summary_df)


if __name__ == "__main__":
    main()
