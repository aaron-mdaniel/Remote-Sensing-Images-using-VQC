
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

comparison_file = RESULTS_DIR / "test_model_comparison.csv"
predictions_file = RESULTS_DIR / "test_predictions.csv"

if not comparison_file.exists() or not predictions_file.exists():
    raise FileNotFoundError(
        "Run evaluate_models.py first to generate the test results."
    )

comparison = pd.read_csv(comparison_file, index_col=0)
predictions = pd.read_csv(predictions_file)

# --------------------------------------------------
# 2. Graph 1: Performance comparison
# --------------------------------------------------

metrics = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]

labels = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1-score"
]

models = comparison.index.tolist()

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

for i, model_name in enumerate(models):
    values = comparison.loc[model_name, metrics].to_numpy(dtype=float)

    bars = ax.bar(
        x + (i - (len(models) - 1) / 2) * width,
        values,
        width,
        label=model_name
    )

    ax.bar_label(
        bars,
        labels=[f"{v:.2%}" for v in values],
        padding=3,
        fontsize=8
    )

ax.set_title("Test Performance: Hybrid Quantum-Classical vs SVM")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.12)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

performance_path = RESULTS_DIR / "performance_comparison.png"
fig.savefig(performance_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# --------------------------------------------------
# 3. Graph 2: Hybrid model confusion matrix
# --------------------------------------------------

class_names = ["Forest", "Highway"]

def plot_confusion_matrix(actual, predicted, title, output_file):
    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Number of images")

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=title
    )

    threshold = cm.max() / 2 if cm.size else 0

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
                fontsize=12
            )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


plot_confusion_matrix(
    predictions["actual_label"],
    predictions["hybrid_prediction"],
    "Hybrid Quantum-Classical: Test Confusion Matrix",
    RESULTS_DIR / "hybrid_confusion_matrix.png"
)

# --------------------------------------------------
# 4. Graph 3: SVM confusion matrix
# --------------------------------------------------

plot_confusion_matrix(
    predictions["actual_label"],
    predictions["svm_prediction"],
    "SVM: Test Confusion Matrix",
    RESULTS_DIR / "svm_confusion_matrix.png"
)

print("\nAll comparison graphs generated successfully.")
print("Saved in:", RESULTS_DIR)
