"""Evaluation utilities for trained chunking models."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    from src.data_loader import read_conll_file
    from src.features import extract_dataset_features
    from src.train import TARGET_COLUMNS, extract_target_labels, flatten_sentence_items
except ImportError:
    from data_loader import read_conll_file
    from features import extract_dataset_features
    from train import TARGET_COLUMNS, extract_target_labels, flatten_sentence_items


DEFAULT_DATA_PATH = Path("data") / "annotated" / "test.conll"
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_METRICS_DIR = Path("outputs") / "metrics"
DEFAULT_FIGURES_DIR = Path("outputs") / "figures"


def load_features_and_labels(data_path, target):
    """Load CoNLL data and return flat features and labels for one target."""
    sentences = read_conll_file(data_path)

    if not sentences:
        raise ValueError(
            f"No evaluation sentences found in {data_path}. "
            "Add annotated CoNLL data before evaluation."
        )

    features_by_sentence = extract_dataset_features(sentences)
    x = flatten_sentence_items(features_by_sentence)
    y_true = extract_target_labels(sentences, target)
    return x, y_true


def calculate_metrics(y_true, y_pred):
    """Calculate common token-level classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_classification_report(y_true, y_pred, target, output_dir):
    """Save a text classification report to the metrics directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_true, y_pred, zero_division=0)
    report_path = output_dir / f"logistic_regression_{target}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def save_metrics_summary(metrics, target, output_dir):
    """Save a compact metrics summary to the metrics directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"logistic_regression_{target}_metrics.txt"
    lines = [
        f"Target: {target}",
        "Model: Logistic Regression",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1-score: {metrics['f1']:.4f}",
    ]
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_path


def save_confusion_matrix(y_true, y_pred, target, output_dir):
    """Save a confusion matrix heatmap as a PNG file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    figure_width = max(6, len(labels) * 0.8)
    figure_height = max(5, len(labels) * 0.7)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_color = "white" if value > threshold else "black"
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    plt.title(f"Logistic Regression Confusion Matrix ({target})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    figure_path = output_dir / f"logistic_regression_{target}_confusion_matrix.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()
    return figure_path


def evaluate_model(
    model_path,
    data_path=DEFAULT_DATA_PATH,
    target="outer",
    metrics_dir=DEFAULT_METRICS_DIR,
    figures_dir=DEFAULT_FIGURES_DIR,
):
    """Evaluate a trained model and save metric and figure outputs."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    x, y_true = load_features_and_labels(data_path, target)
    y_pred = model.predict(x)

    metrics = calculate_metrics(y_true, y_pred)
    report_path = save_classification_report(y_true, y_pred, target, metrics_dir)
    metrics_path = save_metrics_summary(metrics, target, metrics_dir)
    figure_path = save_confusion_matrix(y_true, y_pred, target, figures_dir)

    return {
        "target": target,
        "model_path": model_path,
        "metrics": metrics,
        "classification_report_path": report_path,
        "metrics_path": metrics_path,
        "confusion_matrix_path": figure_path,
        "token_count": len(y_true),
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Logistic Regression chunking model."
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_COLUMNS),
        required=True,
        help="Target label to evaluate: outer, inner, or clause.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained joblib model.",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the CoNLL evaluation data.",
    )
    parser.add_argument(
        "--metrics-dir",
        default=DEFAULT_METRICS_DIR,
        help="Directory where metric files will be saved.",
    )
    parser.add_argument(
        "--figures-dir",
        default=DEFAULT_FIGURES_DIR,
        help="Directory where figure files will be saved.",
    )
    return parser.parse_args()


def main():
    """Run model evaluation from the command line."""
    args = parse_args()
    result = evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        target=args.target,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
    )

    print(f"Target: {result['target']}")
    print(f"Evaluated tokens: {result['token_count']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Precision: {result['metrics']['precision']:.4f}")
    print(f"Recall: {result['metrics']['recall']:.4f}")
    print(f"F1-score: {result['metrics']['f1']:.4f}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Classification report saved to: {result['classification_report_path']}")
    print(f"Confusion matrix saved to: {result['confusion_matrix_path']}")


if __name__ == "__main__":
    main()
