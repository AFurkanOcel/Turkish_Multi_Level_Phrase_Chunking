"""Training utilities for classical machine learning chunking models."""

import argparse
import csv
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

try:
    from src.data_loader import read_conll_file
    from src.features import extract_dataset_features
except ImportError:
    from data_loader import read_conll_file
    from features import extract_dataset_features


TARGET_COLUMNS = {
    "outer": "outer_chunk",
    "inner": "inner_chunk",
    "clause": "clause",
}

MODEL_DISPLAY_NAMES = {
    "majority_baseline": "Majority Class Baseline",
    "multinomial_naive_bayes": "Multinomial Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "mlp_classifier": "MLPClassifier",
}

DEFAULT_DATA_PATH = Path("data") / "annotated" / "train.conll"
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_METRICS_DIR = Path("outputs") / "metrics"
DEFAULT_FIGURES_DIR = Path("outputs") / "figures"


def flatten_sentence_items(sentence_items):
    """Flatten sentence-level items into a token-level list."""
    return [item for sentence in sentence_items for item in sentence]


def extract_target_labels(sentences, target):
    """Extract token-level labels for the selected target."""
    target_column = TARGET_COLUMNS[target]
    return [
        token[target_column]
        for sentence in sentences
        for token in sentence
    ]


def build_model_pipeline(model_name, random_state=42):
    """Build a scikit-learn pipeline for one classical model."""
    if model_name == "majority_baseline":
        classifier = DummyClassifier(strategy="most_frequent")
    elif model_name == "multinomial_naive_bayes":
        classifier = MultinomialNB()
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_name == "mlp_classifier":
        classifier = MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=500,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", classifier),
        ]
    )


def build_logistic_regression_pipeline():
    """Build a Logistic Regression pipeline for backward-compatible imports."""
    return build_model_pipeline("logistic_regression")


def calculate_metrics(y_true, y_pred):
    """Calculate weighted token-level classification metrics."""
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


def prepare_features_and_labels(data_path, target):
    """Load data and prepare flat token-level features and labels."""
    sentences = read_conll_file(data_path)

    if not sentences:
        raise ValueError(
            f"No training sentences found in {data_path}. "
            "Add annotated CoNLL data before training."
        )

    features_by_sentence = extract_dataset_features(sentences)
    x = flatten_sentence_items(features_by_sentence)
    y = extract_target_labels(sentences, target)

    if len(x) < 2:
        raise ValueError(
            "At least two tokens are required to create a train/test split."
        )

    return x, y


def train_single_model(
    x_train,
    x_test,
    y_train,
    y_test,
    target,
    model_name,
    model_dir,
    random_state=42,
):
    """Train, evaluate, and save one model for one target."""
    if len(set(y_train)) < 2 and model_name != "majority_baseline":
        raise ValueError(
            f"{MODEL_DISPLAY_NAMES[model_name]} requires at least two classes "
            f"in the training split for target {target}."
        )

    model = build_model_pipeline(model_name, random_state=random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = calculate_metrics(y_test, predictions)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}_{target}.joblib"
    joblib.dump(model, model_path)

    return {
        "model": model_name,
        "model_display_name": MODEL_DISPLAY_NAMES[model_name],
        "target": target,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "train_tokens": len(x_train),
        "test_tokens": len(x_test),
        "model_path": str(model_path),
    }


def train_models(
    data_path=DEFAULT_DATA_PATH,
    targets=None,
    model_names=None,
    model_dir=DEFAULT_MODEL_DIR,
    metrics_dir=DEFAULT_METRICS_DIR,
    figures_dir=DEFAULT_FIGURES_DIR,
    test_size=0.2,
    random_state=42,
):
    """Train selected models for selected targets and save comparison outputs."""
    if targets is None:
        targets = ["outer"]
    if model_names is None:
        model_names = ["logistic_regression"]

    results = []

    for target in targets:
        x, y = prepare_features_and_labels(data_path, target)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        for model_name in model_names:
            result = train_single_model(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                target=target,
                model_name=model_name,
                model_dir=model_dir,
                random_state=random_state,
            )
            results.append(result)

    comparison_path = save_model_comparison_csv(results, metrics_dir)
    figure_path = save_model_comparison_plot(results, figures_dir)
    label_distribution_path = save_label_distribution_plot(data_path, figures_dir)

    return {
        "results": results,
        "comparison_path": comparison_path,
        "figure_path": figure_path,
        "label_distribution_path": label_distribution_path,
    }


def train_logistic_regression(
    data_path=DEFAULT_DATA_PATH,
    target="outer",
    model_dir=DEFAULT_MODEL_DIR,
    test_size=0.2,
    random_state=42,
):
    """Train and save a Logistic Regression model for one target label."""
    result = train_models(
        data_path=data_path,
        targets=[target],
        model_names=["logistic_regression"],
        model_dir=model_dir,
        test_size=test_size,
        random_state=random_state,
    )
    return result["results"][0]


def save_model_comparison_csv(results, metrics_dir):
    """Save model comparison metrics as a CSV file."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = metrics_dir / "model_comparison.csv"

    fieldnames = [
        "model",
        "model_display_name",
        "target",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "train_tokens",
        "test_tokens",
        "model_path",
    ]

    with comparison_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    return comparison_path


def save_model_comparison_plot(results, figures_dir):
    """Save a grouped F1-score comparison chart as a PNG file."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "model_comparison.png"

    targets = list(TARGET_COLUMNS.keys())
    model_names = list(MODEL_DISPLAY_NAMES.keys())
    result_lookup = {
        (result["target"], result["model"]): result["f1"]
        for result in results
    }

    x_positions = range(len(targets))
    bar_width = 0.18

    plt.figure(figsize=(10, 6))
    for model_index, model_name in enumerate(model_names):
        offsets = [
            position + (model_index - 1.5) * bar_width
            for position in x_positions
        ]
        scores = [
            result_lookup.get((target, model_name), 0.0)
            for target in targets
        ]
        plt.bar(
            offsets,
            scores,
            width=bar_width,
            label=MODEL_DISPLAY_NAMES[model_name],
        )

    plt.title("Model Comparison by Target")
    plt.xlabel("Target")
    plt.ylabel("Weighted F1-score")
    plt.xticks(list(x_positions), targets)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    return figure_path


def save_label_distribution_plot(data_path, figures_dir):
    """Save label distribution charts for all project targets."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "label_distribution.png"

    sentences = read_conll_file(data_path)
    target_names = list(TARGET_COLUMNS.keys())

    fig, axes = plt.subplots(1, len(target_names), figsize=(15, 5))
    if len(target_names) == 1:
        axes = [axes]

    for axis, target in zip(axes, target_names):
        labels = extract_target_labels(sentences, target)
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        sorted_items = sorted(label_counts.items(), key=lambda item: item[0])
        x_labels = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        axis.bar(x_labels, counts)
        axis.set_title(f"{target} Label Distribution")
        axis.set_xlabel("Label")
        axis.set_ylabel("Count")
        axis.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    return figure_path


def expand_targets(target):
    """Expand a target argument into concrete target names."""
    if target == "all":
        return list(TARGET_COLUMNS.keys())
    return [target]


def expand_model_names(model_name):
    """Expand a model argument into concrete model names."""
    if model_name == "all":
        return list(MODEL_DISPLAY_NAMES.keys())
    return [model_name]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train classical machine learning models for Turkish chunking."
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_COLUMNS) + ["all"],
        required=True,
        help="Target label to train: outer, inner, clause, or all.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_DISPLAY_NAMES) + ["all"],
        default="logistic_regression",
        help="Model to train. Use all for model comparison.",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the CoNLL dataset file.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory where trained models will be saved.",
    )
    parser.add_argument(
        "--metrics-dir",
        default=DEFAULT_METRICS_DIR,
        help="Directory where model comparison metrics will be saved.",
    )
    parser.add_argument(
        "--figures-dir",
        default=DEFAULT_FIGURES_DIR,
        help="Directory where model comparison figures will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    return parser.parse_args()


def main():
    """Run model training and comparison from the command line."""
    args = parse_args()
    result = train_models(
        data_path=args.data_path,
        targets=expand_targets(args.target),
        model_names=expand_model_names(args.model),
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    for row in result["results"]:
        print(
            f"{row['model_display_name']} | target={row['target']} | "
            f"accuracy={row['accuracy']:.4f} | f1={row['f1']:.4f} | "
            f"model={row['model_path']}"
        )

    print(f"Model comparison CSV saved to: {result['comparison_path']}")
    print(f"Model comparison figure saved to: {result['figure_path']}")
    print(f"Label distribution figure saved to: {result['label_distribution_path']}")


if __name__ == "__main__":
    main()
