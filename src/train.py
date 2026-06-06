"""Training utilities for Logistic Regression chunking models."""

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

DEFAULT_DATA_PATH = Path("data") / "annotated" / "full_dataset.conll"
DEFAULT_MODEL_DIR = Path("models")


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


def build_logistic_regression_pipeline():
    """Build a simple DictVectorizer and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_logistic_regression(
    data_path=DEFAULT_DATA_PATH,
    target="outer",
    model_dir=DEFAULT_MODEL_DIR,
    test_size=0.2,
    random_state=42,
):
    """Train and save a Logistic Regression model for one target label."""
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

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    model = build_logistic_regression_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"logistic_regression_{target}.joblib"
    joblib.dump(model, model_path)

    return {
        "target": target,
        "model_path": model_path,
        "train_tokens": len(x_train),
        "test_tokens": len(x_test),
        "accuracy": accuracy,
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Logistic Regression model for Turkish chunking."
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_COLUMNS),
        required=True,
        help="Target label to train: outer, inner, or clause.",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the CoNLL dataset file.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory where the trained model will be saved.",
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
    """Run Logistic Regression training from the command line."""
    args = parse_args()
    result = train_logistic_regression(
        data_path=args.data_path,
        target=args.target,
        model_dir=args.model_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Target: {result['target']}")
    print(f"Model saved to: {result['model_path']}")
    print(f"Train tokens: {result['train_tokens']}")
    print(f"Test tokens: {result['test_tokens']}")
    print(f"Test accuracy: {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
