"""Error analysis utilities for incorrect token-level predictions."""

import argparse
import csv
from pathlib import Path

import joblib

try:
    from src.data_loader import read_conll_file
    from src.features import extract_sentence_features
    from src.train import TARGET_COLUMNS
except ImportError:
    from data_loader import read_conll_file
    from features import extract_sentence_features
    from train import TARGET_COLUMNS


DEFAULT_DATA_PATH = Path("data") / "annotated" / "test.conll"
DEFAULT_OUTPUT_PATH = Path("outputs") / "metrics" / "error_analysis.csv"
DEFAULT_MODEL_NAME = "logistic_regression"
DEFAULT_MODEL_DIR = Path("models")


def build_context(sentence, token_index, window_size=2):
    """Build a short token context around an error."""
    start_index = max(0, token_index - window_size)
    end_index = min(len(sentence), token_index + window_size + 1)
    context_tokens = []

    for index in range(start_index, end_index):
        token_text = sentence[index]["form"]
        if index == token_index:
            token_text = f"[{token_text}]"
        context_tokens.append(token_text)

    return " ".join(context_tokens)


def find_prediction_errors(sentences, model, target):
    """Find tokens where the gold label and predicted label are different."""
    target_column = TARGET_COLUMNS[target]
    errors = []

    for sentence_index, sentence in enumerate(sentences, start=1):
        sentence_features = extract_sentence_features(sentence)
        predictions = model.predict(sentence_features)

        for token_index, token in enumerate(sentence):
            gold_label = token[target_column]
            predicted_label = predictions[token_index]

            if gold_label != predicted_label:
                errors.append(
                    {
                        "sentence_id": sentence_index,
                        "token": token["form"],
                        "gold_label": gold_label,
                        "predicted_label": predicted_label,
                        "context": build_context(sentence, token_index),
                    }
                )

    return errors


def save_error_analysis(errors, output_path=DEFAULT_OUTPUT_PATH):
    """Save error analysis rows as a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sentence_id",
        "token",
        "gold_label",
        "predicted_label",
        "context",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    return output_path


def run_error_analysis(
    model_path,
    data_path=DEFAULT_DATA_PATH,
    target="outer",
    output_path=DEFAULT_OUTPUT_PATH,
):
    """Run error analysis for one trained model and target."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    sentences = read_conll_file(data_path)
    if not sentences:
        raise ValueError(
            f"No evaluation sentences found in {data_path}. "
            "Add annotated CoNLL data before error analysis."
        )

    model = joblib.load(model_path)
    errors = find_prediction_errors(sentences, model, target)
    saved_path = save_error_analysis(errors, output_path)

    return {
        "target": target,
        "model_path": model_path,
        "output_path": saved_path,
        "error_count": len(errors),
        "sentence_count": len(sentences),
    }


def default_model_path(model_name, target, model_dir=DEFAULT_MODEL_DIR):
    """Build the default model path for a model family and target."""
    return Path(model_dir) / f"{model_name}_{target}.joblib"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate token-level error analysis for chunking models."
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_COLUMNS),
        required=True,
        help="Target label to analyze: outer, inner, or clause.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to the trained joblib model.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model family name used when --model-path is not provided.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory containing trained joblib models.",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the CoNLL evaluation data.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the error analysis CSV will be saved.",
    )
    return parser.parse_args()


def main():
    """Run error analysis from the command line."""
    args = parse_args()
    model_path = args.model_path

    if model_path is None:
        model_path = default_model_path(args.model, args.target, args.model_dir)

    result = run_error_analysis(
        model_path=model_path,
        data_path=args.data_path,
        target=args.target,
        output_path=args.output_path,
    )

    print(f"Target: {result['target']}")
    print(f"Sentences analyzed: {result['sentence_count']}")
    print(f"Errors found: {result['error_count']}")
    print(f"Error analysis saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
