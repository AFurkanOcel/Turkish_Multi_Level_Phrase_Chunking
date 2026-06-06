"""Command-line prediction utilities for new Turkish sentences."""

import argparse
import re
from pathlib import Path

import joblib

try:
    from src.features import extract_sentence_features
    from src.train import MODEL_DISPLAY_NAMES, TARGET_COLUMNS
except ImportError:
    from features import extract_sentence_features
    from train import MODEL_DISPLAY_NAMES, TARGET_COLUMNS


DEFAULT_MODEL_NAME = "logistic_regression"
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_OUTPUT_PATH = Path("outputs") / "predictions" / "predicted_sentence.conll"
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize_sentence(sentence_text):
    """Tokenize a Turkish sentence with a simple word-and-punctuation pattern."""
    return TOKEN_PATTERN.findall(sentence_text)


def build_unlabeled_sentence(tokens):
    """Create the sentence structure expected by the feature extractor."""
    return [
        {
            "id": token_index + 1,
            "form": token,
            "outer_chunk": "_",
            "inner_chunk": "_",
            "clause": "_",
        }
        for token_index, token in enumerate(tokens)
    ]


def load_target_models(model_name=DEFAULT_MODEL_NAME, model_dir=DEFAULT_MODEL_DIR):
    """Load one trained model for each target label."""
    model_dir = Path(model_dir)
    models = {}

    for target in TARGET_COLUMNS:
        model_path = model_dir / f"{model_name}_{target}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found for target {target}: {model_path}. "
                "Train the required models before prediction."
            )

        models[target] = joblib.load(model_path)

    return models


def predict_sentence(sentence_text, model_name=DEFAULT_MODEL_NAME, model_dir=DEFAULT_MODEL_DIR):
    """Predict outer, inner, and clause labels for a Turkish sentence."""
    tokens = tokenize_sentence(sentence_text)

    if not tokens:
        raise ValueError("The input sentence does not contain any tokens.")

    sentence = build_unlabeled_sentence(tokens)
    features = extract_sentence_features(sentence)
    models = load_target_models(model_name=model_name, model_dir=model_dir)

    predictions = {
        target: models[target].predict(features)
        for target in TARGET_COLUMNS
    }

    rows = []
    for token_index, token in enumerate(tokens):
        rows.append(
            {
                "id": token_index + 1,
                "form": token,
                "outer_chunk": predictions["outer"][token_index],
                "inner_chunk": predictions["inner"][token_index],
                "clause": predictions["clause"][token_index],
            }
        )

    return rows


def format_prediction_table(rows):
    """Format predictions as a compact terminal table."""
    headers = ["ID", "FORM", "OUTER_CHUNK", "INNER_CHUNK", "CLAUSE"]
    data_rows = [
        [
            str(row["id"]),
            row["form"],
            row["outer_chunk"],
            row["inner_chunk"],
            row["clause"],
        ]
        for row in rows
    ]
    table_rows = [headers] + data_rows
    column_widths = [
        max(len(table_row[column_index]) for table_row in table_rows)
        for column_index in range(len(headers))
    ]

    lines = []
    for row_index, table_row in enumerate(table_rows):
        line = "  ".join(
            value.ljust(column_widths[column_index])
            for column_index, value in enumerate(table_row)
        )
        lines.append(line)

        if row_index == 0:
            separator = "  ".join("-" * width for width in column_widths)
            lines.append(separator)

    return "\n".join(lines)


def save_predictions_as_conll(rows, output_path=DEFAULT_OUTPUT_PATH):
    """Save sentence predictions in the project CoNLL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# columns = ID FORM OUTER_CHUNK INNER_CHUNK CLAUSE"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["id"]),
                    row["form"],
                    row["outer_chunk"],
                    row["inner_chunk"],
                    row["clause"],
                ]
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict Turkish multi-level phrase chunk labels."
    )
    parser.add_argument(
        "--sentence",
        required=True,
        help="Turkish sentence to analyze.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_DISPLAY_NAMES),
        default=DEFAULT_MODEL_NAME,
        help="Trained model family to use for all targets.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory containing trained joblib models.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the predicted CoNLL output will be saved.",
    )
    return parser.parse_args()


def main():
    """Run prediction from the command line."""
    args = parse_args()
    rows = predict_sentence(
        sentence_text=args.sentence,
        model_name=args.model,
        model_dir=args.model_dir,
    )
    output_path = save_predictions_as_conll(rows, args.output_path)

    print(format_prediction_table(rows))
    print(f"\nPredicted CoNLL output saved to: {output_path}")


if __name__ == "__main__":
    main()
