"""Main entry point for the Turkish multi-level phrase chunking project."""

import argparse
from pathlib import Path

from src.data_loader import count_tokens, read_conll_file
from src.error_analysis import run_error_analysis
from src.evaluate import evaluate_model
from src.predict import predict_sentence, save_predictions_as_conll
from src.train import MODEL_DISPLAY_NAMES, TARGET_COLUMNS, train_models


DEFAULT_TRAIN_DATA_PATH = Path("data") / "annotated" / "full_dataset.conll"
DEFAULT_EVAL_DATA_PATH = Path("data") / "annotated" / "test.conll"
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_METRICS_DIR = Path("outputs") / "metrics"
DEFAULT_FIGURES_DIR = Path("outputs") / "figures"
DEFAULT_PREDICTION_OUTPUT_PATH = (
    Path("outputs") / "predictions" / "predicted_sentence.conll"
)
DEFAULT_SENTENCE = "Dün akşam öğrenci makaleyi okudu."
DEFAULT_EVALUATION_MODEL = "logistic_regression"


def print_step(title):
    """Print a simple pipeline step title."""
    print(f"\n=== {title} ===")


def load_and_summarize_data(train_data_path, eval_data_path):
    """Load training and evaluation data and print a compact summary."""
    train_sentences = read_conll_file(train_data_path)
    eval_sentences = read_conll_file(eval_data_path)

    if not train_sentences:
        raise ValueError(
            f"No training sentences found in {train_data_path}. "
            "Add annotated CoNLL data before running the full pipeline."
        )

    if not eval_sentences:
        raise ValueError(
            f"No evaluation sentences found in {eval_data_path}. "
            "Add annotated CoNLL data before running the full pipeline."
        )

    print(
        f"Training data: {len(train_sentences)} sentences, "
        f"{count_tokens(train_sentences)} tokens"
    )
    print(
        f"Evaluation data: {len(eval_sentences)} sentences, "
        f"{count_tokens(eval_sentences)} tokens"
    )

    return train_sentences, eval_sentences


def run_training_stage(train_data_path, model_dir, metrics_dir, figures_dir):
    """Train all required models for all targets and save comparison outputs."""
    result = train_models(
        data_path=train_data_path,
        targets=list(TARGET_COLUMNS.keys()),
        model_names=list(MODEL_DISPLAY_NAMES.keys()),
        model_dir=model_dir,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
    )

    for row in result["results"]:
        print(
            f"{row['model_display_name']} | target={row['target']} | "
            f"accuracy={row['accuracy']:.4f} | f1={row['f1']:.4f}"
        )

    print(f"Model comparison CSV: {result['comparison_path']}")
    print(f"Model comparison figure: {result['figure_path']}")
    return result


def run_evaluation_stage(eval_data_path, model_dir, metrics_dir, figures_dir):
    """Evaluate the default model for each target."""
    evaluation_results = []

    for target in TARGET_COLUMNS:
        model_path = Path(model_dir) / f"{DEFAULT_EVALUATION_MODEL}_{target}.joblib"
        result = evaluate_model(
            model_path=model_path,
            data_path=eval_data_path,
            target=target,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
        )
        evaluation_results.append(result)
        print(
            f"{target}: accuracy={result['metrics']['accuracy']:.4f}, "
            f"f1={result['metrics']['f1']:.4f}"
        )
        print(f"  Classification report: {result['classification_report_path']}")
        print(f"  Confusion matrix: {result['confusion_matrix_path']}")

    return evaluation_results


def run_prediction_stage(sentence, prediction_output_path):
    """Run prediction for a sample sentence and save CoNLL output."""
    rows = predict_sentence(
        sentence_text=sentence,
        model_name=DEFAULT_EVALUATION_MODEL,
    )
    output_path = save_predictions_as_conll(rows, prediction_output_path)

    print(f"Sentence: {sentence}")
    print(f"Predicted tokens: {len(rows)}")
    print(f"Prediction output: {output_path}")
    return rows


def run_error_analysis_stage(eval_data_path, model_dir, metrics_dir):
    """Run error analysis for each target using the default model."""
    error_results = []

    for target in TARGET_COLUMNS:
        model_path = Path(model_dir) / f"{DEFAULT_EVALUATION_MODEL}_{target}.joblib"
        output_path = Path(metrics_dir) / f"error_analysis_{target}.csv"
        result = run_error_analysis(
            model_path=model_path,
            data_path=eval_data_path,
            target=target,
            output_path=output_path,
        )
        error_results.append(result)
        print(f"{target}: {result['error_count']} errors -> {result['output_path']}")

    return error_results


def run_full_pipeline(args):
    """Run the complete project pipeline with one command."""
    print_step("Data Loading")
    load_and_summarize_data(args.train_data_path, args.eval_data_path)

    print_step("Training and Model Comparison")
    run_training_stage(
        train_data_path=args.train_data_path,
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
    )

    print_step("Evaluation")
    run_evaluation_stage(
        eval_data_path=args.eval_data_path,
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
    )

    print_step("Prediction Output")
    run_prediction_stage(
        sentence=args.sentence,
        prediction_output_path=args.prediction_output_path,
    )

    print_step("Error Analysis")
    run_error_analysis_stage(
        eval_data_path=args.eval_data_path,
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
    )

    print("\nPipeline completed successfully.")


def parse_args():
    """Parse command-line arguments for the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full Turkish multi-level phrase chunking pipeline."
    )
    parser.add_argument(
        "--train-data-path",
        default=DEFAULT_TRAIN_DATA_PATH,
        help="Path to the CoNLL training dataset.",
    )
    parser.add_argument(
        "--eval-data-path",
        default=DEFAULT_EVAL_DATA_PATH,
        help="Path to the CoNLL evaluation dataset.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory where trained models will be saved.",
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
    parser.add_argument(
        "--prediction-output-path",
        default=DEFAULT_PREDICTION_OUTPUT_PATH,
        help="Path where the predicted CoNLL sentence will be saved.",
    )
    parser.add_argument(
        "--sentence",
        default=DEFAULT_SENTENCE,
        help="Turkish sentence used for prediction output.",
    )
    return parser.parse_args()


def main():
    """Run the command-line pipeline."""
    args = parse_args()
    run_full_pipeline(args)


if __name__ == "__main__":
    main()
