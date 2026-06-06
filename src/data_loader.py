"""Data loading utilities for CoNLL-style Turkish chunking data.

Expected columns:
ID FORM OUTER_CHUNK INNER_CHUNK CLAUSE
"""

from pathlib import Path


EXPECTED_COLUMNS = ("ID", "FORM", "OUTER_CHUNK", "INNER_CHUNK", "CLAUSE")
DEFAULT_ANNOTATED_DIR = Path("data") / "annotated"


def parse_conll_line(line, line_number, file_path):
    """Parse one token line from the project CoNLL format."""
    columns = line.split()

    if len(columns) != len(EXPECTED_COLUMNS):
        expected = len(EXPECTED_COLUMNS)
        found = len(columns)
        raise ValueError(
            f"Invalid CoNLL line in {file_path} at line {line_number}: "
            f"expected {expected} columns ({', '.join(EXPECTED_COLUMNS)}), "
            f"but found {found}. Line content: {line!r}"
        )

    token_id, form, outer_chunk, inner_chunk, clause = columns

    if not token_id.isdigit():
        raise ValueError(
            f"Invalid token ID in {file_path} at line {line_number}: "
            f"expected a positive integer, but found {token_id!r}."
        )

    return {
        "id": int(token_id),
        "form": form,
        "outer_chunk": outer_chunk,
        "inner_chunk": inner_chunk,
        "clause": clause,
    }


def read_conll_file(file_path):
    """Read a CoNLL file and return a list of sentence-level token lists."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CoNLL file not found: {file_path}")

    sentences = []
    current_sentence = []

    with file_path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            if line.startswith("#"):
                continue

            token = parse_conll_line(line, line_number, file_path)
            current_sentence.append(token)

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def load_dataset_splits(annotated_dir=DEFAULT_ANNOTATED_DIR):
    """Load train, test, and full dataset CoNLL files from the annotated directory."""
    annotated_dir = Path(annotated_dir)

    return {
        "train": read_conll_file(annotated_dir / "train.conll"),
        "test": read_conll_file(annotated_dir / "test.conll"),
        "full_dataset": read_conll_file(annotated_dir / "full_dataset.conll"),
    }


def count_tokens(sentences):
    """Count tokens in a sentence-level dataset."""
    return sum(len(sentence) for sentence in sentences)


def print_dataset_summary(dataset_splits):
    """Print a compact summary for loaded dataset splits."""
    for split_name, sentences in dataset_splits.items():
        print(
            f"{split_name}: {len(sentences)} sentences, "
            f"{count_tokens(sentences)} tokens"
        )


if __name__ == "__main__":
    splits = load_dataset_splits()
    print_dataset_summary(splits)
