"""Streamlit interface for Turkish multi-level phrase chunking."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import DEFAULT_MODEL_NAME, predict_sentence, save_predictions_as_conll
from src.train import MODEL_DISPLAY_NAMES


DEFAULT_SENTENCE = "Dün akşam öğrenci makaleyi okudu."


def rows_to_dataframe(rows):
    """Convert prediction rows to a display-friendly DataFrame."""
    return pd.DataFrame(
        [
            {
                "ID": row["id"],
                "FORM": row["form"],
                "OUTER_CHUNK": row["outer_chunk"],
                "INNER_CHUNK": row["inner_chunk"],
                "CLAUSE": row["clause"],
            }
            for row in rows
        ]
    )


def main():
    """Run the Streamlit prediction interface."""
    st.set_page_config(
        page_title="Turkish Multi-Level Phrase Chunking",
        layout="wide",
    )

    st.title("Turkish Multi-Level Phrase Chunking")

    model_name = st.selectbox(
        "Model",
        options=list(MODEL_DISPLAY_NAMES.keys()),
        format_func=lambda name: MODEL_DISPLAY_NAMES[name],
        index=list(MODEL_DISPLAY_NAMES.keys()).index(DEFAULT_MODEL_NAME),
    )

    sentence_text = st.text_area(
        "Turkish sentence",
        value=DEFAULT_SENTENCE,
        height=100,
    )

    predict_clicked = st.button("Predict")

    if predict_clicked:
        try:
            rows = predict_sentence(
                sentence_text=sentence_text,
                model_name=model_name,
            )
            output_path = save_predictions_as_conll(rows)
            st.dataframe(rows_to_dataframe(rows), use_container_width=True)
            st.success(f"Predicted CoNLL output saved to: {output_path}")
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
