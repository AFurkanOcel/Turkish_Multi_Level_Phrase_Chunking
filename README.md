# Turkish Multi-Level Phrase Chunking

This repository contains a Natural Language Processing course project for Turkish multi-level phrase chunking using classical machine learning methods.

## Project Description

The goal of this project is to build a token-level NLP system that predicts three types of labels for Turkish sentences:

- Outer chunk labels
- Inner chunk labels
- Clause labels

The dataset will use a CoNLL-style format with one token per line.

```text
ID FORM OUTER_CHUNK INNER_CHUNK CLAUSE
```

## Planned Models

The first complete version of the project will compare the following classical machine learning models:

- Majority Class Baseline
- Multinomial Naive Bayes
- Logistic Regression
- MLPClassifier

Conditional Random Fields (CRF) may be added later as an optional bonus model after the main system is complete.

## Planned Project Structure

```text
Turkish_Multi_Level_Phrase_Chunking/
├── data/
│   ├── raw/
│   ├── annotated/
│   │   ├── train.conll
│   │   ├── test.conll
│   │   └── full_dataset.conll
│   └── README_DATA.md
├── models/
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── error_analysis.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── report/
│   └── project_report.md
├── requirements.txt
├── README.md
└── main.py
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Current Status

Step 1: Project skeleton.

The repository currently contains the initial folder structure, placeholder files, and starter documentation. Data loading, feature extraction, model training, evaluation, prediction, and the Streamlit interface will be implemented in later steps.
