# Dataset Documentation

This directory will contain the CoNLL-style dataset for Turkish multi-level phrase chunking.

## Format

Each non-empty line will represent one token with the following columns:

```text
ID FORM OUTER_CHUNK INNER_CHUNK CLAUSE
```

Sentences will be separated by a blank line.

## Files

- `annotated/full_dataset.conll`: complete manually annotated dataset
- `annotated/train.conll`: training split
- `annotated/test.conll`: test split
- `raw/`: optional raw Turkish sentences before annotation

## Annotation Status

The dataset has not been added yet. Annotation guidelines and dataset statistics will be documented in a later step.
