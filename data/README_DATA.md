# Dataset Documentation

This directory contains the CoNLL-style dataset used for Turkish multi-level
phrase chunking.

The dataset is manually annotated for three token-level prediction targets:

- `OUTER_CHUNK`
- `INNER_CHUNK`
- `CLAUSE`

## File Structure

| Path | Description |
| --- | --- |
| `annotated/full_dataset.conll` | Complete annotated dataset |
| `annotated/train.conll` | Training split |
| `annotated/test.conll` | Test split |
| `raw/` | Optional location for raw Turkish sentences before annotation |

## Dataset Size

| Split | Sentences | Tokens |
| --- | ---: | ---: |
| Full dataset | 100 | 679 |
| Train | 80 | 535 |
| Test | 20 | 144 |

The split is approximately 80% train and 20% test.

## CoNLL Format

Each non-empty line represents one token. Sentences are separated by a blank
line. Comment lines start with `#`.

```text
ID FORM OUTER_CHUNK INNER_CHUNK CLAUSE
```

Example:

```text
# text = Ayşe dün kütüphanede kitabı dikkatlice okudu.
# columns = ID FORM OUTER_CHUNK INNER_CHUNK CLAUSE
1 Ayşe B-NP _ O
2 dün B-ADVP _ O
3 kütüphanede B-NP _ O
4 kitabı B-NP _ O
5 dikkatlice B-ADVP _ O
6 okudu B-VP _ O
7 . O _ O
```

## Label Sets

### OUTER_CHUNK

`OUTER_CHUNK` marks the main phrase-level chunk of each token.

| Label | Meaning |
| --- | --- |
| `B-NP` | Beginning of a noun phrase |
| `I-NP` | Inside a noun phrase |
| `B-VP` | Beginning of a verb phrase |
| `I-VP` | Inside a verb phrase |
| `B-ADVP` | Beginning of an adverbial phrase |
| `I-ADVP` | Inside an adverbial phrase |
| `O` | Outside a phrase chunk |

### INNER_CHUNK

`INNER_CHUNK` marks the narrower embedded clause trigger or predicate area.
It is intentionally more compact than `CLAUSE`.

| Label | Meaning |
| --- | --- |
| `B-RELCL` | Relative clause predicate/trigger |
| `B-COMPCL` | Complement/adverbial clause predicate/trigger |
| `_` | No inner chunk |

### CLAUSE

`CLAUSE` marks the broader embedded clause span.

| Label | Meaning |
| --- | --- |
| `B-RELCL` | Beginning of a relative clause span |
| `I-RELCL` | Inside a relative clause span |
| `B-COMPCL` | Beginning of a complement/adverbial clause span |
| `I-COMPCL` | Inside a complement/adverbial clause span |
| `O` | Outside an embedded clause |

## Label Distribution

### OUTER_CHUNK

| Label | Count |
| --- | ---: |
| `B-ADVP` | 106 |
| `I-ADVP` | 40 |
| `B-NP` | 187 |
| `I-NP` | 95 |
| `B-VP` | 137 |
| `I-VP` | 8 |
| `O` | 106 |

### INNER_CHUNK

| Label | Count |
| --- | ---: |
| `B-COMPCL` | 31 |
| `B-RELCL` | 40 |
| `_` | 608 |

### CLAUSE

| Label | Count |
| --- | ---: |
| `B-COMPCL` | 31 |
| `I-COMPCL` | 39 |
| `B-RELCL` | 40 |
| `I-RELCL` | 24 |
| `O` | 545 |

## Annotation Guidelines

- Use `B-` for the first token of a span and `I-` for following tokens in the
  same span.
- Mark time and manner adverbials such as `dün`, `bugün`, `sabah erken`,
  `dikkatlice`, `hızlıca`, and `düzgün bir şekilde` as `ADVP` when they are
  used as adverbial phrases.
- Use `RELCL` for relative clause constructions such as `önerdiği`,
  `hazırladığı`, `gelen`, `başlayan`, and `yazılan`.
- Use `COMPCL` for complement or adverbial clause constructions such as
  `gelince`, `bitince`, `konuşurken`, `başlamadan`, `çıkınca`, and
  `ilerledikçe`.
- Keep `INNER_CHUNK` narrower than `CLAUSE`: `INNER_CHUNK` marks the embedded
  trigger/predicate, while `CLAUSE` marks the broader embedded clause span.
- Do not include independent adverbial expressions in `RELCL` or `COMPCL`
  spans unless they are part of the embedded clause itself.

## Notes

The dataset is designed for a university-level classical machine learning NLP
project. It includes simple sentences, noun phrases, adverbial phrases,
relative clauses, complement/adverbial clauses, coordinated sentences, and
academic or technical sentence examples.
