"""Feature extraction utilities for token-level chunking models."""

import string


TURKISH_SUFFIXES = (
    "lar",
    "ler",
    "dan",
    "den",
    "tan",
    "ten",
    "da",
    "de",
    "ta",
    "te",
    "nin",
    "nın",
    "nun",
    "nün",
    "yi",
    "yı",
    "yu",
    "yü",
    "dir",
    "dır",
    "dur",
    "dür",
    "miş",
    "mış",
    "muş",
    "müş",
)


def is_punctuation_token(word):
    """Return True if the token contains only punctuation characters."""
    return bool(word) and all(char in string.punctuation for char in word)


def get_turkish_suffix_features(lower_word):
    """Create simple boolean suffix features for common Turkish suffixes."""
    return {
        f"has_suffix_{suffix}": lower_word.endswith(suffix)
        for suffix in TURKISH_SUFFIXES
    }


def extract_token_features(sentence, token_index):
    """Extract features for one token while using sentence context."""
    token = sentence[token_index]
    word = token["form"]
    lower_word = word.lower()

    previous_word = ""
    next_word = ""

    if token_index > 0:
        previous_word = sentence[token_index - 1]["form"]

    if token_index < len(sentence) - 1:
        next_word = sentence[token_index + 1]["form"]

    features = {
        "word": word,
        "lower": lower_word,
        "prefix_2": lower_word[:2],
        "prefix_3": lower_word[:3],
        "suffix_2": lower_word[-2:],
        "suffix_3": lower_word[-3:],
        "is_capitalized": word[:1].isupper(),
        "is_upper": word.isupper(),
        "is_digit": word.isdigit(),
        "is_punctuation": is_punctuation_token(word),
        "token_length": len(word),
        "previous_word": previous_word,
        "next_word": next_word,
        "is_first_token": token_index == 0,
        "is_last_token": token_index == len(sentence) - 1,
    }

    features.update(get_turkish_suffix_features(lower_word))
    return features


def extract_sentence_features(sentence):
    """Extract token-level features for a sentence."""
    return [
        extract_token_features(sentence, token_index)
        for token_index in range(len(sentence))
    ]


def extract_dataset_features(sentences):
    """Extract features for a sentence-level dataset without flattening it."""
    return [extract_sentence_features(sentence) for sentence in sentences]


if __name__ == "__main__":
    example_sentence = [
        {
            "id": 1,
            "form": "Dün",
            "outer_chunk": "B-ADVP",
            "inner_chunk": "_",
            "clause": "O",
        },
        {
            "id": 2,
            "form": "toplantıdan",
            "outer_chunk": "B-NP",
            "inner_chunk": "B-RELCL",
            "clause": "B-RELCL",
        },
        {
            "id": 3,
            "form": ".",
            "outer_chunk": "O",
            "inner_chunk": "_",
            "clause": "O",
        },
    ]

    for token_features in extract_sentence_features(example_sentence):
        print(token_features)
