"""Unit tests for the tweet preprocessing pipeline."""

import pytest

from src.preprocessing import TweetPreprocessor, light_clean, normalize_tweet


@pytest.fixture(scope="module")
def pp():
    return TweetPreprocessor()


def test_mentions_and_urls_removed():
    out = normalize_tweet("@united check https://t.co/abc123 now")
    assert "@" not in out and "http" not in out and "united" not in out


def test_html_entities_unescaped():
    assert "&amp;" not in normalize_tweet("delays &amp; cancellations")


def test_hashtag_camel_case_split():
    assert "bad service" in normalize_tweet("#BadService")


def test_contractions_expanded():
    out = normalize_tweet("I can't believe it wasn't on time and I won't fly again")
    assert "can not" in out and "was not" in out and "will not" in out


def test_elongation_squashed():
    assert "soo" in normalize_tweet("sooooo bad") and "sooo" not in normalize_tweet("sooooo bad")


def test_emoji_and_emoticons_become_tokens(pp):
    tokens = pp.process("love it :) 😡")
    assert "emo_smile" in tokens
    assert any(t.startswith("emo_") and t != "emo_smile" for t in tokens)


def test_negations_kept_but_other_stopwords_removed(pp):
    tokens = pp.process("The flight was not good")
    assert "not" in tokens
    assert "the" not in tokens and "was" not in tokens


def test_negations_removed_when_disabled():
    assert "not" not in TweetPreprocessor(keep_negations=False).process("The flight was not good")


def test_lemmatisation_is_pos_aware(pp):
    tokens = pp.process("our flights were delayed and bags lost")
    assert {"flight", "delay", "bag", "lose"} <= set(tokens)


def test_punctuation_and_numbers_removed(pp):
    tokens = pp.process("Flight 1234 delayed!!! ... ??")
    assert all(t.isalpha() or "_" in t for t in tokens)
    assert "1234" not in tokens


def test_reference_example(pp):
    assert pp.process("@united I wasn't happy, flights were DELAYED!!! #BadService") == [
        "not", "happy", "flight", "delay", "bad", "service"
    ]


def test_sequence_variant_keeps_stopwords():
    tokens = TweetPreprocessor(remove_stopwords=False).process("the flight was late")
    assert tokens[0] == "the"


def test_process_corpus_matches_process(pp):
    texts = ["great crew, thanks!", "lost my bag again"]
    assert pp.process_corpus(texts) == [pp.process(t) for t in texts]


def test_empty_and_non_string_input(pp):
    assert pp.process("@united") == []
    assert isinstance(normalize_tweet(None), str)


def test_light_clean_preserves_text_for_transformer():
    out = light_clean("@united Not happy!!! https://t.co/x &amp; tired")
    assert out == "@user Not happy!!! http & tired"
