"""Text preprocessing for customer-feedback tweets.

Two pipelines are provided because different model families need different inputs:

* ``TweetPreprocessor`` – the full classical-NLP pipeline required by the
  assignment: normalisation → tokenisation → POS-aware lemmatisation →
  punctuation/number removal → stop-word removal. Used for TF-IDF, Word2Vec and
  (with stop words kept) the BiLSTM.
* ``light_clean`` – minimal normalisation for the Transformer. Pre-trained
  sub-word models were trained on natural text, so removing stop words, punctuation
  or inflection would *destroy* information they rely on (e.g. "not", "!!!").

Design decisions (justified in the paper, Section 4):

* **Negations are kept** even though NLTK lists them as stop words. Removing "not"
  turns "not happy" into "happy" and flips the sentiment.
* **@mentions and URLs are removed.** Mentions identify the airline, not the
  sentiment; keeping them would let the model learn airline base rates.
* **Emoji and emoticons are converted to words** (``😡`` → ``emo_pouting_face``)
  rather than deleted, because they are strong sentiment cues in social media.
* **Hashtags keep their words** (``#BadService`` → ``bad service``).
* **Contractions are expanded** before tokenising so "can't" yields "can not".
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from functools import lru_cache

import emoji
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

NLTK_RESOURCES = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}

# Words that NLTK treats as stop words but that carry sentiment polarity or
# intensity in customer feedback. They are retained.
SENTIMENT_STOPWORDS = {
    "no", "not", "nor", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "cannot", "against", "very", "too", "most", "more", "only",
}

# Common ASCII emoticons → sentiment-bearing placeholder tokens.
EMOTICONS = {
    r"(?::|=|;)-?(?:\)|\]|D|P|p)": " emo_smile ",
    r"<3": " emo_heart ",
    r"(?::|=)'?-?(?:\(|\[|/|\\|\|)": " emo_frown ",
}

CONTRACTIONS = [
    (r"won['’]t", "will not"),
    (r"can['’]t", "can not"),
    (r"shan['’]t", "shall not"),
    (r"n['’]t\b", " not"),
    (r"['’]re\b", " are"),
    (r"['’]m\b", " am"),
    (r"['’]ll\b", " will"),
    (r"['’]ve\b", " have"),
    (r"['’]d\b", " would"),
    (r"['’]s\b", ""),  # possessive or "is" – ambiguous and low information
]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
ELONGATION_RE = re.compile(r"(\w)\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")
VALID_TOKEN_RE = re.compile(r"^[a-z][a-z_]+$")  # ≥2 letters; drops punctuation & numbers


def ensure_nltk_resources(quiet: bool = True) -> None:
    """Download the NLTK corpora used here if they are not installed yet."""
    for resource_path, package in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=quiet)


def _penn_to_wordnet(tag: str) -> str:
    """Map a Penn Treebank POS tag to the WordNet POS expected by the lemmatiser."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def _replace_hashtag(match: re.Match) -> str:
    """'#BadService' → ' Bad Service ' so the words become ordinary tokens."""
    return " " + CAMEL_RE.sub(" ", match.group(1)) + " "


def normalize_tweet(text: str) -> str:
    """Social-media normalisation shared by every classical pipeline.

    Steps: HTML unescape → strip URLs and @mentions → split hashtags → emoticons
    and emoji to tokens → lowercase → expand contractions → squash elongations
    ("soooo" → "soo") → collapse whitespace.
    """
    text = html.unescape(str(text))
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(_replace_hashtag, text)
    for pattern, token in EMOTICONS.items():
        text = re.sub(pattern, token, text)
    text = emoji.demojize(text, delimiters=(" emo_", " "))
    text = text.lower()
    for pattern, replacement in CONTRACTIONS:
        text = re.sub(pattern, replacement, text)
    text = ELONGATION_RE.sub(r"\1\1", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def light_clean(text: str) -> str:
    """Minimal cleaning for Transformer input (keeps case, punctuation, stop words).

    Following the TweetEval convention, user handles become ``@user`` and links
    become ``http`` so the model is not distracted by unique identifiers.
    """
    text = html.unescape(str(text))
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    return WHITESPACE_RE.sub(" ", text).strip()


class TweetPreprocessor:
    """Configurable tokenise → lemmatise → filter pipeline.

    Parameters
    ----------
    remove_stopwords:
        Drop English stop words (except :data:`SENTIMENT_STOPWORDS`).
    lemmatize:
        Apply POS-aware WordNet lemmatisation ("delayed" → "delay").
    keep_negations:
        Keep negation/intensity words even when ``remove_stopwords`` is True.

    Examples
    --------
    >>> TweetPreprocessor().process("@united I wasn't happy, flights were DELAYED!!! #BadService")
    ['not', 'happy', 'flight', 'delay', 'bad', 'service']
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        keep_negations: bool = True,
    ) -> None:
        ensure_nltk_resources()
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.keep_negations = keep_negations
        self._tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        stops = set(stopwords.words("english"))
        if keep_negations:
            stops -= SENTIMENT_STOPWORDS
        self.stopwords = frozenset(stops)

    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> list[str]:
        """Normalise then split into tokens (punctuation still present)."""
        return self._tokenizer.tokenize(normalize_tweet(text))

    def _lemma(self, token: str, tag: str) -> str:
        if token.startswith("emo_"):
            return token
        return _lemma_lookup(token, _penn_to_wordnet(tag))

    def _filter(self, tokens: Iterable[str]) -> list[str]:
        kept = []
        for tok in tokens:
            if not VALID_TOKEN_RE.match(tok):
                continue  # punctuation, numbers, single characters
            if self.remove_stopwords and tok in self.stopwords:
                continue
            kept.append(tok)
        return kept

    def process(self, text: str) -> list[str]:
        """Full pipeline for a single document."""
        return self.process_corpus([text])[0]

    def process_corpus(self, texts: Iterable[str]) -> list[list[str]]:
        """Full pipeline for many documents (batched POS tagging is faster).

        POS tagging is done *before* stop-word removal so the tagger sees
        grammatical context, which improves lemma accuracy.
        """
        tokenized = [self.tokenize(t) for t in texts]
        if self.lemmatize:
            tagged = nltk.pos_tag_sents(tokenized)
            tokenized = [[self._lemma(tok, tag) for tok, tag in sent] for sent in tagged]
        return [self._filter(tokens) for tokens in tokenized]

    @staticmethod
    def join(token_lists: Iterable[list[str]]) -> list[str]:
        """Join token lists back to space-separated strings for vectorisers."""
        return [" ".join(tokens) for tokens in token_lists]


_LEMMATIZER = WordNetLemmatizer()


@lru_cache(maxsize=50_000)
def _lemma_lookup(token: str, pos: str) -> str:
    """Memoised lemmatisation – the vocabulary is small relative to the corpus."""
    return _LEMMATIZER.lemmatize(token, pos)
