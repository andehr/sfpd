import re
from string import punctuation
from importlib import import_module

import spacy
from spacy.util import compile_prefix_regex, compile_suffix_regex
from spacy.tokenizer import Tokenizer


def get_spacy_model(language):
    try:
        return add_stopwords(replace_infix_rules(spacy.load(language, disable=['tagger', 'parser', 'ner'])))
    except OSError:
        raise ValueError(f"Either the language model '{language}' is not supported in spaCy, or is not installed on your system. Try running: python -m spacy download {language}")


def replace_infix_rules(nlp):
    """
    This converts a spacy pipeline such that its tokeniser no longer separates pretty much any token. E.g. contractions,
    hyphenations, honorifics, etc.
    """
    return Tokenizer(nlp.vocab,
        prefix_search=compile_prefix_regex(nlp.Defaults.prefixes).search,
        suffix_search=compile_suffix_regex(nlp.Defaults.suffixes).search,
        infix_finditer=lambda x: iter(()),
        rules={})


EXTRA_STOPWORDS = {
    # Contractions
    "ain't", "amn't", "aren't", "can't", "'cause", "could've", "couldn't", "couldn't've", "daren't", "daresn't", "dasn't", "didn't", "doesn't", "don't", "e'er", "everyone's", "finna", "gimme", "giv'n", "gonna", "gon't", "gotta", "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's", "he've", "how'd", "howdy", "how'll", "how're", "how's", "I'd", "I'll", "I'm", "I'm'a", "I'm'o", "I've", "isn't", "it'd", "it'll", "it's", "let's", "ma'am", "mayn't", "may've", "mightn't", "might've", "mustn't", "mustn't've", "must've", "needn't", "ne'er", "o'clock", "o'er", "ol'", "oughtn't", "'s", "shalln't", "shan't", "she'd", "she'll", "she's", "should've", "shouldn't", "shouldn't've", "somebody's", "someone's", "something's", "so're", "that'll", "that're", "that's", "that'd", "there'd", "there'll", "there're", "there's", "these're", "they'd", "they'll", "they're", "they've", "this's", "those're", "'tis", "to've", "'twas", "wasn't", "we'd", "we'd've", "we'll", "we're", "we've", "weren't", "what'd", "what'll", "what're", "what's", "what've", "when's", "where'd", "where're", "where's", "where've", "which's", "who'd", "who'd've", "who'll", "whom'st", "whom'st'd've", "who're", "who's", "who've", "why'd", "why're", "why's", "won't", "would've", "wouldn't", "y'all", "y'all'd've", "you'd", "you'll", "you're", "you've"
}


def add_stopwords(nlp):
    for stopword in EXTRA_STOPWORDS:
        nlp.vocab[stopword.lower()].is_stop = True
    return nlp


class PhraseTokeniser:

    def __init__(self, language):
        self.nlp = get_spacy_model(language)
        self.language = language

    def __call__(self, text):
        return [token.text for token in self.nlp(text.strip().lower())]

    def get_stopwords(self):
        default_stopwords = import_module(f"spacy.lang.{language}.stop_words").STOP_WORDS
        return default_stopwords | EXTRA_STOPWORDS

class WordTokeniser:

    HASHTAG_PLACEHOLDER = "xxx-hashtag-xxx"

    def __init__(self, language):
        self.nlp = get_spacy_model(language)

    def __call__(self, text):
        text = text.replace("#", self.HASHTAG_PLACEHOLDER)
        tokens = [self.normalise(token) for token in self.nlp(text.strip().lower()) if not token.is_stop]
        tokens = [token for token in tokens if self.filter(token)]
        return tokens

    def filter(self, token):
        if not token:
            return False
        if token.startswith("http"):
            return False
        if re.match("^\W+$", token):
            return False
        if token.startswith("@"):
            return False
        if token.startswith("#"):
            return False
        if token == "n't" or token == "s":
            return False
        return True

    def normalise(self, token):
        text = token.text
        text = text.replace(self.HASHTAG_PLACEHOLDER, "#")
        text = text.lower()
        text = text.strip()
        if len(text) > 1 and (text.startswith("#") or text.startswith("@")):
            return text
        else:
            return text.strip(punctuation)

