import re
from string import punctuation

import spacy


def get_spacy_model(language):
    try:
        return spacy.load(language, disable=['tagger', 'parser', 'ner'])
    except OSError:
        raise ValueError(f"Either the language model '{language}' is not supported in spaCy, or is not installed on your system. Try running: python -m spacy downlod {language}")


class PhraseTokeniser:

    def __init__(self, language):
        self.nlp = get_spacy_model(language)

    def __call__(self, text):
        return [token.text for token in self.nlp(text)]


class WordTokeniser:

    HASHTAG_PLACEHOLDER = "zzzHASHTAGzzz"

    def __init__(self, language):
        self.nlp = get_spacy_model(language)

    def __call__(self, text):
        text = text.replace("#", self.HASHTAG_PLACEHOLDER)
        tokens = [self.normalise(token) for token in self.nlp(text.strip()) if not token.is_stop]
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
