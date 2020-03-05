import heapq
from collections import Counter
from math import log

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2 as chi2_feature_select

from sfpd.llr import llr_compare
from sfpd.tokenise import WordTokeniser


def count_words(texts, min_count=0, language="en"):
    counter = Counter()
    tokeniser = WordTokeniser(language)
    text_count = 0

    for text in texts:
        counts = Counter(tokeniser(text))
        counter.update(counts)
        text_count += 1
        if text_count % 1000 == 0:
            print(f"\r> Processed {text_count} docs", end="", flush=True)
    print(f"\r> Processed {text_count} docs", end="", flush=True)
    return counter if min_count <= 0 else Counter({k:count for k, count in counter.items() if count > min_count})


def dateframe_from_columns(columns, column_names):
    df = pd.DataFrame(columns)
    df.columns = column_names
    return df


def top_words_sfpd(target_counts, background_counts, n=100, l=0.4, smoothing=0.1):
    total_target = sum(target_counts.values())
    print(f"> Target total words: {total_target}")
    total_background = sum(target_counts.values())
    print(f"> Background total words: {total_background}")

    vocab_target = len(target_counts)
    print(f"> Target vocabulary size: {vocab_target}")
    vocab_background = len(background_counts)
    print(f"> Background vocabulary size: {vocab_background}")

    def score(feature):
        # L * log(P(feature|target)) + (1-L)*log(P(feature|target)/P(feature|background))

        target_p = (target_counts[feature] + smoothing) / (total_target + smoothing*vocab_target)
        background_p = (background_counts[feature] + smoothing) / (total_background + smoothing*vocab_background)

        weightedLikelihood = l * log(target_p)
        weightPMI = (1-l) * (log(target_p) - log(background_p))

        return weightedLikelihood + weightPMI

    words = heapq.nlargest(n, target_counts.keys(), key=score)

    columns = [(word, score(word), target_counts[word], background_counts[word]) for word in words]
    return dateframe_from_columns(columns, ["word", "score", "frequency/target", "frequency/background"])


def top_words_chi2(target_counts, background_counts, smoothing=0.1, n=100):
    # Get all features, ordered by how frequently they occur in the target set
    all_features = sorted(set().union(target_counts, background_counts), key=lambda f: target_counts[f], reverse=True)
    counts = [(feature, target_counts[feature] + smoothing, background_counts[feature] + smoothing) for feature in all_features]
    counts = pd.DataFrame(counts, columns=["name", "target", "background"])

    # So that the rows are target/background and columns are features
    X = counts[["target", "background"]].transpose().values
    y = np.array([0, 1]).T
    # Get the scores for the target position (why is target 1? Good Q. I only know because of inspecting results...)
    target_scores = chi2_feature_select(X,y)[1]
    feature_scores = sorted(zip(counts['name'], target_scores), key=lambda feature_score: feature_score[1], reverse=True)[:n]
    columns = [(feature, score, target_counts[feature], background_counts[feature]) for feature, score in feature_scores]
    return dateframe_from_columns(columns, ["word", "score", "frequency/target", "frequency/background"])


def top_words_llr(target_counts, background_counts, n=100):
    print("\n> Computing LLR")
    diff = llr_compare(target_counts, background_counts)
    columns = [(k, v, target_counts[k], background_counts[k]) for k,v in sorted(diff.items(), key=lambda x: x[1], reverse=True)[:n]]
    return dateframe_from_columns(columns, ["word", "score", "frequency/target", "frequency/background"])





