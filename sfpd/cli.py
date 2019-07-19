#!/usr/bin/env python

import argparse
import re
from itertools import islice
from pathlib import Path

import pandas as pd

from sfpd.detection import count_words, top_words_llr, top_words_sfpd, top_words_chi2, get_top_phrases, iter_large_csv_text


def parse_args():
    """Set up command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Interface for finding surprisingly frequent phrases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-t", "--target",
                        help="path to csv containing target data", metavar="PATH")

    parser.add_argument("-b", "--background",
                        help="path to csv containing background documents", metavar="PATH")

    parser.add_argument("-w", "--words",
                        help="path to csv containing words that need expanding", metavar="PATH")

    parser.add_argument("-m", "--method", choices=["llr", "sfpd", "chi2"], default="sfpd",
                        help="word extraction method")

    parser.add_argument("-c", "--pattern", help="If specified with path to target data, then an interactive session will be started for inspecting random samples of occurrences of this regex pattern.")

    parser.add_argument("--compare", metavar="PATHS", nargs="+",
                        help="Compare top phrases in csvs. The first CSV is compared to the rest. Use num-comparisons, and num-comparisons-rest to select the number of phrases compared.")
    parser.add_argument("--num-comparisons", type=int, default=50)
    parser.add_argument("--num-comparisons-rest", type=int, default=50)

    parser.add_argument("--name", help="add experiment name to file name", default="")
    parser.add_argument("--language",
                        default="en",
                        choices="de el en es fr it nl pt xx af ar bg bn ca cs da et fa fi ga he hi hr hu id is ja kn lt lv nb pl ro ru si sk sl sq sv ta te th tl tr tt uk ur vi zh".split())
    parser.add_argument("--text-col-name", default="twitter.tweet/text",
                        help="column name of text in CSV files")

    # Parameters
    parser.add_argument("-n", "--num-words", default=100, type=int,
                        help="the number of words to extract during word extraction methods")
    parser.add_argument("-l", "--likelihood-lift", default=0.4, type=float,
                        help="Parameter for SFPD method. When set to 0, features are ranked solely by how surprisingly frequently they occur in the target data vs background (lift). When set to 1, features are ranked solely according to their likelihood of occurrence in the target data (therefore ignoring the background documents). Values inbetween allow a weighted contribution of lift and likelihood.")
    parser.add_argument("--min-target-word-count", default=4.0, type=float)
    parser.add_argument("--min-background-word-count", default=1.0, type=float)

    # Parameters for phrase discovery
    parser.add_argument("--num-phrases", default=1, type=int, metavar="N")
    parser.add_argument("--min-phrase-size", default=1, type=int, metavar="N")
    parser.add_argument("--max-phrase-size", default=6, type=int, metavar="N")
    parser.add_argument("--leaf-pruning-threshold", default=0.3, type=float, metavar="N",
                        help="The fraction of times a larger phrase must appear more than a sub-phrase")
    parser.add_argument("--min-phrase-count", default=4.0, type=float, metavar="N",
                        help="The minimum occurrences of a phrase in the target data")
    parser.add_argument("--phrase-lvl1", default=5.0, type=float, metavar="N",
                        help="When the occurrences of an n+1grams parent ngram is less than this threshold, the n+1gram must have occurred 100 percent of these times to be considered.")
    parser.add_argument("--phrase-lvl2", default=7.0, type=float, metavar="N",
                        help="When the occurrences of an n+1grams parent ngram is less than this threshold, the n+1gram must have occurred 75 percent of these times to be considered.")
    parser.add_argument("--phrase-lvl3", default=15.0, type=float, metavar="N",
                        help="When the occurrences of an n+1grams parent ngram is less than this threshold, the n+1gram must have occurred 50 percent of these times to be considered.")

    return parser.parse_args()


def find_contexts_interactively(data_path, regex_pattern, text_col_name="twitter.tweet/text"):
    """
    Interactively randomly sample examples of pattern in text data. Ignores case of pattern.
    :param data_path: Path to data csv
    :param regex_pattern: pattern to search for in data text
    :param text_col_name: the name of the csv column containing the text
    """
    print(f"> Printing contexts of {regex_pattern}. Press ENTER for more.")

    def sampler(data_path, context):
        return (text for data_chunk in pd.read_csv(data_path, chunksize=10000)
                        for text in data_chunk[text_col_name].values
                            if re.findall(context, text, flags=re.IGNORECASE))

    text_iter = sampler(data_path, regex_pattern)
    quit = False
    while not quit:
        try:
            sample = list(islice(text_iter, 10))
            if sample:
                for text in sample:
                    print(text)
                ans = input("> Press ENTER for more. Or enter new pattern. or enter :q to quit: ")
                if not ans.strip().lower().startswith(":q"):
                    if ans:
                        text_iter = sampler(data_path, ans)
                    continue
        except KeyboardInterrupt:
            print()
        quit = True


def compares_csvs(csvs, n=20, m=20, word_col_name="word", phrase_col_name="phrases", count_col_name="count"):
    """
    Given multiple CSVs of output from the surprising phrases methods, compare the phrases generated.

    :param csvs: The first CSV is the main comparison point, it is compared one-vs-the-rest
    :param n: The number of phrases to inspect from the first CSV
    :param m: The number of phrases to inspect from each of the other CSVs
    :return:
    """
    paths = [Path(csv) for csv in csvs]
    dfs = [pd.read_csv(path) for path in paths]
    df1 = dfs[0].head(n)
    df2 = pd.concat(df.head(m) for df in dfs[1:])
    items1 = set(df1[word_col_name].values)
    items2 = set(df2[word_col_name].values)
    print(f"> In {paths[0].name}, not in {[path.name for path in paths[1:]]} ({len(items1 - items2)})")
    for item in (items1 - items2):
        row = df1[df1[word_col_name] == item].head(1)
        print(f"  {row[word_col_name].values[0]}: {row[phrase_col_name].values[0]} ({row[count_col_name].values[0]})")

    print(f"> In {[path.name for path in paths[1:]]}, not in {paths[0].name} ({len(items2 - items1)})")
    for item in (items2 - items1):
        row = df2[df2[word_col_name] == item].head(1)
        print(f"  {row[word_col_name].values[0]}: {row[phrase_col_name].values[0]} ({row[count_col_name].values[0]})")

    print(f"> In both ({len(items1 & items2)})")
    for item in (items1 & items2):
        row = df2[df2[word_col_name] == item].head(1)
        print(f"  {row[word_col_name].values[0]}: {row[phrase_col_name].values[0]} ({row[count_col_name].values[0]})")


def top_phrases_to_csv(top_phrases, output_path):
    import_data = [(word, " ".join(phrases[0][0]), phrases[0][1]) for word, phrases in top_phrases.items()]
    df = pd.DataFrame(import_data)
    df.columns = ["word", "phrases", "count"]
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    args = parse_args()

    print("-- Configuration Options --")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")
    print("---------------------------")

    # Find interesting words
    words = None
    if args.target and args.background:

        print("> Counting target words")
        target_counts = count_words(iter_large_csv_text(args.target, args.text_col_name), args.min_target_word_count, args.language)
        print("\n> Counting background words")
        background_counts = count_words(iter_large_csv_text(args.background, args.text_col_name), args.min_background_word_count, args.language)

        print("\n> Extracting surprisingly frequent words")
        if args.method == "llr":
            words = top_words_llr(target_counts, background_counts, args.num_words)
        elif args.method == "sfpd":
            words = top_words_sfpd(target_counts, background_counts, args.num_words, args.likelihood_lift)
        elif args.method == "chi2":
            words = top_words_chi2(target_counts, background_counts, args.num_words)
        else:
            raise ValueError(f"No such method: {args.method}")

        words.to_csv(Path(args.target).with_suffix(f".{args.method}.{args.name}.words.csv"), index=False)

    # Find frequent phrases
    if args.words:
        words = pd.read_csv(args.words)
    if words is not None and args.target:
        print("> Expanding to phrases")

        top_phrases = get_top_phrases(
            words["word"].values,
            iter_large_csv_text(args.target, args.text_col_name),
            args.num_phrases,
            args.language,
            args.min_phrase_size,
            args.max_phrase_size,
            args.leaf_pruning_threshold,
            args.min_phrase_count,
            args.phrase_lvl1,
            args.phrase_lvl2,
            args.phrase_lvl3
        )

        for word, phrases in top_phrases.items():
            phrase = " ".join(phrases[0][0])
            print(f"{word}: {phrase} ({phrases[0][1]})")

        top_phrases_to_csv(top_phrases, Path(args.target).with_suffix(f".{args.method}.{args.name}.phrases.csv"))

    # Explore string contexts
    if args.target and args.pattern:
        find_contexts_interactively(args.target, args.pattern)

    if args.compare:
        compares_csvs(args.compare, n=args.comparisons, m=args.comparisons_rest)
