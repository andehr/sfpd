# Surprisingly Frequent Phrase Detection

Library implementing surprising frequent phrase detection as defined in "[Characterising Semantically Coherent Classes of Text Using Feature Discovery](http://sro.sussex.ac.uk/id/eprint/84841/)".

## Install

From source:

`python setup.py install`

From PyPi (not yet uploaded, need to check licensing issues...)

`pip install sfpd`

Install the relevant spacy language model, e.g. English "en":

`python -m spacy download en`

## Usage

### Command-line

Use command line interface:

`python -m sfpd.cli --help`

For example, the following will find surprising phrases in `target.csv` versus `background.csv` 
where the text is found under the CSV header `text`:

`python -m sfpd.cli --target target.csv --background background.csv --text-col-name text`


### Programmatically

Get an iterable of strings for the target data and background data. If you have two CSV files with text column, 
you can use a helper function:

```python
from sfpd.util import iter_large_csv_text

target = iter_large_csv_text(target_path, text_column_name)
background = iter_large_csv_text(background_path, text_column_name)
```

Build Python Counter objects for frequency distributions of the tokens in target and background:

```python
from sfpd.words import count_words

target_counts = count_words(target, min_count=4, language="en")
background_counts = count_words(background, min_count=4, language="en")
```

This provides the default way we count tokens for finding surprising words. This counting could be tailored however you 
you like by providing your own Counter objects. However, the counting for phrase expansion is done internally in a 
specific way.

Next find surprising words using one of the surprising words methods:

```python
from sfpd.words import top_words_llr, top_words_sfpd, top_words_chi2

# My method 
words = top_words_sfpd(target_counts, background_counts)

# Log likelihood ratio method from https://github.com/tdunning/python-llr
words = top_words_llr(target_counts, background_counts)

# Chi-square method
words = top_words_chi2(target_counts, background_counts)
```

The returned `words` is a pandas data frame that contains each proposed word, its score, and its count in both the 
target and background corpora.

Next you can expand these words to phrases:

```python
from sfpd.phrases import get_top_phrases

top_phrases = get_top_phrases(words["word"].values, iter_large_csv_text(target_path, text_column_name))
```

## Citation

If you use this tool, please include the following citation:

Robertson, Andrew David, 2019. *Characterising semantically coherent classes of text through feature discovery* (Doctoral dissertation, University of Sussex).

Here's the bibtex:

```bibtex
@phdthesis{robertson2019characterising,
  title={Characterising semantically coherent classes of text through feature discovery},
  author={Robertson, Andrew David},
  year={2019},
  school={University of Sussex}
}
```
