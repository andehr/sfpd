import pandas as pd


def iter_large_csv_text(path, text_col_name, chunksize=10000):
    """
    Return iterator over texts in a CSV, loading a fixed amount of the CSV into memory at any time.
    """
    for data_chunk in pd.read_csv(path, chunksize=chunksize):
        for text in data_chunk[text_col_name].values:
            yield text
