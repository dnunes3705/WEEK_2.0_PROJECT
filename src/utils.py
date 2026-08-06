import pandas as pd


def load_csv(path, **kwargs):
    """Load a CSV into a DataFrame."""
    return pd.read_csv(path, **kwargs)
