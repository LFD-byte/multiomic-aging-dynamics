import pandas as pd

def read_parquet(file_path):
    """
    Read a Parquet file into a pandas DataFrame.
    """
    df = pd.read_parquet(file_path)
    df = df.reset_index()
    if 'participant_id' in df.columns:
        df['participant_id'] = df['participant_id'].astype(str)

    return df