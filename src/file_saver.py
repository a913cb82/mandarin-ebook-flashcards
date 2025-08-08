
import pandas as pd
from pathlib import Path

def save_dataframe_to_tsv(df: pd.DataFrame, output_path: Path):
    """Saves a DataFrame to a TSV file."""
    df.to_csv(output_path, sep="\t", index=False, header=False)

