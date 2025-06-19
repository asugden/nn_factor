import numpy as np
import pandas as pd


def read_width(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read width data

    Args:
        path (str): location of file

    Returns:
        pd.DataFrame: the complete data with 'width' as the labels
    """
    df = pd.read_csv(path)
    return df.drop(columns=["width"]).to_numpy(), df["width"].to_numpy()
