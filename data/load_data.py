"""
Download examples:

Whole archive:
curl 'https://amubox.univ-amu.fr/s/gkfA7rZCWGQFqif/download' -H 'Connection: keep-alive' -H 'Accept-Encoding: gzip, deflate, br' --output archive.zip

Specific files with path:
curl 'https://amubox.univ-amu.fr/s/gkfA7rZCWGQFqif/download?path=%2Fvideo%2F&files=openface.zip' -H 'Connection: keep-alive' -H 'Accept-Encoding: gzip, deflate, br' --output video/openface.zip
"""

import pandas as pd
import os
from glob import glob


def load_all_ipus(folder_path: str = "transcr", load_words: bool = False):
    """
    Load all IPUs from csv files in the specified folder path.

    Parameters:
    - folder_path (str): The path to the folder containing the csv files. Default is 'transcr'.
    - load_words (bool): Whether to load words or not. Default is False.

    Returns:
    - data (pd.DataFrame): The concatenated dataframe containing all the loaded IPUs.
    """
    file_list = glob(
        os.path.join(folder_path, f"*_merge{'_words' if load_words else ''}.csv")
    )

    # Load all csv files
    data = []
    for file in file_list:
        df = pd.read_csv(file, na_values=[""])  # one speaker name is 'NA'
        df["dyad"] = file.split("/")[-1].split("_")[0]
        data.append(df)

    data = pd.concat(data, axis=0).reset_index(drop=True)
    print(data.shape)
    plabels = [
        col
        for col in data.columns
        if not any(
            [
                col.startswith(c)
                for c in [
                    "dyad",
                    "ipu_id",
                    "speaker",
                    "start",
                    "stop",
                    "text",
                    "duration",
                ]
            ]
        )
    ]
    print(data[plabels].sum(axis=0) / data.shape[0])
    return data


def filter_after_jokes(df_ipu: pd.DataFrame):
    """First few ipus are useless / common to all conversations"""
    jokes_end = (
        df_ipu[
            df_ipu.text.apply(
                lambda x: (
                    False
                    if isinstance(x, float)
                    else (
                        ("il y avait un âne" in x) or ("qui parle ça c'est cool" in x)
                    )
                )
            )
        ]
        .groupby("dyad")
        .agg({"ipu_id": "max"})
        .to_dict()["ipu_id"]
    )
    return (
        df_ipu[df_ipu.apply(lambda x: x.ipu_id > jokes_end.get(x.dyad, 0), axis=1)],
        jokes_end,
    )


if __name__ == "__main__":
    from icecream import ic

    data = load_all_ipus()
    ic(data)
