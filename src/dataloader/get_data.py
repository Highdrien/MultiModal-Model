import pandas as pd
import soundfile as sf

import torch
from torch import Tensor


def get_text(info: pd.DataFrame, num_line_to_load: int = 5) -> list[str]:
    """
    Get a specified number of lines from a CSV file and return the text as a list of words.

    Args:
        info (pd.DataFrame): A DataFrame containing the file path and IPU ID.
        num_line_to_load (int, optional): The number of lines to load from the file. Defaults to 5.

    Returns:
        list[str]: A list of words from the loaded text.
    """
    filepath = info["text_filepath"]
    ipu = info["ipu_id"]

    df = pd.read_csv(
        filepath, skiprows=range(1, ipu - num_line_to_load + 2), nrows=num_line_to_load
    )

    text = df["text"].str.cat(sep=" ")
    return text


def get_frame(
    info: pd.DataFrame, video_size: int, speaker: int, useless_info_number: int = 5
) -> Tensor:
    """
    Retrieves the last <video_size> frames for a given speaker from a DataFrame.

    Args:
        info (pd.DataFrame): DataFrame containing information about the frames.
        video_size (int): Number of frames to retrieve.
        speaker (int): Identifier for the speaker.
        useless_info_number (int, optional):
            Number of initial columns to skip in the DataFrame. Defaults to 5.

    Returns:
        Tensor: A tensor containing the last <video_size> frames
            with shape (<video_size>, 709).
    """
    filepath = info[f"frame_path_{speaker}"]
    frame = info[f"frame_index_{speaker}"]
    df = pd.read_csv(
        filepath, skiprows=range(1, frame - video_size + 1), nrows=video_size
    )

    colonnes_a_inclure = df.columns[useless_info_number:]
    frames = df[colonnes_a_inclure].astype("float32").to_numpy()
    frames = torch.tensor(frames)

    return frames


def get_audio_sf(info: pd.DataFrame, audio_length: int) -> Tensor:
    """
    Extracts a segment of audio from a file using soundfile and converts it to a PyTorch tensor.

    Args:
        info (pd.DataFrame): A DataFrame containing audio file information,
            including the 'stoptime' and 'audio_filepath'.
        audio_length (int): The length of the audio segment to extract, in milliseconds.

    Returns:
        Tensor: A tensor containing the extracted audio segment, converted to float32.
    """
    end_time = int(info["stoptime"] * 1000)
    audio, _ = sf.read(
        file=info["audio_filepath"], start=end_time - audio_length, stop=end_time
    )
    audio = torch.tensor(audio).to(torch.float32)
    return audio
