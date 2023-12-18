import pandas as pd
import soundfile as sf
from typing import List, Optional

import torch


def get_text(info: pd.DataFrame) -> List[str]:
    filepath = info['text_filepath']
    ipu = info['ipu_id']

    df = pd.read_csv(filepath,
                     skiprows=range(1, ipu - 8 + 2),
                     nrows=5)
    
    text = df['text'].str.cat(sep=' ')
    return text


def get_frame(info: pd.DataFrame,
              video_size: int,
              speaker: int,
              useless_info_number: Optional[int]=5
              ) -> torch.Tensor:
    """
    get the last <video_size> frame
    output shape: (<video_size>, 709)
    """
    filepath = info[f'frame_path_{speaker}']
    frame = info[f'frame_index_{speaker}']
    df = pd.read_csv(filepath,
                     skiprows=range(1, frame - video_size + 1),
                     nrows=video_size)

    colonnes_a_inclure = df.columns[useless_info_number:]
    frames = df[colonnes_a_inclure].astype('float32').to_numpy()
    frames = torch.tensor(frames)

    return frames


def get_audio_sf(info: pd.DataFrame,
                 audio_length: int
                 ) -> torch.Tensor:
    """ audio length in ms """
    end_time = int(info['stoptime'] * 1000)
    audio, _ = sf.read(file=info['audio_filepath'],
                       start=end_time - audio_length,
                       stop=end_time)
    audio = torch.tensor(audio).to(torch.float32)
    return audio

