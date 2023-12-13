import pandas as pd
from typing import List, Optional

import torch

def get_text(info: pd.DataFrame,
             sequence_size: int
             ) -> List[str]:
    filepath = info['text_filepath']
    ipu = info['ipu_id']
    df = pd.read_csv(filepath)

    index_ipu_6 = df[df['ipu_id'] == ipu].index[0]

    text = ' '.join(df.loc[index_ipu_6 - 5:index_ipu_6 - 1, 'text'].astype(str))
    text = text.split(' ')[-sequence_size:]
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
    df = pd.read_csv(filepath)
    frames = []
    for _ in range(frame - video_size + 1, frame + 1):
        frames.append(df.iloc[frame].tolist()[useless_info_number:])
    
    return torch.tensor(frames, dtype=torch.float32)

