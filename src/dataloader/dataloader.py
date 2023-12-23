import os
import pandas as pd
from typing import List, Dict
from transformers import DistilBertTokenizer

import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader

try:
    from dataloader import get_data
except:
    import get_data

torch.random.seed()


class DataGenerator(Dataset):
    def __init__(self,
                 mode: str,
                 data_path: str,
                 load: Dict[str, bool],
                 sequence_size: int,
                 audio_size: int,
                 video_size: int) -> None:
        super().__init__()

        print(f'creating {mode} generator')
        self.mode = mode
        self.data_path = data_path
        self.load = load

        self.sequence_size = sequence_size
        self.audio_size = audio_size
        self.video_size = video_size

        self.df = pd.read_csv(os.path.join(data_path, f"item_{mode}.csv"))
        self.num_data = len(self.df)

        self.num_line_to_load_for_text = 8

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self) -> int:
        return self.num_data
    
    def __getitem__(self, index: int) -> List[Tensor]:
        """ renvoie 4 tensors: text, audio, video et label
        les 3 premiers peuvent valoir torch.zeros(0) si load[<type>]=False

        SHAPE AFTER BATCH
        text  shape:    (batch_size, sequence size)
        audio shape:    (batch_size, audio_length, 2)
        video shape:    (batch_size, num_frames, num_features, 2)
        label shape:    (batch_size, num_classes)
        """
        line = self.df.loc[index]
        label = torch.tensor(int(line['label']))
        label = one_hot(label, num_classes=2).to(torch.float32)
        text, audio, video = [torch.zeros(1)] * 3

        if self.load['text']:
            text = get_data.get_text(info=line, num_line_to_load=self.num_line_to_load_for_text)
            text = self.tokenizer(text)['input_ids'][:self.sequence_size]

            if len(text) < self.sequence_size:
                error_message = f'text must be have more than {self.sequence_size} elements, but found only {len(text)} elements.'
                error_message += f"\n the file is: {line['text_filepath']} in the line (ipu)={line['ipu_id']}. Number line to load is {self.num_line_to_load_for_text}.\n"
                raise ValueError(error_message)
            
            text = torch.tensor(text)
        
        if self.load['audio']:
            audio = get_data.get_audio_sf(info=line, audio_length=self.audio_size)

        if self.load['video']:
            s0 = get_data.get_frame(info=line, video_size=self.video_size, speaker=0)
            s1 = get_data.get_frame(info=line, video_size=self.video_size, speaker=1)

            # video = torch.concat([s0, s1], dim=0)
            video = torch.stack([s0, s1], dim=len(s0.shape))
        
        return text, audio, video, label
        


def create_dataloader(mode: str, config: dict) -> DataLoader:
    assert mode in ['train', 'val', 'test'], f"mode must be train, val or test but is '{mode}'"

    load = dict(map(lambda x: (x, config.task in [x, 'all']), ['text', 'audio', 'video']))

    generator = DataGenerator(mode=mode,
                              data_path=config.data.path,
                              load=load, 
                              sequence_size=config.data.sequence_size,
                              audio_size=config.data.audio_length,
                              video_size=config.data.num_frames)
       
    dataloader = DataLoader(generator,
                            batch_size=config.learning.batch_size,
                            shuffle=True,
                            drop_last=True)
    return dataloader




if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    from icecream import ic

    stream = open('config/config.yaml', 'r')
    config = EasyDict(yaml.safe_load(stream))
    config.task = 'all'
    ic(config)

    test_dataloader = create_dataloader(mode='test', config=config)
    text, audio, video, label = next(iter(test_dataloader))
    print('text shape:', text.shape)
    print('audio shape:', audio.shape)
    print('video shape:', video.shape)
    print('label shape:', label.shape)
