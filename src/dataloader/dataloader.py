import os
import json
from icecream import ic
from easydict import EasyDict
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

import process_getitem_index

LOAD = {'audio': False, 'text': True, 'video': False}


class DataGenerator(Dataset):
    def __init__(self,
                 mode: str,
                 data_path: str,
                 load: Dict[str, bool],
                 sequence_size: int,
                 audio_size: int,
                 num_frame: int) -> None:
        super().__init__()

        print(f'creating {mode} generator')
        self.mode = mode
        self.data_path = data_path
        self.load = load
        filename = os.path.join(data_path, 'filename.json')
        with open(filename, 'r') as f:
            filename = json.load(f)
            f.close()

        files = filename[mode]
        speakers_list = list(map(lambda x: x.split('_'), files))

        self.files = {}
        for i in range(len(files)):
            file_name = speakers_list[i][0] + speakers_list[i][1]
            self.files[file_name] = speakers_list[i] 
        ic(self.files)

    def process_batch_index(self) -> list:
        text_output = process_getitem_index.process_text_index(self.files, self.data_path)
        return text_output


if __name__ == '__main__':
    generator = DataGenerator(mode='val',
                              data_path='data',
                              load=LOAD, 
                              sequence_size=10,
                              audio_size=1,
                              num_frame=10)
    print(generator.process_batch_index())


