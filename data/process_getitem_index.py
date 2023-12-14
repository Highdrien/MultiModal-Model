import os
import csv
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional


FILES  ={"train": ["LJ_JL", "AA_OR", "PR_EM", "MC_MRH", "ER_AG", "CM_MCC", "FB_CB", "JDS_LS", "PO_MH", "ML_HE", "BE_CR", "AS_EP", "NL_PG", "AW_CG", "MA_PC"],
         "val": ["JS_CL", "MD_AD", "JA_EA", "RPA_BN", "LP_MA"],
         "test": ["LS_NA", "FS_MG", "AC_MZ", "LE_LB", "LB_MA"]}


def process_text_index(file: str,
                       data_path: Optional[str]='data'
                       ) -> Dict[str, list]:
    """
    file: (filename, [speaker1, speaker2])
    """
    filepath = os.path.join(data_path, 'transcr', file + '_merge.csv')
    df = pd.read_csv(filepath)
    
    output = {'text_filepath': [filepath] * len(df),
              'ipu_id': df['ipu_id'].tolist(),
              'stoptime': df['stop'].tolist(),
              'label': df['turn_after'].tolist()}
    
    return output


def process_index_video(file: str,
                        speakers: Tuple[str, str],
                        text_output: Dict[str, list],
                        data_path: Optional[str]='data',
                        step_frame: Optional[float]=0.040
                        ) -> None:
    data_path = os.path.join(data_path, 'video', 'openface')
    file_list = os.listdir(data_path)

    dataset = 'cheese'
    filename = f"{dataset}_{file}_{speakers[0]}.csv"

    if filename not in file_list:
        dataset = 'paco'
    
    for n_speaker in range(len(speakers)):
        frame_index = []
        filename = os.path.join(data_path, f"{dataset}_{file}_{speakers[n_speaker]}.csv")

        df = pd.read_csv(filename)
        timestamp = df[' timestamp'].tolist()
        max_timestamp = max(timestamp)
        del df

        for time in tqdm(text_output['stoptime'], desc=f"file:{file}"):
            if time < max_timestamp:
                frame_time = step_frame * (time // step_frame)
                frame_time = round(frame_time, 3)
                if frame_time not in timestamp:
                    print(f"attention erreur: {file = }, {time = }, {frame_time = }")
                    exit()
                index = timestamp.index(frame_time) + 1 # car les images commance à 1
                frame_index.append(index)
    
        output[f'frame_path_{n_speaker}'] = [filename] * len(frame_index)
        output[f'frame_index_{n_speaker}'] = frame_index


def process_audio(speakers: Tuple[str, str],
                  output: Dict[str, list],
                  data_path: Optional[str]='data'
                  ) -> None:
    file_name = f"{speakers[0]}_{speakers[1]}.wav"
    filepath = os.path.join(data_path, 'audio', '2_channels', file_name)
    output['audio_filepath'] = [filepath] * len(output['text_filepath'])




def merge_all_files(list_dico: List[Dict[str, list]]) -> Dict[str, list]:
    big_dict = list_dico[0]
    N = len(big_dict['text_filepath'])
    big_dict['item'] = list(range(0, N))

    for dico in list_dico[1:]:
        n = len(dico['text_filepath'])
        for key in dico.keys():
            big_dict[key] += dico[key]
        big_dict['item'] += list(range(N, N + n))
        N += n 

    return big_dict


def save_output(dico: Dict[str, list], mode: str) -> None:
    with open(os.path.join('data', f'item_{mode}.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(dico.keys())
        writer.writerows(zip(*dico.values()))
        f.close()



if __name__ == '__main__':
    data_path = 'data'
    for mode in ['val', 'train', 'test']:
        print(f'{mode = }')
        dict_list = []
        for file in FILES[mode]:
            speakers = file.split('_')
            file_name = speakers[0] + speakers[1]
            print(file_name, speakers)
            output = process_text_index(file=file_name, data_path=data_path)
            process_index_video(file=file_name, speakers=speakers, text_output=output)
            process_audio(speakers=speakers, output=output, data_path=data_path)
            dict_list.append(output)
        big_dict = merge_all_files(list_dico=dict_list)
        save_output(dico=big_dict, mode=mode)
        del big_dict, dict_list
