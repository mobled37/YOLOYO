import os
import torch
import json
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from torch.utils.data.dataloader import DataLoader

class PSAD_Dataset(Dataset):

    def __init__(self,
                 filename_dir,
                 annotation_dir,
                 audio_dir):

        self.filename_csv = pd.read_csv(filename_dir)
        self.annotation_dir = annotation_dir
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.filename_csv)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        annotation_path = self._get_annotation_path(index)
        label = self._get_audio_label(annotation_path)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.filename_csv.iloc[index][0])
        return path

    def _get_annotation_path(self, index):
        path = os.path.join(self.annotation_dir, self.filename_csv.iloc[index][1])
        return path

    def _get_audio_label(self, annotation_path):
        data_list = pd.read_json(annotation_path)
        start_time_list = []
        for idx in range(len(data_list)):
            start_time_list.append(data_list['speech_segments'][idx]['start_time'])
        return start_time_list


if __name__ == "__main__":
    FILENAME_DIR = '/Users/valleotb/Desktop/Valleotb/sample_filename/metadata.csv'
    ANNOTATION_DIR = '/Users/valleotb/Desktop/Valleotb/sample_metadata'
    AUDIO_DIR = '/Users/valleotb/Desktop/Valleotb/sample_save'

    dataset = PSAD_Dataset(
        filename_dir=FILENAME_DIR,
        annotation_dir=ANNOTATION_DIR,
        audio_dir=AUDIO_DIR
    )

    # print(f"There are {len(PSAD_Dataset)} samples in the dataset.")
    temp_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)
    # print(PSAD_Dataset)
    print(next(iter(temp_loader)))
