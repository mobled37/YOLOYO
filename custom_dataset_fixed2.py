import os
import glob
import os.path
import torch
import json
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from torch.utils.data.dataloader import DataLoader

class PSAD_Dataset(Dataset):

    def __init__(self,
                 audio_folder_dir,
                 file_name_dir,
                 device):

        self.filename_csv = pd.read_csv(file_name_dir)
        self.audio_dir = audio_folder_dir
        self.device = device

    def __len__(self):
        return len(self.filename_csv)

    def __getitem__(self, index):
        audio_dir = self._get_audio_file_path(index)
        signal, sr = torchaudio.load(audio_dir)
        signal = signal.to(self.device)
        label = audio_dir[-5]
        return signal, label

    def _get_audio_file_path(self, index):
        path = os.path.join(self.filename_csv['audio_file'][index])
        return path




if __name__ == "__main__":
    FILENAME_DIR = '/Users/valleotb/Desktop/Valleotb/sample_filename/metadata.csv'
    AUDIO_DIR = '/Users/valleotb/Desktop/Valleotb/sample_save'

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"using device: {device}")

    dataset = PSAD_Dataset(
        audio_folder_dir=AUDIO_DIR,
        file_name_dir=FILENAME_DIR,
        device=device
    )

    print(f"There are {len(dataset)} samples in the dataset.")
    print(dataset)

    # temp_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)
    # print(PSAD_Dataset)
    # print(next(iter(temp_loader)))
