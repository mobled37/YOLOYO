import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import json

class Librispeech_Dataset(Dataset):

    def __init__(self,
                 file_name_dir,
                 ANNOTATION_DIR,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):

        #   self.annotations = pd.read_csv(ANNOTATION_DIR)
        #   self.annotations = pd.read_json(ANNOTATION_DIR)
        self.file_dir_csv = pd.read_csv(file_name_dir)
        self.annotations_dir = ANNOTATION_DIR
        self.audio_dir = audio_dir
        self.audio_label_dir = audio_label_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    # 이거 리턴이 무슨 의미지?

    # index가 어디서 오는거지?
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label_data_path = self._get_audio_label_path(index)
        # label = self._get_audio_sample_label(index)
        label = pd.read_json(label_data_path)
        signal, sr = torchaudio.load(audio_sample_path)
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        signal = signal.to(self.device)
        # signal = self._resample_if_necessary(signal, sr)
        # signal = self._mix_down_if_necessary(signal)    # mix down to mono
        # signal = self._cut_if_necessary(signal)
        # signal = self._right_pad_if_necessary(signal)
        # signal = self.transformation(signal)    # mel_spectrogram
        return signal, label

    def _cut_if_necessary(self, signal):    # cutting signal
        # signal -> Tensor -> (num_channel, num_samples) = (1, num_samples) -> signal.shape[1] = num_samples
        # ex (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # [1, 1, 1] -> [1, 1, 1, 0, 0] -> padding
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)     # (1, 1, 2, 2)
            # (1, num_samples)
            # [1, 1, 1] -> [1, 1, 1, 0, 0]
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate :
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:     # (2, 1000)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    # 같은 이름 다른 확장자 명의 파일에 접근하고 싶은데, 어떻게 접근해야하지?
    # csv 파일 하나를 더 만들까?
    def _get_audio_sample_path(self, index):
        # fold = f"fold{self.annotations.iloc[index, 2]}"
        path = os.path.join(self.audio_dir, self.file_dir_csv.iloc[index, 0])
        return path

    def _get_audio_label_path(self, index):
        path = os.path.join(self.annotations_dir, self.file_dir_csv.iloc[index, 1])
        return path

    # label을 어떻게 얻어야 하는거지?
    # label이 frame 정보 하나를 가져오는건가?
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]



if __name__ == "__main__":
    FILE_NAME_DIR = "/Users/valleotb/Desktop/Valleotb/sample_filename/metadata.csv"
    ANNOTATION_DIR = "/Users/valleotb/Desktop/Valleotb/sample_metadata"
    AUDIO_DIR = "/Users/valleotb/Desktop/Valleotb/sample_save"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"using device: {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )
    # ms = mel_spectrogram(signal)

    usd = Librispeech_Dataset(ANNOTATION_DIR,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device
                            )
    print(f"There are {len(usd)} samples in the dataset.")

    # signal, label = usd[0]
