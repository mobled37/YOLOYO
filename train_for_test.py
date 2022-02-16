import torch
import torchaudio

from torch import nn
from torch.utils.data import DataLoader
from custom_dataset_fixed2 import PSAD_Dataset
from resnet1d.cnn1d import CNN

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "/Users/valleotb/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "/Users/valleotb/Downloads/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):

    for inputs, targets in data_loader:
        inputs, targets = inputs, targets

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-------------------")
    print('Finished Training')

if __name__ == "__main__":
    BATCH_SIZE = 128
    EPOCHS = 1
    LEARNING_RATE = 0.001

    FILENAME_DIR = '/Users/valleotb/Desktop/Valleotb/sample_filename/metadata.csv'
    AUDIO_DIR = '/Users/valleotb/Desktop/Valleotb/sample_save'
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000

    if torch.cuda.is_available():
        device = "cuda"

    else:
        device = "cpu"
    print(f'Using {device} device')

    # instatiationg our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )
    # ms = mel_spectrogram(signal)

    usd = PSAD_Dataset(
        audio_folder_dir=AUDIO_DIR,
        file_name_dir=FILENAME_DIR,
        device=device
    )
    # create a data loader for the train set
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)



    # construct model and assign it to device
    cnn = CNN(
        in_channels=1,
        out_channels=1,
        n_len_seg=256,
        n_classes=1,
        device=device
    ).to(device)

    # instantiate loss function + optimiser
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Model Trained and Stored at cnn.pth")