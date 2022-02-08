import torch
import torchaudio

from torch import nn
from torch.utils.data import DataLoader
from custom_dataset import UrbanSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "/Users/valleotb/Desktop/Valleotb/librispeech_metadata/metadata.csv"
AUDIO_DIR = "/Users/valleotb/Desktop/Valleotb/librispeech_processed/"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

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

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device
                            )
    # create a data loader for the train set
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)



    # construct model and assign it to device
    cnn = CNNNetwork().to(device)

    # instantiate loss function + optimiser
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Model Trained and Stored at cnn.pth")