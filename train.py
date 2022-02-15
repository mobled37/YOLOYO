import torch
import torchaudio
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from custom_dataset import Librispeech_Dataset
from cnn import CNNNetwork
from resnet1d.resnet1d import ResNet1D
from custom_dataset_fixed2 import PSAD_Dataset
from resnet1d.resnet1d import MyDataset
from resnet1d.util import read_data_physionet_4

BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.001

FILENAME_DIR = '/Users/valleotb/Desktop/Valleotb/sample_filename/metadata.csv'
AUDIO_DIR = '/Users/valleotb/Desktop/Valleotb/sample_save'

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
    FILENAME_DIR = '/Users/valleotb/Desktop/Valleotb/sample_filename/metadata.csv'
    AUDIO_DIR = '/Users/valleotb/Desktop/Valleotb/sample_save'

    BATCH_SIZE = 128
    EPOCHS = 1
    LEARNING_RATE = 0.001

    if torch.cuda.is_available():
        device = "cuda"

    else:
        device = "cpu"
    print(f'Using {device} device')


    pd = PSAD_Dataset(
        audio_folder_dir=AUDIO_DIR,
        file_name_dir=FILENAME_DIR,
        device=device
    )

    train_data_loader = DataLoader(pd, batch_size=BATCH_SIZE)

    # construct model and assign it to device
    # cnn = CNNNetwork().to(device)
    resnet = ResNet1D(
        in_channels=1,
        base_filters=64,    # 64 for ResNet1D
        kernel_size=16,
        stride=2,
        groups=32,
        n_block=48,  # resnet을 얼마나 쌓을 건지
        n_classes=1
    )
    resnet.to(device)
    summary(resnet, device=device)

    ''
    # instantiate loss function + optimiser
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(resnet.parameters(),
                                 lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=10)

    # train model
    # train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    # train(resnet, train_data_loader, loss_fn, optimiser, device, EPOCHS)

#     torch.save(cnn.state_dict(), "feedforwardnet.pth")
    # print("Model Trained and Stored at cnn.pth")

    n_epoch = 50
    step = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        resnet.train()
        prog_iter = tqdm(train_data_loader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = resnet(input_x)
            loss = loss_fn(pred, input_y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            step += 1

        scheduler.step(_)

        # test
        resnet.eval()
        prog_iter_test = tqdm(train_data_loader, desc="Testing", leave=False)