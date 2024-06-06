"""
Implement the neural network model for lip reading.
:cls LipReader: implement the neural network model for lip reading
"""

import torch
import torch.nn.functional as F
from torch import nn
from utils.char_converter import CharConverter


class LipReader(nn.Module):
    """
    Implement the neural network model for lip reading.
    :attr conv3d_1: nn.Conv3d representing the first 3D convolutional layer
    :attr conv3d_2: nn.Conv3d representing the second 3D convolutional layer
    :attr conv3d_3: nn.Conv3d representing the third 3D convolutional layer
    :attr maxpool3d: nn.MaxPool3d representing the 3D max pooling layer
    :attr lstm_1: nn.LSTM representing the first LSTM layer
    :attr lstm_2: nn.LSTM representing the second LSTM layer
    :attr dropout: nn.Dropout representing the dropout layer
    :attr fully_connected: nn.Linear representing the fully-connected layer
    :meth __init__(vocabulary_size): initialize an instance of the LipReader class
    :meth forward(input_data): implement the forward pass of the LipReader model
    """

    def __init__(self, vocabulary_size: int = CharConverter().get_vocabulary_size()) -> None:
        """
        Initialize an instance of the LipReader class.
        :param vocabulary_size: int representing the size of the vocabulary
        :return: None
        """
        # Call the constructor of the parent class
        super().__init__()

        # Store the layers of the LipReader model
        self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=128, kernel_size=3, padding="same")
        self.conv3d_2 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding="same")
        self.conv3d_3 = nn.Conv3d(in_channels=256, out_channels=75, kernel_size=3, padding="same")

        self.maxpool3d = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.lstm_1 = nn.LSTM(input_size=5 * 17 * 75, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(p=0.5)

        self.fully_connected = nn.Linear(in_features=256, out_features=vocabulary_size)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the LipReader model.
        :param input_data: torch.Tensor representing the input data
        :return: torch.Tensor representing the output of the LipReader model
        """
        # Convolutional layers
        input_data = F.relu(self.conv3d_1(input_data))
        input_data = self.maxpool3d(input_data)
        input_data = F.relu(self.conv3d_2(input_data))
        input_data = self.maxpool3d(input_data)
        input_data = F.relu(self.conv3d_3(input_data))
        input_data = self.maxpool3d(input_data)

        # Prepare for LSTM layers
        input_data = input_data.permute(0, 4, 2, 3, 1)
        input_data = input_data.reshape(input_data.size(0), input_data.size(1), -1)

        # LSTM layers
        input_data, _ = self.lstm_1(input_data)
        input_data = self.dropout(input_data)
        input_data, _ = self.lstm_2(input_data)
        input_data = self.dropout(input_data)

        # Fully-connected layer
        input_data = self.fully_connected(input_data)

        return input_data
