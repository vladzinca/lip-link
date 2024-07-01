"""
Implement the neural network model for lip reading.
:cls LipReader: implement the neural network model for lip reading
"""

import torch.nn as nn
import torch.nn.functional as F


class LipReader(nn.Module):
    """
    Implement the neural network model for lip reading.
    :attr conv3d_1: nn.Conv3d representing the first 3D convolutional layer
    :attr pool_1: nn.MaxPool3d representing the first 3D max pooling layer
    :attr conv3d_2: nn.Conv3d representing the second 3D convolutional layer
    :attr pool_2: nn.MaxPool3d representing the second 3D max pooling layer
    :attr conv3d_3: nn.Conv3d representing the third 3D convolutional layer
    :attr pool_3: nn.MaxPool3d representing the third 3D max pooling layer
    :attr gru_1: nn.GRU representing the first GRU layer
    :attr dropout_1: nn.Dropout representing the first dropout layer
    :attr gru_2: nn.GRU representing the second GRU layer
    :attr dropout_2: nn.Dropout representing the second dropout layer
    :attr fc: nn.Linear representing the fully-connected layer
    :meth __init__(num_classes): initialize an instance of the LipReader class
    :meth forward(x): implement the forward pass of the LipReader model
    """

    def __init__(self, num_classes: int = 40):
        """
        Initialize an instance of the LipReader class.
        :param num_classes: int representing the size of the vocabulary
        :return: None
        """
        # Call the constructor of the parent class
        super(LipReader, self).__init__()

        # Store the layers of the LipReader model
        self.conv3d_1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)
        self.pool_1 = nn.MaxPool3d((1, 2, 2))
        self.conv3d_2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool_2 = nn.MaxPool3d((1, 2, 2))
        self.conv3d_3 = nn.Conv3d(256, 75, kernel_size=3, padding=1)
        self.pool_3 = nn.MaxPool3d((1, 2, 2))

        self.gru_1 = nn.GRU(6375, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout_1 = nn.Dropout(0.5)
        self.gru_2 = nn.GRU(256, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout_2 = nn.Dropout(0.5)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Implement the forward pass of the LipReader model.
        :param x: torch.Tensor representing the tensor to pass through the LipReader model
        :return: torch.Tensor representing the output of the LipReader model
        """
        # Convolutional layers
        x = F.relu(self.conv3d_1(x))
        x = self.pool_1(x)
        x = F.relu(self.conv3d_2(x))
        x = self.pool_2(x)
        x = F.relu(self.conv3d_3(x))
        x = self.pool_3(x)

        # Prepare for GRU layers
        x = x.view(x.size(0), x.size(2), -1)

        # GRU layers
        x, _ = self.gru_1(x)
        x = self.dropout_1(x)
        x, _ = self.gru_2(x)
        x = self.dropout_2(x)

        # Fully-connected layer
        x = self.fc(x)

        return x
