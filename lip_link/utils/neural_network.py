"""
DocString
"""

import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """
    DocString
    """

    def __init__(self, num_classes: int = 49):
        """
        DocString
        """
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv3d(1, 128, kernel_size=3, padding="same")
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding="same")
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.conv3 = nn.Conv3d(256, 75, kernel_size=3, padding="same")
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        # TimeDistributed layer
        self.flatten = nn.Flatten()

        self.lstm1 = nn.LSTM(75 * 5 * 17, 128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)  # 128 * 2 because of bidirectional
        self.dropout2 = nn.Dropout(0.5)

        self.dense = nn.Linear(256, num_classes)  # 128 * 2 because of bidirectional

    def forward(self, x):
        """
        DocString
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Adjusting the shape for LSTM input
        # We need to collapse the spatial dimensions since we will treat each spatial location
        # as a sequence step for the LSTM
        _, color, depth, height, width = x.shape
        x = self.flatten(x)

        # Then, reshape x to [batch_size, seq_len (D), features]
        x = x.view(-1, depth, color * height * width)

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = self.dense(x)
        return x
