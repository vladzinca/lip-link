"""
DocString
"""


import random

import imageio
import numpy as np
import torch
from utils.char_converter import CharConverter
from utils.data_pipeliner import DataPipeliner
from utils.neural_network import NeuralNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_pipeliner = DataPipeliner()
train, test = data_pipeliner.pipeline_data()

batch = random.choice(train)
print(batch[0].shape)
print(batch[1].shape)

frames_tensor = batch[0][0]
print(batch[0][0])
print(batch[0][0].shape)

frames_numpy = frames_tensor.squeeze().numpy()
frames_numpy = ((frames_numpy + 1) * 127.5).astype(np.uint8)

imageio.mimsave("./animation.gif", frames_numpy, fps=10)

alignments_tensor = batch[1][0]
print(
    "".join(
        [x.decode("utf-8") for x in data_pipeliner.data_loader.char_converter.convert_idx_to_char(alignments_tensor)]
    )
)

char_converter = CharConverter()
model = NeuralNetwork().to(device)

# Set the model to evaluation mode
model.eval()

inputs, _ = batch
inputs = inputs.permute(0, 2, 1, 3, 4)

# Send inputs to the same device as the model
inputs = inputs.to(device)

# Do not compute gradient since we're only predicting
with torch.no_grad():
    outputs = model(inputs)

# Convert the model predictions (outputs) to class indices
predicted_indices = torch.argmax(outputs, dim=2)

print(
    "".join(
        [x.decode("utf-8") for x in data_pipeliner.data_loader.char_converter.convert_idx_to_char(predicted_indices[1])]
    )
)
