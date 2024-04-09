"""
DocString
"""

import random

import imageio
import numpy as np
from utils.data_pipeliner import DataPipeliner

# TEST_PATH = "./data/s1/bgah2p.mpg"

# from utils.data_loader import DataLoader

# data_loader = DataLoader()
# data_loader.load_mpg_file(TEST_PATH)


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
