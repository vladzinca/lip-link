"""
Use the DataLoader to pipeline the data needed for training.
:cls DataPipeliner: pipeline the data needed for training
"""

import glob
import os
import random
from typing import List, Tuple

import torch
from utils.data_loader import DataLoader


class DataPipeliner:
    """
    Pipeline the data needed for training.
    :const DEFAULT_MPG_DIR_PATH: str representing the default directory containing the MPG data
    :attr mpg_dir_path: str representing the directory containing the MPG data
    :attr mpg_file_paths: List[str] representing the paths to the MPG files
    :attr data_loader: DataLoader representing the data loader used to load the data
    :meth __init__(data_loader): initialize an instance of the DataPipeliner class
    :meth shuffle_data(): shuffle the data needed for training
    :meth pad_data(mpg_file_path): pad the data from the specified path
    :meth pipeline_data(): pipeline the data needed for training
    """

    DEFAULT_MPG_DIR_PATH = "./data/s1"

    def __init__(self, mpg_dir_path: str = DEFAULT_MPG_DIR_PATH, data_loader: DataLoader = DataLoader()) -> None:
        """
        Initialize an instance of the DataPipeliner class.
        :param mpg_dir_path: str representing the directory containing the MPG data
        :param data_loader: DataLoader representing the data loader used to load the data
        :return: None
        """
        # Store the data pipelining attributes
        self.mpg_dir_path = mpg_dir_path
        self.mpg_file_paths = glob.glob(os.path.join(self.mpg_dir_path, "*.mpg"))

        # Initialize the DataLoader
        self.data_loader = data_loader

    def shuffle_data(self) -> None:
        """
        Shuffle the data needed for training.
        :return: None
        """
        random.shuffle(self.mpg_file_paths)

    def pad_data(self, mpg_file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad the data from the specified path.
        :param mpg_file_path: str representing the path to the MPG file
        :return: None
        """
        # Load the data from the specified path
        frames, alignments = self.data_loader.load_data(mpg_file_path)

        # Pad the frames if necessary to ensure there are 75 frames
        frames_to_pad = 75 - frames.size(0)
        if frames_to_pad > 0:
            frames = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, 0, frames_to_pad, 0), "constant", 0)

        # Pad the alignments if necessary to ensure they have 48 characters
        chars_to_pad = 48 - alignments.size(0)
        if chars_to_pad > 0:
            alignments = torch.nn.functional.pad(alignments, (0, chars_to_pad), "constant", 0)

        return frames, alignments

    def pipeline_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pipeline the data needed for training.
        :return: List[Tuple[torch.Tensor, torch.Tensor]] representing the data batches
                 needed for training
        """
        # Shuffle the data
        self.shuffle_data()

        # Initialize the data batch list
        batches = []

        count = 0

        # Iterate over the MPG file paths in pairs
        for i in range(0, len(self.mpg_file_paths), 2):
            # Load and pad data for two consecutive videos
            frames_1, alignments_1 = self.pad_data(self.mpg_file_paths[i])
            frames_2, alignments_2 = self.pad_data(self.mpg_file_paths[i + 1])

            # Stack along a new dimension to create a batch
            batched_frames = torch.stack([frames_1, frames_2], dim=0)
            batched_alignments = torch.stack([alignments_1, alignments_2], dim=0)

            # Add the batch to the list of batches
            batches.append((batched_frames, batched_alignments))

            count += 1

            print("Pipelining data: ", end="")
            print(count / 5, end="")
            print("%")

        train = batches[:450]
        test = batches[450:]

        return train, test
