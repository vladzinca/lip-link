"""
Implement the dataset used to obtain the data loaders used by the lip reader model.
:cls LipReaderDataset: implement the dataset used to obtain the data loaders used by the lip reader model.
"""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from utils.data_preprocessor import DataPreprocessor


class LipReaderDataset(Dataset):
    """
    Implement the dataset used to obtain the data loaders used by the lip reader model.
    :attr mpg_file_paths: List[str] containing the paths to the MPG files
    :attr data_preprocessor: DataPreprocessor object used to load the data
    :meth __init__(mpg_file_paths): initialize the dataset with the list of MPG files
    :meth __len__(): return the number of items in the dataset
    :meth __getitem__(index): return a single item from the dataset
    """

    def __init__(self, mpg_file_paths: List[str], data_preprocessor: DataPreprocessor = DataPreprocessor()) -> None:
        """
        Initialize the dataset with the list of MPG files.
        :param mpg_file_paths: List[str] containing the paths to the MPG files
        :param data_preprocessor: DataPreprocessor object used to load the data
        :return: None
        """
        self.mpg_file_paths = mpg_file_paths
        self.data_preprocessor = data_preprocessor

    def __len__(self) -> None:
        """
        Return the number of items in the dataset.
        :return: None
        """
        return len(self.mpg_file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a single item from the dataset.
        :param index: int representing the index of the item to return
        :return: Tuple[torch.Tensor, torch.Tensor] representing the item at that index
        """
        # Get the path to the MPG file
        mpg_file_path = self.mpg_file_paths[index]

        # Load the data from the MPG and the align file
        frames, alignments = self.data_preprocessor.load_data(mpg_file_path)

        return frames, alignments
