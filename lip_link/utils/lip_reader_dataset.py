"""
Implement the dataset used to obtain the data loaders used by the lip reader model.
:cls LipReaderDataset: implement the dataset used to obtain the data loaders used by the lip reader model.
"""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class LipReaderDataset(Dataset):
    """
    Implement the dataset used to obtain the data loaders used by the lip reader model.
    :attr data: List[Tuple[torch.Tensor, torch.Tensor]] representing the preprocessed data
    :meth __init__(data): initialize the dataset with the preprocessed data
    :meth __len__(): return the number of items in the dataset
    :meth __getitem__(index): return a single item from the dataset
    """

    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        Initialize the dataset with the preprocessed data.
        :param data: List[Tuple[torch.Tensor, torch.Tensor]], where each tuple contains frames and alignments
        :return: None
        """
        self.data = data

    def __len__(self) -> None:
        """
        Return the number of items in the dataset.
        :return: None
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a single item from the dataset.
        :param index: int representing the index of the item to return
        :return: Tuple[torch.Tensor, torch.Tensor] representing the item at that index
        """
        frames, alignments = self.data[index]

        return frames, alignments
