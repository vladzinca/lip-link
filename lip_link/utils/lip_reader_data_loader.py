"""
Load the data for the lip reader model.
:cls LipReaderDataLoader: load the data for the lip reader model
"""

import glob
import os
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from utils.lip_reader_dataset import LipReaderDataset


class LipReaderDataLoader:
    """
    Load the data for the lip reader model.
    :const DEFAULT_MPG_DIR_PATH: str representing the default directory containing the MPG data
    :attr mpg_dir_path: str representing the directory containing the MPG data
    :attr mpg_file_paths: List[str] representing the paths to the MPG files
    :meth __init__(mpg_dir_path): initialize an instance of the LipReaderDataLoader class
    :meth pad_data_from_batch(batch): pad the data from a batch
    :meth __call__(): preprocess data to obtain the training and testing data loaders needed to train and test a
                      PyTorch-based model
    """

    DEFAULT_MPG_DIR_PATH = "./data/s1"

    def __init__(
        self,
        mpg_dir_path: str = DEFAULT_MPG_DIR_PATH,
    ) -> None:
        """
        Initialize an instance of the DataPreprocessor class.
        :param mpg_dir_path: str representing the directory containing the MPG data
        :param char_converter: CharConverter representing the char converter used to convert between characters and
                               their corresponding indices
        :return: None
        """
        # Store the data preprocessor attributes
        self.mpg_dir_path = mpg_dir_path
        self.mpg_file_paths = glob.glob(os.path.join(self.mpg_dir_path, "*.mpg"))

    @staticmethod
    def pad_data_from_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad the data from a batch.
        :param batch: List[Tuple[torch.Tensor, torch.Tensor]] representing the batch
        :return: Tuple[torch.Tensor, torch.Tensor] representing the padded batch
        """
        # Load the data from the specified path
        frames, alignments = zip(*batch)

        # Pad the frames if necessary
        padded_frames = pad_sequence(
            [x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in frames], batch_first=True
        )

        # Pad the alignments if necessary
        padded_alignments = pad_sequence(
            [x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in alignments],
            batch_first=True,
            padding_value=0,
        )

        return padded_frames, padded_alignments

    def __call__(self) -> Tuple[DataLoader, DataLoader]:
        """
        Preprocess data to obtain the training and testing data loaders needed to train and test a PyTorch-based model.
        :return: Tuple[DataLoader, DataLoader] representing the data loaders
        """
        # Create the dataset
        lip_reader_dataset = LipReaderDataset(self.mpg_file_paths)
        dataset_length = len(lip_reader_dataset)

        # Split the dataset into training and testing datasets
        train_dataset_length = int(0.9 * dataset_length)
        test_dataset_length = len(lip_reader_dataset) - train_dataset_length
        train_dataset, test_dataset = random_split(lip_reader_dataset, [train_dataset_length, test_dataset_length])

        # Create the data loaders
        train_data_loader = DataLoader(
            lip_reader_dataset, batch_size=2, shuffle=True, collate_fn=self.pad_data_from_batch
        )
        test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=self.pad_data_from_batch)

        return train_data_loader, test_data_loader
