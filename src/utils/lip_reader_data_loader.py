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
    :const PERSISTENT_TEST_FILE_PATHS: List[str] representing the paths to the persistent test files
    :attr persistent_test: bool representing whether to use the persistent test data that was not used to train the
                           model in experiment 2
    :attr mpg_dir_path: str representing the directory containing the MPG data
    :attr mpg_file_paths: List[str] representing the paths to the MPG files
    :attr persistent_test_file_paths: List[str] representing the paths to the persistent test files
    :meth __init__(persistent_test, mpg_dir_path): initialize an instance of the LipReaderDataLoader class
    :meth pad_data_from_batch(batch): pad the data from a batch
    :meth __call__(): preprocess data to obtain the training and testing data loaders needed to train and test a
                      PyTorch-based model
    """

    DEFAULT_MPG_DIR_PATH = "./data/s1"
    PERSISTENT_TEST_FILE_PATHS = [
        "./data/s1/swih7s.mpg",
        "./data/s1/srbu9a.mpg",
        "./data/s1/pbbi9s.mpg",
        "./data/s1/pgad9s.mpg",  # 0 - 3
        "./data/s1/bgwi1a.mpg",
        "./data/s1/lwaz3a.mpg",
        "./data/s1/lwal5a.mpg",
        "./data/s1/bwitzp.mpg",  # 4 - 7
        "./data/s1/bwwu1s.mpg",
        "./data/s1/sraozp.mpg",
        "./data/s1/swwp4p.mpg",
        "./data/s1/bwig2p.mpg",  # 8 - 11
        "./data/s1/brwg6n.mpg",
        "./data/s1/lgiz2n.mpg",
        "./data/s1/sbwb3s.mpg",
        "./data/s1/sbbh5a.mpg",  # 12 - 15
        "./data/s1/lwws5s.mpg",
        "./data/s1/lriq9a.mpg",
        "./data/s1/pwij4p.mpg",
        "./data/s1/srwo9a.mpg",  # 16 - 19
        "./data/s1/bbbz9s.mpg",
        "./data/s1/bbws8n.mpg",
        "./data/s1/prbj6p.mpg",
        "./data/s1/prap5s.mpg",  # 20 - 23
        "./data/s1/bgan6p.mpg",
        "./data/s1/lriy3a.mpg",
        "./data/s1/pgwlzn.mpg",
        "./data/s1/pbav4p.mpg",  # 24 - 27
        "./data/s1/bbir8p.mpg",
        "./data/s1/pwaj6n.mpg",
        "./data/s1/pric5a.mpg",
        "./data/s1/sgwqzp.mpg",  # 28 - 31
        "./data/s1/pbbjzp.mpg",
        "./data/s1/srao1a.mpg",
        "./data/s1/sbbb1a.mpg",
        "./data/s1/lbbe1s.mpg",  # 32 - 35
        "./data/s1/bgwizp.mpg",
        "./data/s1/swwi8n.mpg",
        "./data/s1/bbaf4p.mpg",
        "./data/s1/swbpzp.mpg",  # 36 - 39
        "./data/s1/bgbu3s.mpg",
        "./data/s1/pgby4n.mpg",
        "./data/s1/brbt1s.mpg",
        "./data/s1/lwal3s.mpg",  # 40 - 43
        "./data/s1/srwi5a.mpg",
        "./data/s1/lwaz1s.mpg",
        "./data/s1/pgay2p.mpg",
        "./data/s1/lgaf6p.mpg",  # 44 - 47
        "./data/s1/pgwe7s.mpg",
        "./data/s1/sbaa5s.mpg",
        "./data/s1/swao5s.mpg",
        "./data/s1/lwwf7s.mpg",  # 48 - 51
        "./data/s1/srbb5s.mpg",
        "./data/s1/sbit2n.mpg",
        "./data/s1/prbdzn.mpg",
        "./data/s1/sbwb2n.mpg",  # 52 - 55
        "./data/s1/bbaf3s.mpg",
        "./data/s1/srwb9s.mpg",
        "./data/s1/pbap1a.mpg",
        "./data/s1/pgwr5s.mpg",  # 56 - 59
        "./data/s1/lrie1a.mpg",
        "./data/s1/sgio7s.mpg",
        "./data/s1/lbbq8n.mpg",
        "./data/s1/lrbl3a.mpg",  # 60 - 63
        "./data/s1/prwd6p.mpg",
        "./data/s1/srwb8n.mpg",
        "./data/s1/pbib6n.mpg",
        "./data/s1/sgav6p.mpg",  # 64 - 67
        "./data/s1/sbbbzp.mpg",
        "./data/s1/bgbb3a.mpg",
        "./data/s1/lwwf9a.mpg",
        "./data/s1/bgwb7a.mpg",  # 68 - 71
        "./data/s1/swbi4n.mpg",
        "./data/s1/sbwo3a.mpg",
        "./data/s1/brwa4p.mpg",
        "./data/s1/sgac4p.mpg",  # 72 - 75
        "./data/s1/sgwx3s.mpg",
        "./data/s1/swwv6n.mpg",
        "./data/s1/lgil5s.mpg",
        "./data/s1/lbwr5a.mpg",  # 76 - 79
        "./data/s1/lbad9a.mpg",
        "./data/s1/lbij7s.mpg",
        "./data/s1/lbakzn.mpg",
        "./data/s1/pgby5s.mpg",  # 80 - 83
        "./data/s1/bwit1a.mpg",
        "./data/s1/bras6n.mpg",
        "./data/s1/lgwm9a.mpg",
        "./data/s1/pbwj5a.mpg",  # 84 - 87
        "./data/s1/swbczn.mpg",
        "./data/s1/prbqzp.mpg",
        "./data/s1/lgbs6n.mpg",
        "./data/s1/pwwy4p.mpg",  # 88 - 91
        "./data/s1/pbio4n.mpg",
        "./data/s1/lrak8p.mpg",
        "./data/s1/lwilzp.mpg",
        "./data/s1/prac6n.mpg",  # 92 - 95
        "./data/s1/sgib9s.mpg",
        "./data/s1/sbwo2p.mpg",
        "./data/s1/bbal7s.mpg",
        "./data/s1/pgij9s.mpg",  # 96 - 99
    ]

    def __init__(
        self,
        persistent_test: bool = True,
        mpg_dir_path: str = DEFAULT_MPG_DIR_PATH,
    ) -> None:
        """
        Initialize an instance of the DataPreprocessor class.
        :param persistent_test: bool representing whether to use the persistent test data that was not used to train
                                the model in experiment 2
        :param mpg_dir_path: str representing the directory containing the MPG data
        :param char_converter: CharConverter representing the char converter used to convert between characters and
                               their corresponding indices
        :return: None
        """
        # Store the data preprocessor attributes
        self.persistent_test = persistent_test
        self.mpg_dir_path = mpg_dir_path
        self.mpg_file_paths = glob.glob(os.path.join(self.mpg_dir_path, "*.mpg"))

        if self.persistent_test:
            self.persistent_test_file_paths = self.PERSISTENT_TEST_FILE_PATHS

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

    def __call__(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Preprocess data to obtain the training and testing data loaders needed to train and test a PyTorch-based model.
        :return: Tuple[DataLoader, DataLoader] representing the data loaders
        """
        # Create the dataset
        lip_reader_dataset = LipReaderDataset(self.mpg_file_paths)
        dataset_length = len(lip_reader_dataset)

        # Split the dataset into training and testing datasets
        train_dataset_length = int(0.9 * dataset_length)
        validation_dataset_length = len(lip_reader_dataset) - train_dataset_length
        train_dataset, validation_dataset = random_split(
            lip_reader_dataset, [train_dataset_length, validation_dataset_length]
        )

        # Override the test dataset if using the persistent test data
        if self.persistent_test:
            test_dataset = LipReaderDataset(self.persistent_test_file_paths)
        else:
            test_dataset = validation_dataset

        # Create the data loaders
        train_data_loader = DataLoader(
            lip_reader_dataset, batch_size=2, shuffle=True, collate_fn=self.pad_data_from_batch
        )
        validation_data_loader = DataLoader(
            validation_dataset, batch_size=2, shuffle=False, collate_fn=self.pad_data_from_batch
        )
        test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=self.pad_data_from_batch)

        return train_data_loader, validation_data_loader, test_data_loader
