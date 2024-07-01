"""
Run the LipLink application.
:cls LipLink: run the LipLink application
"""

import sys
import time
from datetime import datetime
from typing import Tuple

from torch.utils.data import DataLoader

from utils.data_fetcher import DataFetcher
from utils.lip_reader_data_loader import LipReaderDataLoader
from utils.lip_reader_tester import LipReaderTester
from utils.lip_reader_trainer import LipReaderTrainer


class LipLink:
    """
    Run the LipLink application.
    :meth convert_seconds(seconds): convert the time from seconds to hours, minutes, and seconds
    :meth fetch(): fetch the data needed by LipLink
    :meth test_lip_reader_data_loader(): test the data preprocessor
    :meth test_lip_reader(): test the lip reader
    :meth test_lip_reader_trainer(): test the lip reader trainer
    :meth __call__(): call the object to run the LipLink application
    """

    def convert_seconds(self, seconds: int) -> Tuple[int, int, int]:
        """
        Convert the time from seconds to hours, minutes, and seconds.
        :param seconds: int representing the time in seconds
        :return: Tuple[int, int, int] representing the time in hours, minutes, and seconds
        """
        # Convert the time from seconds to hours, minutes, and seconds
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        return int(hours), int(minutes), int(seconds)

    def fetch(self) -> None:
        """
        Fetch the data needed by LipLink.
        :return: None
        """
        # Store starting time
        starting_time = time.time()
        print(f'From lip-link-kernel: Process started at "{datetime.fromtimestamp(starting_time)}".', file=sys.stderr)

        # Fetch the data needed by LipLink
        data_fetcher = DataFetcher()
        data_fetcher()

        # Store the ending time
        ending_time = time.time()
        print(f'From lip-link-kernel: Process ended at "{datetime.fromtimestamp(ending_time)}".', file=sys.stderr)

        # Compute elapsed time
        elapsed_time = ending_time - starting_time
        elapsed_hours, elapsed_minutes, elapsed_seconds = self.convert_seconds(elapsed_time)
        print(
            f"From lip-link-kernel: Whole process took {elapsed_hours:02d}:{elapsed_minutes:02d}:",
            f"{elapsed_seconds:02d}.",
            file=sys.stderr,
        )

    def train(self, train_data_loader: DataLoader, test_data_loader: DataLoader) -> None:
        """
        Train the LipLink model.
        :param train_data_loader: DataLoader representing the training data loader
        :param test_data_loader: DataLoader representing the testing data loader
        :return: None
        """
        # Store starting time
        starting_time = time.time()
        print(f'From lip-link-kernel: Process started at "{datetime.fromtimestamp(starting_time)}".', file=sys.stderr)

        # Initialize the lip reader trainer
        lip_reader_trainer = LipReaderTrainer(train_data_loader, test_data_loader)

        # Train the model
        lip_reader_trainer()

        # Store the ending time
        ending_time = time.time()
        print(f'From lip-link-kernel: Process ended at "{datetime.fromtimestamp(ending_time)}".', file=sys.stderr)

        # Compute elapsed time
        elapsed_time = ending_time - starting_time
        elapsed_hours, elapsed_minutes, elapsed_seconds = self.convert_seconds(elapsed_time)
        print(
            f"From lip-link-kernel: Whole process took {elapsed_hours:02d}:{elapsed_minutes:02d}:",
            f"{elapsed_seconds:02d}.",
            file=sys.stderr,
        )

    def test(self, test_data_loader: DataLoader) -> None:
        """
        Test the LipLink model.
        :param test_data_loader: DataLoader representing the testing data loader
        :return: None
        """
        # Store starting time
        starting_time = time.time()
        print(f'From lip-link-kernel: Process started at "{datetime.fromtimestamp(starting_time)}".', file=sys.stderr)

        # Initialize the lip reader tester
        lip_reader_tester = LipReaderTester(test_data_loader)

        # Test the model
        lip_reader_tester()

        # Store the ending time
        ending_time = time.time()
        print(f'From lip-link-kernel: Process ended at "{datetime.fromtimestamp(ending_time)}".', file=sys.stderr)

        # Compute elapsed time
        elapsed_time = ending_time - starting_time
        elapsed_hours, elapsed_minutes, elapsed_seconds = self.convert_seconds(elapsed_time)
        print(
            f"From lip-link-kernel: Whole process took {elapsed_hours:02d}:{elapsed_minutes:02d}:"
            f"{elapsed_seconds:02d}.",
            file=sys.stderr,
        )

    def __call__(self) -> None:
        """
        Call the object to run the LipLink application.
        :return: None
        """
        if "--fetch" in sys.argv:
            # Fetch the data needed by LipLink
            print("From lip-link-kernel: Initializing fetching...", file=sys.stderr)
            self.fetch()

        # Preprocess the data and obtain the data loaders
        lip_reader_data_loader = LipReaderDataLoader()
        train_data_loader, validation_data_loader, test_data_loader = lip_reader_data_loader()

        if "--train" in sys.argv:
            # Train the LipLink model
            print("From lip-link-kernel: Initializing training...", file=sys.stderr)
            self.train(train_data_loader, validation_data_loader)

        if "--test" in sys.argv:
            # Test the LipLink model
            print("From lip-link-kernel: Initializing testing...", file=sys.stderr)
            self.test(test_data_loader)

        if "--fetch" not in sys.argv and "--train" not in sys.argv and "--test" not in sys.argv:
            print("From lip-link-kernel: Use: python src/main.py [--fetch / --train / --test].", file=sys.stderr)


# Check if the script is run as the main program
if __name__ == "__main__":
    # Initialize an instance of the LipLink class and run the application
    lip_link = LipLink()
    lip_link()
