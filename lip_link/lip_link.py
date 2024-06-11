"""
Run the LipLink application.
:cls LipLink: run the LipLink application
"""

import time
from datetime import datetime
from typing import Tuple

import torch
from utils.data_preprocessor import DataPreprocessor
from utils.lip_reader import LipReader
from utils.lip_reader_trainer import LipReaderTrainer


class LipLink:
    """
    Run the LipLink application.
    :meth __init__(): initialize an instance of the LipLink class
    :meth set_device(): set the device
    :meth convert_seconds(seconds): convert the time from seconds to hours, minutes, and seconds
    :meth test_data_preprocessor(): test the data preprocessor
    :meth test_lip_reader(): test the lip reader
    :meth test_lip_reader_trainer(): test the lip reader trainer
    :meth __call__(): call the object to run the LipLink application
    """

    def __init__(self) -> None:
        """
        Initialize an instance of the LipLink class.
        :return: None
        """
        self.set_device()

    def set_device(self) -> None:
        """
        Set the device.
        :return: None
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'From lip-link-kernel: Device set to "{self.device}".')

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

    def test_data_preprocessor(self) -> None:
        """
        Test the data preprocessor.
        :return: None
        """
        # Initialize the data preprocessor
        data_preprocessor = DataPreprocessor()

        # Preprocess the data
        train_data_loader, _ = data_preprocessor()

        # Get the frames and alignments from a batch of the train data loader
        train_iter = iter(train_data_loader)
        batch = next(train_iter)
        frames, alignments = batch

        # Print the shape of the first video in the batch
        print(f"Shape of video frames in the batch: {frames[0].shape}.")

        # Print the shape of the first video's corresponding alignments
        print(f"Shape of alignments in the batch: {alignments[0].shape}.")

        # Decode and print the text of the first alignment in the batch
        decoded_text = "".join(
            [x.decode("utf-8") for x in data_preprocessor.char_converter.convert_idx_to_char(alignments[0])]
        )
        print(f"Decoded alignments text corresponding to the first video in the batch: {decoded_text}.")

    def test_lip_reader(self) -> None:
        """
        Test the lip reader neural network model by running a batch through it and decoding the output.
        :return: None
        """
        # Initialize the data preprocessor
        data_preprocessor = DataPreprocessor()

        # Preprocess the data
        train_data_loader, _ = data_preprocessor()

        # Get the frames and alignments from a batch of the train data loader
        train_iter = iter(train_data_loader)
        batch = next(train_iter)
        frames, alignments = batch

        # Initialize the LipReader model and set it to evaluation mode
        lip_reader = LipReader().to(self.device)
        lip_reader.eval()

        # Move the batch to the same device as the model
        frames = frames.to(self.device)

        # Run the batch through the LipReader model
        with torch.no_grad():
            outputs = lip_reader(frames)

        # Decode the outputs
        _, predicted_indices = torch.max(outputs, dim=2)
        predicted_texts = [
            "".join([x.decode("utf-8") for x in data_preprocessor.char_converter.convert_idx_to_char(idx)])
            for idx in predicted_indices
        ]

        # Decode the alignments
        decoded_alignments = [
            "".join([x.decode("utf-8") for x in data_preprocessor.char_converter.convert_idx_to_char(alignment)])
            for alignment in alignments
        ]

        # Print the decoded texts
        for predicted_text, decoded_alignment in zip(predicted_texts, decoded_alignments):
            print(f"Decoded predicted text from LipReader: {predicted_text}.")
            print(f"Decoded actual text corresponding to the video: {decoded_alignment}.")

    def test_lip_reader_trainer(self) -> None:
        """
        Test the lip reader trainer.
        :return: None
        """
        # Store starting time
        starting_time = time.time()
        print(f'From lip-link-kernel: Process started at "{datetime.fromtimestamp(starting_time)}".')

        # Preprocess the data and obtain the data loaders
        data_preprocessor = DataPreprocessor()
        train_data_loader, test_data_loader = data_preprocessor()

        # Initialize the lip reader model
        model = LipReader()

        # Initialize the lip reader trainer
        lip_reader_trainer = LipReaderTrainer(model, train_data_loader, test_data_loader)

        # Train the model
        lip_reader_trainer()

        # Store the ending time
        ending_time = time.time()
        print(f'From lip-link-kernel: Process ended at "{datetime.fromtimestamp(ending_time)}".')

        # Compute elapsed time
        elapsed_time = ending_time - starting_time
        elapsed_hours, elapsed_minutes, elapsed_seconds = self.convert_seconds(elapsed_time)
        print(
            f"From lip-link-kernel: Whole process took {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}."
        )

    def __call__(self) -> None:
        """
        Call the object to run the LipLink application.
        :return: None
        """
        self.test_lip_reader_trainer()


# Check if the script is run as the main program
if __name__ == "__main__":
    # Initialize an instance of the LipLink class and run the application
    lip_link = LipLink()
    lip_link()
