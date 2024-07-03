"""
Run a video through the LipLink model and return its output.
:cls RunVideo: run a video through the LipLink model and return its output
"""

import sys

import torch
from torch import nn

from utils.char_converter import CharConverter
from utils.data_decoder import DataDecoder
from utils.data_preprocessor import DataPreprocessor
from utils.lip_reader import LipReader


class RunVideo:
    """
    Run a video through the LipLink model and return its output.
    :attr video_path: str representing the path to the video file
    :attr data_preprocessor: DataPreprocessor representing the object used to preprocess the data
    :attr model: nn.Module representing the lip reader model
    :attr char_converter: CharConverter representing the object used to convert between characters and indices
    :attr data_decoder: DataDecoder representing the object used to decode the model outputs to text and
                        correct the decoded texts
    :meth __init__(video_path, data_preprocessor, model, char_converter, data_decoder): initialize an instance of the
                                                                                        RunVideo class
    :meth set_device(): set the device
    :meth load_model(): load the model
    """

    def __init__(
        self,
        video_path: str,
        data_preprocessor: DataPreprocessor = DataPreprocessor(),
        model: nn.Module = LipReader(),
        char_converter: CharConverter = CharConverter(),
        data_decoder: DataDecoder = DataDecoder(),
    ) -> None:
        """
        Initialize an instance of the RunVideo class.
        :param video_path: str representing the path to the video file
        :param data_preprocessor: DataPreprocessor representing the object used to preprocess the data
        :param model: nn.Module representing the lip reader model
        :param char_converter: CharConverter representing the object used to convert between characters and indices
        :param data_decoder: DataDecoder representing the object used to decode the model outputs to text and
                             correct the decoded texts
        :return: None
        """
        # Store the video path
        self.video_path = video_path
        self.data_preprocessor = data_preprocessor
        self.model = model
        self.char_converter = char_converter
        self.data_decoder = data_decoder

        # Set the device and load the model
        self.set_device()
        self.load_model()

    def set_device(self) -> None:
        """
        Set the device.
        :return: None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f'From lip-link-kernel: Device set to "{self.device}".', file=sys.stderr)

    def load_model(self) -> None:
        """
        Load the model.
        :return: None
        """
        # Load the model
        if torch.cuda.is_available():
            self.model.load_state_dict(
                torch.load("./checkpoints/experiment_2/checkpoint_epoch_100.pth", map_location=torch.device("cuda"))
            )
            print("From lip-link-kernel: Model loaded on the GPU.", file=sys.stderr)
        else:
            self.model.load_state_dict(
                torch.load("./checkpoints/experiment_2/checkpoint_epoch_100.pth", map_location=torch.device("cpu"))
            )
            print("From lip-link-kernel: Model loaded on the CPU.", file=sys.stderr)

    def __call__(self) -> str:
        """
        Run a video through the LipLink model and return its output.
        :return: str representing the output of the LipLink model for the given video
        """
        # Set the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Load and preprocess the video before feeding it to the model
            frames = self.data_preprocessor.load_mpg_file(self.video_path)

            # Add a batch dimension at position 0 and a channel dimension at position 1
            frames = frames.unsqueeze(0)
            frames = frames.unsqueeze(1)

            # Move the frames to the device
            frames = frames.to(self.device)

            # Obtain the model results
            result = self.model(frames)

            # Decode and correct the model results
            decoded_results = self.data_decoder.decode_result(result)
            corrected_results = [self.data_decoder.correct_result(decoded_result) for decoded_result in decoded_results]

            # Obtain the final prediction and return it
            pred = corrected_results[0]

            return pred
