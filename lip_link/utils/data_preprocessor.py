"""
Preprocess the data and obtain the data loaders needed by LipLink.
:cls DataPreprocessor: preprocess the data and obtain the data loaders needed by LipLink
"""

import os
from typing import Tuple

import cv2
import torch
from utils.char_converter import CharConverter


class DataPreprocessor:
    """
    Preprocess the data needed by LipLink.
    :attr char_converter: CharConverter representing the object used to convert between characters and indices
    :meth __init__(mouth_crop, char_converter): initialize an instance of the DataPreprocessor class
    :meth load_mpg_file(mpg_file_path): load the data from an MPG file
    :meth load_align_file(align_file_path): load the data from an align file
    :meth load_data(mpg_file_path): load the data from an MPG file and an align file
    """

    def __init__(
        self,
        char_converter: CharConverter = CharConverter(),
    ) -> None:
        """
        Initialize an instance of the DataPreprocessor class.
        :param char_converter: CharConverter representing the char converter used to convert between characters and
                               their corresponding indices
        :return: None
        """
        # Store the data preprocessor attributes
        self.char_converter = char_converter

    def load_mpg_file(self, mpg_file_path: str) -> torch.Tensor:
        """
        Load the data from an MPG file.
        :param mpg_file_path: str representing the path to the MPG file
        :return: torch.Tensor representing the data from the MPG file
        """
        # Initialize the frame list
        frames = []

        # Open the video file
        capture = cv2.VideoCapture(mpg_file_path)

        # Read the frames and apply the transform
        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[190:236, 80:220]
            frames.append(frame)

        # Normalize the frames to a mean equal to 0 and a standard deviation equal to 1
        frames = torch.tensor(frames, dtype=torch.float32)
        mean = frames.mean()
        standard_deviation = frames.std()
        frames = (frames - mean) / standard_deviation

        return frames

    def load_align_file(self, align_file_path: str) -> torch.Tensor:
        """
        Load the data from an align file.
        :param align_file_path: str representing the path to the align file
        :return: torch.Tensor representing the data from the align file
        """
        # Open the align file
        with open(align_file_path, "r", encoding="utf-8") as align_file:
            lines = align_file.readlines()

        # Extract the tokens from the align file
        tokens = []
        for line in lines:
            line = line.strip().split()

            # Check if the line is not empty or a silence token
            if line and line[2] != "sil":
                tokens.extend([" "] + list(line[2]))

        # Convert the tokens to indices
        indices = self.char_converter.convert_char_to_idx(tokens[1:])

        return indices

    def load_data(self, mpg_file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load the data from an MPG file and an align file.
        :param mpg_file_path: str representing the path to the MPG file
        :return: Tuple[torch.Tensor, torch.Tensor] representing the frames and alignments data from the MPG file and the
                 corresponding align file
        """
        # Assume the directory separator for the current system is '/'
        dir_separator = "/"

        # Change it to '\\' if the current system uses Windows
        if os.name == "nt":
            dir_separator = "//"

        # Get the file name from the path
        file_name = mpg_file_path.split(dir_separator)[-1].split(".")[0]

        # Get the paths to the MPG and align files
        mpg_file_path = os.path.join("data", "s1", f"{file_name}.mpg")
        align_file_path = os.path.join("data", "alignments", "s1", f"{file_name}.align")

        # Load the data from the MPG and align files
        frames = self.load_mpg_file(mpg_file_path)
        alignments = self.load_align_file(align_file_path)

        return frames, alignments
