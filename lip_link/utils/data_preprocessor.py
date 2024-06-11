"""
Preprocess the data and obtain the data loaders needed by LipLink.
:cls DataPreprocessor: preprocess the data and obtain the data loaders needed by LipLink
"""

import glob
import os
import sys
from typing import Tuple

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.char_converter import CharConverter
from utils.lip_reader_dataset import LipReaderDataset


class DataPreprocessor:
    """
    Preprocess the data needed by LipLink.
    :const DEFAULT_MOUTH_CROP: Tuple[int, int, int, int] representing the default crop to apply to the mouth
    :const DEFAULT_MPG_DIR_PATH: str representing the default directory containing the MPG data
    :attr mouth_crop: Tuple[int, int, int, int] representing the crop to apply to the mouth
    :attr mpg_dir_path: str representing the directory containing the MPG data
    :attr mpg_file_paths: List[str] representing the paths to the MPG files
    :attr char_converter: CharConverter representing the object used to convert between characters and indices
    :meth __init__(mouth_crop, mpg_dir_path, char_converter): initialize an instance of the DataPreprocessor class
    :meth load_mpg_file(mpg_file_path): load the data from an MPG file
    :meth load_align_file(align_file_path): load the data from an align file
    :meth load_data(): load the data from an MPG file and an align file
    :meth pad_data(): pad the data from an MPG file and an align file
    :meth preprocess_data(): preprocess data to obtain the training and testing data loaders needed to train and test a
                             PyTorch-based model
    """

    DEFAULT_MOUTH_CROP = (80, 190, 220, 236)
    DEFAULT_MPG_DIR_PATH = "./data/s1"

    def __init__(
        self,
        mouth_crop: Tuple[int, int, int, int] = DEFAULT_MOUTH_CROP,
        mpg_dir_path: str = DEFAULT_MPG_DIR_PATH,
        char_converter: CharConverter = CharConverter(),
    ) -> None:
        """
        Initialize an instance of the DataPreprocessor class.
        :param mpg_dir_path: str representing the directory containing the MPG data
        :param char_converter: CharConverter representing the char converter used to convert between characters and
                               their corresponding indices
        :return: None
        """
        # Store the data preprocessor attributes
        self.mouth_crop = mouth_crop
        self.mpg_dir_path = mpg_dir_path
        self.mpg_file_paths = glob.glob(os.path.join(self.mpg_dir_path, "*.mpg"))
        self.char_converter = char_converter

    def load_mpg_file(self, mpg_file_path: str) -> torch.Tensor:
        """
        Load the data from an MPG file.
        :param mpg_file_path: str representing the path to the MPG file
        :return: torch.Tensor representing the data from the MPG file
        """
        # Initialize the frame list
        frames = []

        # Compose the transform to be applied on each frame
        transform_frame = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: x.crop(self.mouth_crop)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).byte()),
            ]
        )

        # Open the video file
        capture = cv2.VideoCapture(mpg_file_path)

        # Read the frames and apply the transform
        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break
            frame = transform_frame(frame)
            frames.append(frame)

        # Normalize the frames to a mean equal to 0 and a standard deviation equal to 1
        frames = torch.stack(frames).to(torch.float32)
        mean = torch.round(torch.mean(frames))
        standard_deviation = torch.std(frames)
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

        return indices.clone().detach().long()

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

    def pad_data(self, mpg_file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad the data from an MPG file and an align file.
        :param mpg_file_path: str representing the path to the MPG file
        :return: Tuple[torch.Tensor, torch.Tensor] representing the frames and alignments data from the MPG file and the
                 corresponding align file
        """
        # Load the data from the specified path
        frames, alignments = self.load_data(mpg_file_path)

        # Pad the frames if necessary to ensure there are 75 frames
        frames_to_pad = 75 - frames.size(0)
        if frames_to_pad > 0:
            frames = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, 0, frames_to_pad, 0), "constant", 0)

        # Pad the alignments if necessary to ensure they have 40 characters
        chars_to_pad = 40 - alignments.size(0)
        if chars_to_pad > 0:
            alignments = torch.nn.functional.pad(alignments, (0, chars_to_pad), "constant", 0)

        return frames, alignments

    def preprocess_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Preprocess data to obtain the training and testing data loaders needed to train and test a PyTorch-based model.
        :return: Tuple[DataLoader, DataLoader] representing the data loaders
        """
        # Initialize the pipelined data list and count
        pipelined_data = []
        pipelined_data_count = 0

        # Iterate over the MPG file paths
        for mpg_file_path in self.mpg_file_paths:
            # Load and pad data for a video and its corresponding align file
            frames, alignments = self.pad_data(mpg_file_path)

            # Permute the frames to have the frame dimension last
            frames = frames.permute(0, 2, 3, 1)
            frames = frames.permute(3, 1, 2, 0)

            # Pipeline data
            pipelined_data.append((frames, alignments))

            # Increment the count and calculate the total percentage
            pipelined_data_count += 1
            pipelined_data_percentage = pipelined_data_count * 100 / len(self.mpg_file_paths)

            if pipelined_data_percentage % 25 == 0:
                print(f"From lip-link-kernel: Pipelining data at {int(pipelined_data_percentage)}%.", file=sys.stderr)

        # Calculate how many data points are in the training set
        train_data_count = int(0.9 * len(pipelined_data))

        # Split the data into train and test
        train_data = pipelined_data[:train_data_count]
        test_data = pipelined_data[train_data_count:]

        # Add the data to the training and testing sets
        train_dataset = LipReaderDataset(train_data)
        test_dataset = LipReaderDataset(test_data)

        # Create the training and testing data loaders
        train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        return train_data_loader, test_data_loader

    def __call__(self) -> Tuple[DataLoader, DataLoader]:
        """
        Call the object to preprocess the data and obtain the data loaders needed by LipLink.
        :return: Tuple[DataLoader, DataLoader] representing the data loaders
        """
        # Preprocess data
        train_data_loader, test_data_loader = self.preprocess_data()

        return train_data_loader, test_data_loader
