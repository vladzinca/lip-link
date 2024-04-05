"""
Load the data needed by LipLink.
:cls DataLoader: load the data needed by LipLink
"""

import os
import sys
from typing import Tuple
from zipfile import ZipFile

import cv2
import requests
import torch
from torchvision import transforms
from utils.char_converter import CharConverter


class DataLoader:
    """
    Load the data needed by LipLink.
    :const DEFAULT_DATA_URL: str representing the default URL to download the data from
    :const DEFAULT_DATA_DIR_NAME: str representing the default directory to store the data in
    :const DEFAULT_DATA_FILE_NAME: str representing the default name of the data file
    :const DEFAULT_DATA_FILE_SIZE: int representing the default size in bytes of the data file
    :const DEFAULT_MOUTH_CROP: Tuple[int, int, int, int] representing the default crop to apply to the mouth
    :attr data_url: str representing the URL to download the data from
    :attr data_dir_path: str representing the directory to store the data in
    :attr data_file_path: str representing the path to store the data at
    :attr data_file_name: str representing the name of the data file
    :attr data_file_size: int representing the size in bytes of the data file
    :attr mouth_crop: Tuple[int, int, int, int] representing the crop to apply to the mouth
    :attr char_converter: CharConverter representing the object used to convert between characters and indices
    :meth __init__(data_url, data_dir_name, data_file_name, data_file_size): initialize an instance of the DataLoader
                                                                             class
    :meth get_data_dir_path(data_dir_name): get the path to the directory where to store data
    :meth download_data(): download the data needed by LipLink
    :meth extract_data(): extract the data needed by LipLink
    :meth get_data(): download and extract the data needed by LipLink
    :meth load_mpg_file(mpg_file_path): load the data from an MPG file
    :meth load_align_file(align_file_path): load the data from an align file
    :meth load_data(): load the data needed by LipLink
    """

    DEFAULT_DATA_URL = "https://zenodo.org/api/records/3625687/files-archive"
    DEFAULT_DATA_DIR_NAME = "data"
    DEFAULT_DATA_FILE_NAME = "3625687.zip"
    DEFAULT_DATA_FILE_SIZE = 16211532898
    DEFAULT_MOUTH_CROP = (80, 190, 220, 236)

    def __init__(
        self,
        data_url: str = DEFAULT_DATA_URL,
        data_dir_name: str = DEFAULT_DATA_DIR_NAME,
        data_file_name: str = DEFAULT_DATA_FILE_NAME,
        data_file_size: int = DEFAULT_DATA_FILE_SIZE,
        mouth_crop: Tuple[int, int, int, int] = DEFAULT_MOUTH_CROP,
    ) -> None:
        """
        Initialize an instance of the DataLoader class.
        :param data_url: str representing the URL to download the data from
        :param data_dir_name: str representing the name of the data directory
        :param data_file_name: str representing the name of the data file
        :param data_file_size: int representing the size in bytes of the data file
        :param mouth_crop: Tuple[int, int, int, int] representing the crop to apply to the mouth
        """
        # Store the data attributes
        self.data_url = data_url
        self.data_dir_path = self.get_data_dir_path(data_dir_name)
        self.data_file_path = os.path.join(self.data_dir_path, data_file_name)
        self.data_file_name = data_file_name
        self.data_file_size = data_file_size

        # Store the mouth crop
        self.mouth_crop = mouth_crop

        # Initialize the CharConverter
        self.char_converter = CharConverter()

    def get_data_dir_path(self, data_dir_name: str) -> str:
        """
        Get the path to the directory where to store data.
        :param data_dir_name: str representing the name of the data directory
        :return: str representing the path to the data directory
        """
        # Get the parent path of the data directory
        data_dir_parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Get the path to the data directory
        data_dir_path = os.path.join(data_dir_parent_path, data_dir_name)

        return data_dir_path

    def download_data(self) -> None:
        """
        Download the data needed by LipLink.
        :return: None
        """
        # Create the data directory if it does not exist
        os.makedirs(self.data_dir_path, exist_ok=True)

        while True:
            # Download the file
            response = requests.get(self.data_url, stream=True)
            response.raise_for_status()

            # Write the file to the download path in chunks
            with open(self.data_file_path, "wb") as data_file:
                for chunk in response.iter_content(chunk_size=8192):
                    data_file.write(chunk)

            # Check if the downloaded file matches the expected size
            if os.path.getsize(self.data_file_path) == self.data_file_size:
                print(
                    f"From lip-link-kernel: Successfully downloaded {self.data_file_name} "
                    "file that matches expected size.",
                    file=sys.stderr,
                )
                break
            print(
                f"From lip-link-kernel: Failed to download {self.data_file_name} file. "
                f"Expected {self.data_file_size} bytes, but got {os.path.getsize(self.data_file_path)} instead.\n"
                "From lip-link-kernel: Retrying download...",
                file=sys.stderr,
            )

    def extract_data(self) -> None:
        """
        Extract the data needed by LipLink.
        :return: None
        """
        # Unzip the initial data zip file
        with ZipFile(self.data_file_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir_path)

        # Delete the initial data zip file and other unnecessary files
        os.remove(self.data_file_path)
        os.remove(os.path.join(self.data_dir_path, "audio_25k.zip"))
        os.remove(os.path.join(self.data_dir_path, "jasagrid.pdf"))

        # Unzip all the zip files left in data
        for file_name in os.listdir(self.data_dir_path):
            if file_name.endswith(".zip"):
                file_path = os.path.join(self.data_dir_path, file_name)

                with ZipFile(file_path, "r") as zip_ref:
                    # Get a list of the ZipFile contents without the __MACOSX directory
                    members = [m for m in zip_ref.namelist() if not m.startswith("__MACOSX")]

                    # Extract only the filtered members
                    zip_ref.extractall(self.data_dir_path, members=members)

                # Delete the zip file after extraction
                os.remove(file_path)

        # Print a success message
        print(f"From lip-link-kernel: Successfully extracted data from {self.data_file_name} file.", file=sys.stderr)

    def get_data(self) -> None:
        """
        Download and extract the data needed by LipLink.
        :return: None
        """
        # Download the data
        print("From lip-link-kernel: Downloading data...", file=sys.stderr)
        self.download_data()

        # Extract the data
        print(f"From lip-link-kernel: Extracting data from {self.data_file_name} file...", file=sys.stderr)
        self.extract_data()

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
                transforms.Lambda(lambda x: x.crop(self.DEFAULT_MOUTH_CROP)),
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
        Load the data needed by LipLink.
        :param mpg_file_path: str representing the path to the MPG file
        :return: Tuple[torch.Tensor, torch.Tensor] representing the frames and
                 alignments data needed by LipLink
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
