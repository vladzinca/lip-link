"""
DocString
"""

from typing import List

import cv2
import matplotlib
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from lip_link.utils.char_converter import CharConverter

char_converter = CharConverter()

matplotlib.use("Agg")


class DataProcessing:
    """
    DocString
    """

    MOUTH_CROP = (80, 190, 220, 236)

    def load_mpg_file(self, mpg_file_path: str) -> List[float]:
        """
        DocString
        """
        # Initialize the frame list
        frames = []

        # Compose the transform to be applied on each frame
        transform_frame = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Lambda(lambda x: x.crop(self.MOUTH_CROP)),
                transforms.ToTensor(),
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

        # Normalize the frames
        frames = torch.stack(frames, dim=0)
        mean = torch.mean(frames)
        standard_deviation = torch.std(frames)
        frames = (frames - mean) / standard_deviation

        return frames


def load_alignments(path: str) -> torch.Tensor:
    """
    DocString
    """
    with open(path, "r") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.strip().split()
        if line and line[2] != "sil":
            tokens.extend([" "] + list(line[2]))
    indices = char_converter.chars_to_indices(tokens[1:])
    return torch.tensor(indices, dtype=torch.long)


alignments = load_alignments("data/alignments/s1/bbaf2n.align")
print(alignments)

print(char_converter.convert_idx_to_char(alignments))

data_processing = DataProcessing()
frames = data_processing.load_mpg_file("data/s1/bbaf2n.mpg")

frame_to_display = frames[100].squeeze()

plt.imshow(frame_to_display, cmap="gray")

plt.savefig("frame.png")
