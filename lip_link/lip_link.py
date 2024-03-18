from typing import List

import cv2
import torch
from torchvision import transforms

"""
DocString
"""


def load_mpg_file(mpg_file_path: str) -> List[float]:
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
            transforms.Lambda(lambda x: x.crop((80, 190, 220, 236))),
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
