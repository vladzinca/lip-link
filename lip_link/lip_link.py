"""
DocString
"""

from typing import List

import cv2
import torch
from torchvision import transforms


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


class StringLookup:
    """
    DocString
    """

    VOCAB = "abcdefghijklmnopqrstuvwxyz'?!123456789 "
    OOV_TOKEN = ""

    def __init__(self):
        """
        DocString
        """
        # Define the vocabulary
        self.vocabulary = list(self.VOCAB)

        # Add the out-of-vocabulary token. When the model encounters a character
        # that is not in the vocabulary, it will use this token
        self.out_of_vocabulary_token = self.OOV_TOKEN

        self.vocabulary.insert(0, self.out_of_vocabulary_token)

        # Define the functions to convert from characters to indices and vice versa
        self.char_by_index = {char: i for i, char in enumerate(self.vocabulary)}
        self.index_by_char = {i: char.encode() for char, i in self.char_by_index.items()}

    # Define the functions to convert from characters to indices and vice versa
    def char_to_index(self, char):
        """
        DocString
        """
        return self.char_by_index[char]

    def chars_to_indices(self, chars):
        """
        DocString
        """
        return [self.char_by_index[char] for char in chars]

    def index_to_char(self, index):
        """
        DocString
        """
        return self.index_by_char[index]

    def indices_to_chars(self, indices):
        """
        DocString
        """
        return [self.index_by_char[index] for index in indices]

    def char_to_num(self, char):
        """
        DocString
        """
        if isinstance(char, char):
            return self.char_to_index(char)
        if isinstance(char, list):
            return self.chars_to_indices(char)
        raise ValueError("Input must be a character or list of characters.")

    def num_to_char(self, num):
        """
        DocString
        """
        if isinstance(num, int):
            return self.index_to_char(num)
        if isinstance(num, list):
            return self.indices_to_chars(num)
        raise ValueError("Input must be an int or list of ints.")


string_lookup = StringLookup()

# Example of usage:
text = ["v", "l", "a", "d", " ", "z", "i", "n", "c", "a"]
numbers = string_lookup.chars_to_indices(text)
print(numbers)
characters = string_lookup.indices_to_chars(numbers)
print(characters)

text = ["n", "i", "c", "k"]
numbers = string_lookup.chars_to_indices(text)
print(numbers)
characters = string_lookup.indices_to_chars(numbers)
print(characters)

print(len(string_lookup.vocabulary))


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
    indices = string_lookup.chars_to_indices(tokens[1:])
    return torch.tensor(indices, dtype=torch.long)


alignments = load_alignments("data/alignments/s1/bbaf2n.align")
print(alignments)
