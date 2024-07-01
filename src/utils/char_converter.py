"""
Convert between characters and their corresponding indices.
:cls CharConverter: convert between characters and their corresponding indices
"""

from typing import List, Union

import torch


class CharConverter:
    """
    Convert between characters and their corresponding indices.
    :const DEFAULT_VOCABULARY: str representing the default vocabulary
    :const DEFAULT_OOV_TOKEN: str representing the default out-of-vocabulary token
    :attr vocabulary: List[str] representing the characters that the model can recognize
    :attr oov_token: str representing the out-of-vocabulary token that the model uses to represent characters that are
                     not in the vocabulary
    :attr char_to_idx: Dict[bytes, int] representing the mapping between characters and their corresponding indices
    :attr idx_to_char: Dict[int, bytes] representing the mapping between indices and their corresponding characters
    :meth __init__(vocabulary, oov_token): initialize an instance of the CharConverter class
    :meth get_vocabulary(): get the vocabulary used by the CharConverter
    :meth get_vocabulary_size(): get the size of the vocabulary used by the CharConverter
    :meth convert_char_to_idx(char): convert a character or a list of characters to their corresponding indices
    :meth convert_idx_to_char(idx): convert an index or a list of indices to their corresponding characters
    """

    DEFAULT_VOCABULARY = "abcdefghijklmnopqrstuvwxyz'?!123456789 "
    DEFAULT_OOV_TOKEN = ""

    def __init__(self, vocabulary: str = DEFAULT_VOCABULARY, oov_token: str = DEFAULT_OOV_TOKEN) -> None:
        """
        Initialize an instance of the CharConverter class.
        :param vocabulary: str representing the characters that the model can recognize
        :param oov_token: str representing the out-of-vocabulary token that the model uses to represent characters that
                          are not in the vocabulary
        """
        # Store the vocabulary and the out-of-vocabulary token
        self.vocabulary = list(vocabulary)
        self.oov_token = oov_token

        # Add the out-of-vocabulary token to the vocabulary and encode the letters
        self.vocabulary.insert(0, self.oov_token)

        # Create two dictionaries used to map characters to indices and vice versa
        self.char_to_idx = {char.encode(): i for i, char in enumerate(self.vocabulary)}
        self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}

    def get_vocabulary(self) -> List[str]:
        """
        Get the vocabulary used by the CharConverter.
        :return: List[str] representing the vocabulary
        """
        return self.vocabulary

    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary used by the CharConverter.
        :return: int representing the size of the vocabulary
        """
        return len(self.vocabulary)

    def convert_char_to_idx(self, char: Union[str, List[str], bytes, List[bytes]]) -> torch.Tensor:
        """
        Convert a character or a list of characters to their corresponding indices.
        :param char: Union[str, List[str], bytes, List[bytes]] representing a character or a list of characters
        :return: torch.Tensor representing the indices corresponding to the characters
        """
        if isinstance(char, bytes):
            return torch.tensor(self.char_to_idx[char], dtype=torch.int32)
        if isinstance(char, str):
            return torch.tensor(self.char_to_idx[char.encode()], dtype=torch.int32)
        # Convert a list of characters to the list of their corresponding indices
        idx = []
        for character in char:
            if isinstance(character, bytes):
                idx.append(self.char_to_idx[character])
            elif isinstance(character, str):
                idx.append(self.char_to_idx[character.encode()])
        return torch.tensor(idx, dtype=torch.int32)

    def convert_idx_to_char(self, idx: Union[int, List[int], torch.Tensor]) -> Union[bytes, List[bytes]]:
        """
        Convert an index or a list of indices to their corresponding characters.
        :param idx: Union[int, List[int], torch.Tensor] representing an index or a list of indices
        :return: Union[bytes, List[bytes]] representing the characters corresponding to the indices
        """
        if isinstance(idx, int):
            return self.idx_to_char[idx]
        if isinstance(idx, list):
            return [self.idx_to_char[index] for index in idx]
        # Convert a tensor of indices to the list of their corresponding characters
        if idx.dim() == 0:
            return self.idx_to_char[idx.item()]
        return [self.idx_to_char[index] for index in idx.tolist()]
