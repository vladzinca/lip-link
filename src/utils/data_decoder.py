"""
Decode the model outputs to text and correct the decoded texts.
:cls DataDecoder: decode the model outputs to text and correct the decoded texts
"""

from typing import List

import torch
import torch.nn.functional as F
from textblob import TextBlob

from utils.char_converter import CharConverter


class DataDecoder:
    """
    Decode the model outputs to text and correct the decoded texts.
    :attr char_converter: CharConverter representing the object used to convert between characters and indices
    :meth __init__(char_converter): initialize an instance of the DataDecoder class
    :meth decode_result(data): decode the model outputs given in data to text
    :meth correct_result(data): correct the spelling of the text given in data
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

    def decode_result(self, data: torch.Tensor) -> List[str]:
        """
        Decode the model outputs given in data to text.
        :param data: torch.Tensor representing the model outputs
        :return: List[str] representing the decoded texts
        """
        # Apply softmax to convert logits to probabilities and get the largest probability indices
        probabilities = F.softmax(data, dim=2)
        max_indices = torch.argmax(probabilities, dim=2)

        # Decode each prediction in the batch
        decoded_results = []
        for i in range(max_indices.size(0)):
            decoded_result = "".join(
                [x.decode("utf-8") for x in self.char_converter.convert_idx_to_char(max_indices[i])]
            )
            decoded_results.append(decoded_result)

        return decoded_results

    def correct_result(self, data: str) -> str:
        """
        Correct the spelling of the text given in data.
        :param data: str representing the text to correct
        :return: str representing the corrected text
        """
        # Correct spelling using TextBlob
        text_blob = TextBlob(data)
        corrected_text = str(text_blob.correct())

        # Remove additional whitespace
        corrected_text_words = corrected_text.split()
        corrected_results = ""
        for word in corrected_text_words:
            if len(corrected_results) + len(word) + 1 > 31:
                break
            corrected_results += word + " "

        # Trim result to the first 6 words if it exceeds 6 words
        corrected_results_words = corrected_results.strip().split()
        if len(corrected_results_words) > 6:
            corrected_results_words = corrected_results_words[:6]
        corrected_results = " ".join(corrected_results_words)

        return corrected_results
