"""
Test the LipLink model.
:cls LipReaderTester: test the LipLink model
"""

import sys

import Levenshtein as lv
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.char_converter import CharConverter
from utils.data_decoder import DataDecoder
from utils.lip_reader import LipReader


class LipReaderTester:
    """
    Test the LipLink model.
    :const DEFAULT_DATA_POINTS: int representing the default number of data points in the testing dataset
    :const DEFAULT_WORD_COUNT_PER_DATA_POINT: int representing the default number of words per data point
    :attr test_data_loader: DataLoader representing the testing data loader
    :attr model: nn.Module representing the lip reader model
    :attr char_converter: CharConverter representing the object used to convert between characters and indices
    :attr data_decoder: DataDecoder representing the object used to decode the model outputs to text and
                        correct the decoded texts
    :attr data_points: int representing the number of data points in the testing dataset
    :attr word_count_per_data_point: int representing the number of words per data point
    :attr device: torch.device representing the device used for testing
    :meth __init__(test_data_loader, model, char_converter, data_decoder, data_points, word_count_per_data_point):
                   initialize an instance of the LipReaderTester class
    :meth set_device(): set the device
    :meth load_model(): load the model
    :meth compute_word_error(target, pred): compute the number of word errors between the target and predicted texts
    :meth compute_levenshtein_distance(target, pred): compute the Levenshtein distance between the target and predicted
                                                      texts
    :meth compute_accuracy(total_levenshtein_distance): compute the accuracy
    :meth __call__(): test the LipLink model
    """

    DEFAULT_DATA_POINTS = 100
    DEFAULT_WORD_COUNT_PER_DATA_POINT = 6

    def __init__(
        self,
        test_data_loader: DataLoader,
        model: nn.Module = LipReader(),
        char_converter: CharConverter = CharConverter(),
        data_decoder: DataDecoder = DataDecoder(),
        data_points: int = DEFAULT_DATA_POINTS,
        word_count_per_data_point: int = DEFAULT_WORD_COUNT_PER_DATA_POINT,
    ) -> None:
        """
        Initialize an instance of the LipReaderTester class.
        :param test_data_loader: DataLoader representing the testing data loader
        :param model: nn.Module representing the lip reader model
        :param char_converter: CharConverter representing the object used to convert between characters and indices
        :param data_decoder: DataDecoder representing the object used to decode the model outputs to text and
                             correct the decoded texts
        :param data_points: int representing the number of data points in the testing dataset
        :param word_count_per_data_point: int representing the number of words per data point
        :return: None
        """
        # Store the lip reader tester attributes
        self.test_data_loader = test_data_loader
        self.model = model
        self.char_converter = char_converter
        self.data_decoder = data_decoder
        self.data_points = data_points
        self.word_count_per_data_point = word_count_per_data_point

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
            print("From lip-link-kernel: The testing will use the GPU.", file=sys.stderr)
        else:
            self.model.load_state_dict(
                torch.load("./checkpoints/experiment_2/checkpoint_epoch_100.pth", map_location=torch.device("cpu"))
            )
            print("From lip-link-kernel: The testing will use the CPU.", file=sys.stderr)

    @staticmethod
    def compute_word_error(target: str, pred: str) -> int:
        """
        Compute the number of word errors between the target and predicted texts.
        :param target: str representing the target text
        :param pred: str representing the predicted text
        :return: int representing the number of word errors
        """
        # Split the texts into sets of words
        target_words = set(target.split())
        pred_words = set(pred.split())

        # Find the words that are in pred but are not in target
        error_words = pred_words - target_words

        # Compute the word error
        word_error = len(error_words)

        return word_error

    @staticmethod
    def compute_levenshtein_distance(target: str, pred: str) -> int:
        """
        Compute the Levenshtein distance between the target and predicted texts.
        :param target: str representing the target text
        :param pred: str representing the predicted text
        :return: int representing the Levenshtein distance
        """
        levenshtein_distance = lv.distance(target, pred)

        return levenshtein_distance

    @staticmethod
    def compute_accuracy(total_levenshtein_distance: float) -> float:
        """
        Compute the accuracy as the opposite of the total Levenshtein distance.
        :param total_levenshtein_distance: float representing the total Levenshtein distance
        :return: float representing the accuracy
        """
        accuracy = 100 - total_levenshtein_distance

        return accuracy

    def __call__(self) -> None:
        """
        Test the LipLink model.
        :return: None
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Initialize the variabls needed to compute the accuracy
        total_word_count = self.data_points * self.word_count_per_data_point
        total_character_count = 0
        total_word_error = 0
        total_levenshtein_distance = 0

        # Iterate over the testing data loader
        data_point_position = 0
        with torch.no_grad():
            for frames, alignments in self.test_data_loader:
                # Add a channel dimension at position 1
                frames = frames.unsqueeze(1)

                # Move the frames and alignments to the device
                frames, alignments = frames.to(self.device), alignments.to(self.device)

                # Obtain the model results
                result = self.model(frames)

                # Decode and correct the model results
                decoded_results = self.data_decoder.decode_result(result)
                corrected_results = [
                    self.data_decoder.correct_result(decoded_result) for decoded_result in decoded_results
                ]

                # Retrieve the targets from the alignments
                targets = []
                for alignment in alignments:
                    target = "".join([x.decode("utf-8") for x in self.char_converter.convert_idx_to_char(alignment)])
                    targets.append(target)

                for target, pred in zip(targets, corrected_results):
                    # Print the current data position in the dataset
                    data_point_position += 1
                    print(f"Predicting for data point {data_point_position}/{self.data_points}:")

                    # Print the targets and the predictions
                    print(f'Target: "{target}".')
                    print(f'Prediction: "{pred}".')
                    print("=" * 64)

                    # Compute the character count
                    character_count = len(target)
                    total_character_count += character_count

                    # Compute word error
                    word_error = self.compute_word_error(target, pred)
                    total_word_error += word_error

                    # Compute Levenshtein distance
                    levenshtein_distance = self.compute_levenshtein_distance(target, pred)
                    total_levenshtein_distance += levenshtein_distance

        # Compute the word error rate as the ratio of total word errors to the total number of words in the targets
        word_error_rate = round(total_word_error / total_word_count * 100, 2)
        average_word_error_rate = total_word_error / 100

        # Compute the ratio of total Levenshtein distance to the total number of characters in the targets
        general_levenshtein_distance = round(total_levenshtein_distance / total_character_count * 100, 2)
        average_levenshtein_distance = total_levenshtein_distance / 100

        # Compute the accuracy
        accuracy = round(self.compute_accuracy(general_levenshtein_distance), 2)

        # Print the results
        print("Results:")
        print(f"Average word count, per sentence: {self.word_count_per_data_point}.")
        print(f"Average character count, per sentence: {total_character_count / self.data_points}.")
        print(f"Word error rate, total: {word_error_rate}%.")
        print(f"Word error rate, per sentence: {average_word_error_rate}.")
        print(f"Levenshtein distance, total: {general_levenshtein_distance}%.")
        print(f"Levenshtein distance, characters per sentence: {average_levenshtein_distance}.")
        print(f"Accuracy, total: {accuracy}%.")
        print("=" * 64)
