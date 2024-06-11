"""
Train the lip reader model.
:cls LipReaderTrainer: train the lip reader model
"""

import math
import os
import time
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from utils.char_converter import CharConverter


class LipReaderTrainer:
    """
    Train the lip reader model.
    :const DEFAULT_LEARNING_RATE: int representing the default learning rate
    :const DEFAULT_CHECKPOINT_DIR_PATH: str representing the default path where to save checkpoints
    :const DEFAULT_EPOCHS: int representing the default number of epochs
    :const DEFAULT_EXAMPLES_COUNT: int representing the default number of examples to produce at the end of each epoch
    :attr model: nn.Module representing the lip reader model
    :attr train_data_loader: DataLoader representing the training data loader
    :attr test_data_loader: DataLoader representing the testing data loader
    :attr learning_rate: int representing the learning rate
    :attr checkpoint_dir_path: str representing the path where to save checkpoints
    :attr char_converter: CharConverter representing the object used to convert between characters and indices
    :attr device: torch.device representing the device
    :attr criterion: CTCLoss representing the loss function
    :attr optimizer: torch.optim.Adam representing the optimizer
    :attr scheduler: LambdaLR representing the learning rate scheduler
    :meth __init__(model, train_data_loader, test_data_loader, learning_rate): initialize an instance of the
                                                                               LipReaderTrainer class
    :meth set_device(): set the device
    :meth lr_lambda(epoch): define the learning rate lambda function
    :meth set_hyperparameters(): set the hyperparameters
    :meth convert_seconds(seconds): convert the time from seconds to hours, minutes, and seconds
    :meth compute_ctc_loss(y_pred, y_target): compute the Connectionist temporal classification loss
    :meth train(): train the lip reader model for an epoch
    :meth evaluate(): evaluate the lip reader model
    :meth save_checkpoint(epoch, checkpoint_file_name): save the current model checkpoint
    :meth decode_alignments(alignments, is_prediction): decode the alignments
    :meth produce_examples(examples_count): produce examples
    :meth __call__(epochs): train the lip reader model
    """

    DEFAULT_LEARNING_RATE = 0.0001
    DEFAULT_CHECKPOINT_DIR_PATH = "./checkpoints/experiment_0"
    DEFAULT_EPOCHS = 100
    DEFAULT_EXAMPLES_COUNT = 1

    def __init__(
        self,
        model: nn.Module,
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
        learning_rate: int = DEFAULT_LEARNING_RATE,
        checkpoint_dir_path: str = DEFAULT_CHECKPOINT_DIR_PATH,
        char_converter: CharConverter = CharConverter(),
    ) -> None:
        """
        Initialize an instance of the LipReaderTrainer class.
        :param model: nn.Module representing the lip reader model
        :param train_data_loader: DataLoader representing the training data loader
        :param test_data_loader: DataLoader representing the testing data loader
        :param learning_rate: int representing the learning rate
        :param checkpoint_dir_path: str representing the path where to save checkpoints
        :param char_converter: CharConverter representing the char converter used to convert between characters and
                               their corresponding indices
        :return: None
        """
        # Store the lip reader trainer attributes
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.learning_rate = learning_rate
        self.checkpoint_dir_path = checkpoint_dir_path
        self.char_converter = char_converter

        # Set the device and other hyperparameters
        self.set_device()
        self.set_hyperparameters()

    def set_device(self) -> None:
        """
        Set the device.
        :return: None
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f'From lip-link-kernel: Device set to "{self.device}".')

    def lr_lambda(self, epoch: int) -> float:
        """
        Define the learning rate lambda function.
        :param epoch: int representing the epoch
        :return: float representing the learning rate
        """
        if epoch < 30:
            return 1.0
        return pow(math.exp(-0.1), (epoch - 30))

    def set_hyperparameters(self) -> None:
        """
        Set the hyperparameters.
        :return: None
        """
        # Set the loss function as CTCLoss
        self.criterion = CTCLoss(blank=0)

        # Set the optimizer as torch.optim.Adam
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Set the learning rate scheduler as LambdaLR
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

    def convert_seconds(self, seconds: int) -> Tuple[int, int, int]:
        """
        Convert the time from seconds to hours, minutes, and seconds.
        :param seconds: int representing the time in seconds
        :return: Tuple[int, int, int] representing the time in hours, minutes, and seconds
        """
        # Convert the time from seconds to hours, minutes, and seconds
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        return int(hours), int(minutes), int(seconds)

    def compute_ctc_loss(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Connectionist temporal classification loss.
        :param y_pred: torch.Tensor representing the predicted values
        :param y_target: torch.Tensor representing the target values
        :return: torch.Tensor representing the loss
        """
        # Define input_lengths and target_lengths
        input_lengths = torch.full(size=(y_pred.size(0),), fill_value=y_pred.size(1), dtype=torch.long).to(self.device)
        target_lengths = torch.full(size=(y_target.size(0),), fill_value=y_target.size(1), dtype=torch.long).to(
            self.device
        )

        # y_pred needs to be of shape (time_steps, batch_size, num_classes) for CTCLoss
        y_pred = y_pred.permute(1, 0, 2)

        # Compute the loss
        loss = self.criterion(y_pred, y_target, input_lengths, target_lengths)
        return loss

    def train(self) -> int:
        """
        Train the lip reader model for an epoch.
        :return: int representing the average training loss
        """
        # Set the model to training mode
        self.model.train()

        # Iterate over the training data loader
        total_train_loss = 0
        for frames, alignments in self.train_data_loader:
            # Move the frames and alignments to the device
            frames, alignments = frames.to(self.device), alignments.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(frames)
            train_loss = self.compute_ctc_loss(outputs, alignments)

            # Backward pass
            train_loss.backward()
            self.optimizer.step()

            # Add the loss to the total training loss
            total_train_loss += train_loss.item()

        # Calculate the average training loss
        average_train_loss = total_train_loss / len(self.train_data_loader)
        return average_train_loss

    def evaluate(self) -> int:
        """
        Evaluate the lip reader model.
        :return: int representing the average validation loss
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Iterate over the testing data loader
        total_val_loss = 0
        with torch.no_grad():
            for frames, alignments in self.test_data_loader:
                # Move the frames and alignments to the device
                frames, alignments = frames.to(self.device), alignments.to(self.device)

                # Calculate the loss
                outputs = self.model(frames)
                val_loss = self.compute_ctc_loss(outputs, alignments)

                # Add the loss to the total validation loss
                total_val_loss += val_loss.item()

        # Calculate the average validation loss
        average_val_loss = total_val_loss / len(self.test_data_loader)
        return average_val_loss

    def save_checkpoint(self, epoch: int, checkpoint_file_name: str = "checkpoint.pth") -> None:
        """
        Save the current model checkpoint.
        :param epoch: int representing the epoch
        :param checkpoint_file_name: str representing the checkpoint file name

        """
        # Define the path
        path = os.path.join(self.checkpoint_dir_path, checkpoint_file_name)

        # Save the current model checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def decode_alignments(self, alignments: torch.Tensor, is_prediction: bool = False) -> List[str]:
        """
        Decode the alignments.
        :param alignments: torch.Tensor representing the alignments
        :param is_prediction: bool representing whether the alignments are predictions
        :return: List[str] representing the decoded alignments
        """
        # Apply softmax to the prediction to get the actual predicted alignments
        if is_prediction:
            _, alignments = torch.max(alignments, dim=2)

        # Decode the alignments
        decoded_alignments = [
            "".join([x.decode("utf-8") for x in self.char_converter.convert_idx_to_char(alignment)])
            for alignment in alignments
        ]
        return decoded_alignments

    def produce_examples(self, examples_count: int = DEFAULT_EXAMPLES_COUNT):
        """
        Produce examples.
        :param examples_count: int representing the number of examples to produce
        :return: None
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Get an iterator for the testing data loader
        test_iter = iter(self.test_data_loader)

        # Iterate over the first examples_count pieces of data in testing data loader
        with torch.no_grad():
            for _ in range(examples_count):
                # Get the frames and alignments from a batch
                batch = next(test_iter)
                frames, alignments = batch

                # Move the frames and alignments to the device
                frames, alignments = frames.to(self.device), alignments.to(self.device)

                # Calculate the outputs
                outputs = self.model(frames)

                # Decode the predictions and targets
                y_pred = self.decode_alignments(outputs, is_prediction=True)
                y_target = self.decode_alignments(alignments)

                # Print the produced examples
                for pred, target in zip(y_pred, y_target):
                    print(f"Target: {target}")
                    print(f"Prediction: {pred}")
                    print("=" * 64)

    def __call__(self, epochs: int = DEFAULT_EPOCHS) -> None:
        """
        Train the lip reader model.
        :param epochs: int representing the number of epochs
        :return: None
        """
        for epoch in range(epochs):
            # Store starting time
            starting_time = time.time()

            # Print the current epoch
            print(f"Training epoch {epoch + 1}/{epochs}...")

            # Train the model
            train_loss = self.train()

            # Evaluate the model
            val_loss = self.evaluate()

            # Print the epoch, training loss, and validation loss
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {current_lr:.4f}")

            # Store the ending time
            ending_time = time.time()

            # Compute elapsed time
            elapsed_time = ending_time - starting_time
            elapsed_hours, elapsed_minutes, elapsed_seconds = self.convert_seconds(elapsed_time)
            print(f"Epoch took {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}.")

            # Step the learning rate scheduler
            self.scheduler.step()

            # Save the model checkpoint
            self.save_checkpoint(epoch, checkpoint_file_name=f"checkpoint_epoch_{epoch}.pth")

            # Produce examples
            self.produce_examples()
