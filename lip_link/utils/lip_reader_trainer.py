"""
Train the lip reader model.
:cls LipReaderTrainer: train the lip reader model
"""

import csv
import os
import time
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.char_converter import CharConverter
from utils.lip_reader import LipReader


class LipReaderTrainer:
    """
    Train the lip reader model.
    :const DEFAULT_LEARNING_RATE: int representing the default learning rate
    :const DEFAULT_CHECKPOINT_DIR_PATH: str representing the default path where to save checkpoints
    :const DEFAULT_EPOCHS: int representing the default number of epochs
    :const DEFAULT_EXAMPLES_COUNT: int representing the default number of examples to produce at the end of each epoch
    :attr train_data_loader: DataLoader representing the training data loader
    :attr test_data_loader: DataLoader representing the testing data loader
    :attr model: nn.Module representing the lip reader model
    :attr learning_rate: int representing the learning rate
    :attr checkpoint_dir_path: str representing the path where to save checkpoints
    :attr epochs: int representing the number of epochs
    :attr char_converter: CharConverter representing the object used to convert between characters and indices
    :attr device: torch.device representing the device
    :attr criterion: CTCLoss representing the loss function
    :attr optimizer: torch.optim.Adam representing the optimizer
    :meth __init__(train_data_loader, test_data_loader, model, learning_rate, checkpoint_dir_path, epochs,
                   char_converter): initialize an instance of the LipReaderTrainer class
    :meth set_device(): set the device
    :meth set_hyperparameters(): set the hyperparameters
    :meth convert_seconds(seconds): convert the time from seconds to hours, minutes, and seconds
    :meth scheduler(optimizer, epoch, initial_learning_rate, decay_rate): define and set the learning rate
    :meth save_checkpoint(epoch, checkpoint_file_name): save the current model checkpoint
    :meth __call__(): train the lip reader model
    """

    DEFAULT_LEARNING_RATE = 0.0001
    DEFAULT_CHECKPOINT_DIR_PATH = "./checkpoints/experiment_4"
    DEFAULT_EPOCHS = 150

    def __init__(
        self,
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
        model: nn.Module = LipReader(),
        learning_rate: int = DEFAULT_LEARNING_RATE,
        checkpoint_dir_path: str = DEFAULT_CHECKPOINT_DIR_PATH,
        epochs: int = DEFAULT_EPOCHS,
        char_converter: CharConverter = CharConverter(),
    ) -> None:
        """
        Initialize an instance of the LipReaderTrainer class.
        :param train_data_loader: DataLoader representing the training data loader
        :param test_data_loader: DataLoader representing the testing data loader
        :param model: nn.Module representing the lip reader model
        :param learning_rate: int representing the learning rate
        :param checkpoint_dir_path: str representing the path where to save checkpoints
        :param char_converter: CharConverter representing the char converter used to convert between characters and
                               their corresponding indices
        :return: None
        """
        # Store the lip reader trainer attributes
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.model = model

        self.learning_rate = learning_rate
        self.checkpoint_dir_path = checkpoint_dir_path
        self.epochs = epochs

        self.char_converter = char_converter

        # Set the device and other hyperparameters
        self.set_device()
        self.set_hyperparameters()

    def set_device(self) -> None:
        """
        Set the device.
        :return: None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f'From lip-link-kernel: Device set to "{self.device}".')

    def set_hyperparameters(self) -> None:
        """
        Set the hyperparameters.
        :return: None
        """
        # Set the optimizer as torch.optim.Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Set the loss function as CTCLoss
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

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

    @staticmethod
    def scheduler(optimizer, epoch, initial_learning_rate=0.0001, decay_rate=0.1):
        """
        Define and set the learning rate.
        :param optimizer: torch.optim.Adam representing the optimizer for which to set the learning rate
        :param epoch: int representing the epoch
        :param initial_learning_rate: float representing the initial learning rate
        :param decay_rate: float representing the decay rate
        :return: float representing the learning rate
        """
        if epoch < 30:
            return initial_learning_rate
        else:
            learning_rate = initial_learning_rate * torch.exp(torch.tensor(-decay_rate))
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
            return learning_rate

    def save_checkpoint(self, epoch: int, checkpoint_file_name: str = "checkpoint.pth") -> None:
        """
        Save the current model checkpoint.
        :param epoch: int representing the epoch
        :param checkpoint_file_name: str representing the checkpoint file name
        :return: None
        """
        # Define the path
        path = os.path.join(self.checkpoint_dir_path, checkpoint_file_name)

        # Save the model checkpoint
        torch.save(self.model.state_dict(), path)

    def __call__(self):
        """
        Train the lip reader model.
        :return: None
        """
        # Initialize lists to store losses and learning rates
        train_loss_list = []
        val_loss_list = []
        lr_list = []

        for epoch in range(self.epochs):
            # Store starting time
            starting_time = time.time()

            # Print the current epoch
            print(f"Training epoch {epoch + 1}/{self.epochs}...")

            # Set the model to training mode
            self.model.train()

            # Iterate over the training data loader
            total_train_loss = 0.0
            for frames, alignments in self.train_data_loader:
                # Add a channel dimension at position 1
                frames = frames.unsqueeze(1)

                # Move the frames and alignments to the device
                frames, alignments = frames.to(self.device), alignments.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(frames)
                frame_lengths = torch.full((frames.size(0),), outputs.size(2), dtype=torch.long)
                alignment_lengths = torch.tensor([len(alignment) for alignment in alignments], dtype=torch.long)
                train_loss = self.criterion(
                    outputs.log_softmax(2).permute(1, 0, 2), alignments, frame_lengths, alignment_lengths
                )

                # Backward pass
                train_loss.backward()
                self.optimizer.step()

                # Add the loss to the total training loss
                total_train_loss += train_loss.item()

            # Decode the last outputs that passed through the model during training
            decoded_outputs = torch.argmax(outputs, dim=2)

            # Set the model to evaluation mode
            self.model.eval()

            # Iterate over the testing data loader
            total_val_loss = 0.0
            with torch.no_grad():
                for frames, alignments in self.test_data_loader:
                    # Add a channel dimension at position 1
                    frames = frames.unsqueeze(1)

                    # Move the frames and alignments to the device
                    frames, alignments = frames.to(self.device), alignments.to(self.device)

                    # Calculate the loss
                    outputs = self.model(frames)
                    frame_lengths = torch.full((frames.size(0),), outputs.size(2), dtype=torch.long)
                    alignment_lengths = torch.tensor([len(alignment) for alignment in alignments], dtype=torch.long)
                    val_loss = self.criterion(
                        outputs.log_softmax(2).permute(1, 0, 2), alignments, frame_lengths, alignment_lengths
                    )

                    # Add the loss to the total validation loss
                    total_val_loss += val_loss.item()

            if self.scheduler is not None:
                self.scheduler(self.optimizer, epoch)

            # Store the ending time
            ending_time = time.time()

            # Compute elapsed time
            elapsed_time = ending_time - starting_time
            elapsed_hours, elapsed_minutes, elapsed_seconds = self.convert_seconds(elapsed_time)

            # Calculate the average training and validation losses
            average_train_loss = total_train_loss / len(self.train_data_loader)
            average_val_loss = total_val_loss / len(self.test_data_loader)

            # Print the epoch, training loss, and validation loss
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch took {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}, ",
                f"train_loss: {average_train_loss:.4f} - val_loss: {average_val_loss:.4f} - lr: {current_lr}",
            )

            # Append the losses and learning rate to the lists
            train_loss_list.append(average_train_loss)
            val_loss_list.append(average_val_loss)
            lr_list.append(current_lr)

            # Save a checkpoint
            self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch + 1}.pth")

            # Print an example
            target = "".join([x.decode("utf-8") for x in self.char_converter.convert_idx_to_char(alignments[0])])
            pred = "".join([x.decode("utf-8") for x in self.char_converter.convert_idx_to_char(decoded_outputs[0])])
            print(f"Target: {target}")
            print(f"Prediction: {pred}")
            print("=" * 64)

        # Save the results to a CSV file
        with open("results.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.checkpoint_dir_path, "train_loss"] + train_loss_list)
            writer.writerow([self.checkpoint_dir_path, "val_loss"] + val_loss_list)
            writer.writerow([self.checkpoint_dir_path, "lr"] + lr_list)
