"""
Run the LipLink demo application.
:cls Demo: run the LipLink demo application
"""

import sys
import time
from datetime import datetime
from typing import Tuple, Union

from run_video import RunVideo


class Demo:
    """
    Run the LipLink demo application.
    :meth convert_seconds(seconds): convert the time from seconds to hours, minutes, and seconds
    :meth run_video(video_path): run a video through the LipLink model and return its output
    :meth __call__(): call the object to run the LipLink demo application
    """

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

    def run_video(self, video_path: Union[None, str]) -> str:
        """
        Run a video through the LipLink model and return its output.
        :param video_path: Union[None, str] representing the path to the video file
        :return: str representing the output of the LipLink model
        """
        # Store starting time
        starting_time = time.time()
        print(f'From lip-link-kernel: Process started at "{datetime.fromtimestamp(starting_time)}".', file=sys.stderr)

        # Run the video through the LipLink model and obtain a prediction
        run_video = RunVideo(video_path)
        pred = run_video()
        print(f'Prediction: "{pred}".')

        # Store the ending time
        ending_time = time.time()
        print(f'From lip-link-kernel: Process ended at "{datetime.fromtimestamp(ending_time)}".', file=sys.stderr)

        # Compute elapsed time
        elapsed_time = ending_time - starting_time
        hours, minutes, seconds = self.convert_seconds(int(elapsed_time))
        print(
            f"From lip-link-kernel: Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds.", file=sys.stderr
        )

    def __call__(self) -> None:
        """
        Call the object to run the LipLink demo application.
        :return: None
        """
        # Check if the video path is provided
        if len(sys.argv) != 2:
            print("From lip-link-kernel: Use: python src/demo.py <video_path>.", file=sys.stderr)

        # Run the video through the LipLink model
        else:
            video_path = sys.argv[1]
            self.run_video(video_path)


# Check if the script is run as the main program
if __name__ == "__main__":
    # Initialize an instance of the Demo class and run the demo application
    demo = Demo()
    demo()
