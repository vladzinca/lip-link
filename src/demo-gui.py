"""
Run the LipLink GUI demo application.
:cls DemoGUI: run the LipLink GUI demo application
"""

import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk


class DemoGUI:
    """
    Run the LipLink GUI demo application.
    :attr root: tk.Tk representing the root window of the application
    :attr logo: ImageTk.PhotoImage representing the logo image
    :attr logo_label: tk.Label representing the label displaying the logo image
    :attr run_button: tk.Button representing the button to run the video
    :attr output_text: tk.Text representing the text widget to display output
    :attr loading_label: tk.Label representing the label displaying the loading wheel
    :meth __init__(root): initialize an instance of the DemoGUI class
    :meth run_video(): run the video
    :meth process_video(video_path): process the video
    """

    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize an instance of the DemoGUI class.
        :param root: tk.Tk representing the root window of the application
        :return: None
        """
        self.root = root
        self.root.title("LipLink Demo v1.1")

        # Load and display the logo image
        self.logo = Image.open("./assets/logo_400x400.png")
        self.logo = ImageTk.PhotoImage(self.logo)
        self.root.iconphoto(False, self.logo)  # Add logo to start bar
        self.logo_label = tk.Label(self.root, image=self.logo)
        self.logo_label.pack(pady=10)

        # Button to run video
        self.run_button = tk.Button(self.root, text="Run video...", command=self.run_video)
        self.run_button.pack(pady=5)

        # Text widget to display output
        self.output_text = tk.Text(self.root, height=5, width=50)
        self.output_text.pack(pady=10)

        # Loading wheel label
        self.loading_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.loading_label.pack(pady=5)

    def run_video(self) -> None:
        """
        Run the video.
        :return: None
        """
        # Open file dialog to select video file
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mpg")])
        if not video_path:
            return

        # Clear previous output
        self.output_text.delete(1.0, tk.END)

        # Start the processing in a new thread
        threading.Thread(target=self.process_video, args=(video_path,)).start()

    def process_video(self, video_path: str) -> None:
        """
        Process the video.
        :param video_path: str representing the path to the video file
        :return: None
        """
        self.run_button.config(state=tk.DISABLED)
        self.loading_label.config(text="Processing...")

        try:
            # Run the demo script with the selected video path
            result = subprocess.run(["python", "src/demo.py", video_path], capture_output=True, text=True)

            # Display the output in the text widget
            self.output_text.insert(tk.END, result.stdout + "\n")
            print(result.stderr)  # Print stderr to console

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

        self.run_button.config(state=tk.NORMAL)
        self.loading_label.config(text="")


# Check if the script is run as the main program
if __name__ == "__main__":
    # Initialize an instance of the tk.Tk class and run the demo GUI application
    root = tk.Tk()
    app = DemoGUI(root)
    root.mainloop()
