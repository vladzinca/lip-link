"""
DocString
"""

from utils.data_loader import DataLoader

data_loader = DataLoader()

TEST_PATH = "./data/s1/bbaf2n.mpg"

frames, alignments = data_loader.load_data(TEST_PATH)

print(frames)
print(alignments)
