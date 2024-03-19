"""
Test the data_loading module.
:meth test_char_converter_get_vocabulary(): test the get_vocabulary() method of the CharConverter class
:meth test_char_converter_get_vocabulary_size(): test the get_vocabulary_size() method of the CharConverter class
:meth test_char_converter_convert_char_to_idx_and_convert_idx_to_char_simple(): test the convert_char_to_idx(char) and
      convert_idx_to_char(idx) methods of the CharConverter class with a single character or index
:meth test_char_converter_convert_char_to_idx_and_convert_idx_to_char_list(): test the convert_char_to_idx(char) and
      convert_idx_to_char(idx) methods of the CharConverter class with a list containing a single character or index
:meth test_char_converter_convert_char_to_idx_and_convert_idx_to_char_multiple(): test the convert_char_to_idx(char) and
      convert_idx_to_char(idx) methods of the CharConverter class with a list containing multiple characters or indices
"""

import torch

from ..utils import data_loading

char_converter = data_loading.CharConverter()


def test_char_converter_get_vocabulary() -> None:
    """
    Test the get_vocabulary() method of the CharConverter class.
    :return: None
    """
    assert char_converter.get_vocabulary() == [
        "",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
        "?",
        "!",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        " ",
    ]


def test_char_converter_get_vocabulary_size() -> None:
    """
    Test the get_vocabulary_size() method of the CharConverter class.
    :return: None
    """
    assert char_converter.get_vocabulary_size() == 40


def test_char_converter_convert_char_to_idx_and_convert_idx_to_char_simple() -> None:
    """
    Test the convert_char_to_idx(char) and convert_idx_to_char(idx) methods of the CharConverter class with a single
    character or index.
    :return: None
    """
    assert char_converter.convert_char_to_idx("a") == 1
    assert char_converter.convert_char_to_idx(b"a") == 1
    assert char_converter.convert_idx_to_char(1) == b"a"
    assert char_converter.convert_idx_to_char(torch.tensor(1, dtype=torch.int32)) == b"a"


def test_char_converter_convert_char_to_idx_and_convert_idx_to_char_list() -> None:
    """
    Test the convert_char_to_idx(char) and convert_idx_to_char(idx) methods of the CharConverter class with a list
    containing a single character or index.
    :return: None
    """
    assert char_converter.convert_char_to_idx(["a"]) == torch.tensor([1], dtype=torch.int32)
    assert char_converter.convert_char_to_idx([b"a"]) == torch.tensor([1], dtype=torch.int32)
    assert char_converter.convert_idx_to_char([1]) == [b"a"]
    assert char_converter.convert_idx_to_char(torch.tensor([1], dtype=torch.int32)) == [b"a"]


def test_char_converter_convert_char_to_idx_and_convert_idx_to_char_multiple() -> None:
    """
    Test the convert_char_to_idx(char) and convert_idx_to_char(idx) methods of the CharConverter class with a list
    containing multiple characters or indices.
    """
    assert torch.equal(
        char_converter.convert_char_to_idx(["v", "l", "a", "d"]), torch.tensor([22, 12, 1, 4], dtype=torch.int32)
    )
    assert torch.equal(
        char_converter.convert_char_to_idx([b"v", b"l", b"a", b"d"]), torch.tensor([22, 12, 1, 4], dtype=torch.int32)
    )
    assert char_converter.convert_idx_to_char([22, 12, 1, 4]) == [b"v", b"l", b"a", b"d"]
    assert char_converter.convert_idx_to_char(torch.tensor([22, 12, 1, 4], dtype=torch.int32)) == [
        b"v",
        b"l",
        b"a",
        b"d",
    ]
