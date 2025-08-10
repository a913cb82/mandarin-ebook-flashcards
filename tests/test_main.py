import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.main import (
    create_flashcards,
    extract_words,
    main,
    read_epub,
    save_flashcards,
)


def mock_create_flashcards(*args, **kwargs) -> pd.DataFrame:
    """
    A mock version of the create_flashcards function.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing sample flashcards.
    """
    data = {
        "hanzi": ["你好", "世界", "我们", "在", "打包", "东西"],
        "pinyin": ["nǐ hǎo", "shìjiè", "wǒmen", "zài", "dǎbāo", "dōngxi"],
        "definition": [
            "hello",
            "world",
            "we/us",
            "at/in",
            "to pack",
            "thing/stuff",
        ],
        "partofspeech": [
            "greeting",
            "noun",
            "pronoun",
            "preposition",
            "verb",
            "noun",
        ],
        "sentencehanzi": [
            "你好吗？",
            "世界是圆的。",
            "我们是朋友。",
            "我在家。",
            "我在打包行李。",
            "这是什么东西？",
        ],
        "sentencepinyin": [
            "Nǐ hǎo ma?",
            "Shìjiè shì yuán de.",
            "Wǒmen shì péngyǒu.",
            "Wǒ zài jiā.",
            "Wǒ zài dǎbāo xínglǐ.",
            "Zhè shì shénme dōngxi?",
        ],
        "sentencetranslation": [
            "How are you?",
            "The world is round.",
            "We are friends.",
            "I am at home.",
            "I am packing luggage.",
            "What is this thing?",
        ],
    }
    return pd.DataFrame(data)


def test_read_epub() -> None:
    """
    Tests the read_epub function.
    """
    content = read_epub("tests/test_book.epub")
    assert isinstance(content, str)
    assert "第一章" in content
    assert "第二章" in content
    assert "第三章" in content
    assert content.find("第一章") < content.find("第二章") < content.find("第三章")


def test_extract_words() -> None:
    """
    Tests the extract_words function.
    """
    text = "你好世界你好"
    words = extract_words(text)
    assert isinstance(words, list)
    assert words == ["你好", "世界"]


@patch("src.main.completion")
def test_create_flashcards_with_retry(mock_completion: MagicMock) -> None:
    """
    Tests the create_flashcards function with retry logic.

    Parameters
    ----------
    mock_completion : MagicMock
        A mock of the litellm.completion function.
    """
    invalid_response = MagicMock()
    invalid_response.choices[0].message.content = "invalid"
    valid_response = MagicMock()
    valid_response.choices[0].message.content = (
        "hanzi\tpinyin\tdefinition\tpartofspeech\tsentencehanzi\tsentencepinyin\tsentencetranslation\n"
        "你好\tnǐ hǎo\thello\tgreeting\t你好吗？\tNǐ hǎo ma?\tHow are you?"
    )
    mock_completion.side_effect = [invalid_response, valid_response]

    words = ["你好"]
    flashcards = create_flashcards(words, batch_size=1, retries=2)
    assert len(flashcards) == 1
    assert mock_completion.call_count == 2


@patch("src.main.completion")
def test_create_flashcards_raises_error_after_retries(
    mock_completion: MagicMock,
) -> None:
    """
    Tests that create_flashcards raises an error after all retries fail.

    Parameters
    ----------
    mock_completion : MagicMock
        A mock of the litellm.completion function.
    """
    invalid_response = MagicMock()
    invalid_response.choices[0].message.content = "invalid"
    mock_completion.return_value = invalid_response

    words = ["你好"]
    with pytest.raises(ValueError):
        create_flashcards(words, batch_size=1, retries=3)
    assert mock_completion.call_count == 3



@patch("src.main.create_flashcards", new=mock_create_flashcards)
def test_end_to_end() -> None:
    """
    Tests the main function in an end-to-end scenario.
    """
    epub_path = "tests/test_book.epub"
    output_path = "tests/output.tsv"

    sys.argv = ["main.py", epub_path, output_path, "--batch_size", "10"]

    main()

    assert os.path.exists(output_path)

    with open(output_path, "r") as f:
        content = f.read()
        assert "你好" in content

    # Clean up the output file
    os.remove(output_path)
