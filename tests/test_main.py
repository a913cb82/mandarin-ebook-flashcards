import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.main import (
    create_flashcards,
    extract_vocabulary,
    main,
    read_epub,
    save_flashcards,
)


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


def test_extract_vocabulary() -> None:
    """
    Tests the extract_vocabulary function.
    """
    text = "你好世界你好, this is a test"
    words = extract_vocabulary(text)
    assert isinstance(words, list)
    assert set(words) == {"你好", "世界"}


def test_extract_vocabulary_with_min_freq() -> None:
    """
    Tests the extract_vocabulary function with the min_freq argument.
    """
    text = "你好世界你好我们我们我们"
    words = extract_vocabulary(text, min_freq=2)
    assert set(words) == {"你好", "我们"}


@patch("builtins.print")
def test_extract_vocabulary_with_verbose(mock_print: MagicMock) -> None:
    """
    Tests the extract_vocabulary function with the verbose argument.
    """
    text = "你好世界你好我们我们我们"
    extract_vocabulary(text, min_freq=3, verbose=True)
    mock_print.assert_any_call("Vocabulary size with min_freq=1: 3")
    mock_print.assert_any_call("Vocabulary size with min_freq=2: 2")
    mock_print.assert_any_call("Vocabulary size with min_freq=3: 1")


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


def test_extract_vocabulary_with_stop_words() -> None:
    """
    Tests the extract_vocabulary function with a stop words file.
    """
    text = "你好世界你好，我们"
    stop_words_path = "tests/stop_words.txt"
    with open(stop_words_path, "w") as f:
        f.write("你好,我们")
    words = extract_vocabulary(text, stop_words_path=stop_words_path)
    assert words == ["世界"]
    os.remove(stop_words_path)



def test_main_vocab_only() -> None:
    """
    Tests the main function with the --vocab-only flag.
    """
    epub_path = "tests/test_book.epub"
    output_path = "tests/output.txt"
    sys.argv = ["main.py", epub_path, output_path, "--vocab-only"]
    main()
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
        # The exact content depends on jieba's output, so we'll just check for a few expected words
        assert "第一章" in content
        assert "这是" in content
    os.remove(output_path)


@patch("src.main.completion")
def test_create_flashcards_with_custom_model(mock_completion: MagicMock) -> None:
    """
    Tests that the create_flashcards function uses the custom model.

    Parameters
    ----------
    mock_completion : MagicMock
        A mock of the litellm.completion function.
    """
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        "hanzi\tpinyin\tdefinition\tpartofspeech\tsentencehanzi\tsentencepinyin\tsentencetranslation\n"
        "你好\tnǐ hǎo\thello\tgreeting\t你好吗？\tNǐ hǎo ma?\tHow are you?"
    )
    mock_completion.return_value = mock_response

    words = ["你好"]
    create_flashcards(words, batch_size=1, model="custom-model")
    with open("src/promot.txt", "r") as f:
        SYSTEM_PROMPT = f.read()
    mock_completion.assert_called_with(
        model="custom-model",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "你好"},
        ],
    )


