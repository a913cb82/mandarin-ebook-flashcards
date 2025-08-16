import json
import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.main import (
    SYSTEM_PROMPT,
    create_flashcards,
    extract_vocabulary,
    main,
    read_epub,
    save_flashcards,
    validate_flashcard,
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

def test_extract_vocabulary_with_stop_words(tmp_path) -> None:
    """
    Tests the extract_vocabulary function with a stop words file.
    """
    text = "你好世界你好，我们"
    stop_words_path = tmp_path / "stop_words.txt"
    with open(stop_words_path, "w") as f:
        f.write("你好\n我们")
    words = extract_vocabulary(text, stop_words_path=str(stop_words_path))
    assert words == ["世界"]


def test_validate_flashcard():
    """
    Tests the validate_flashcard function.
    """
    valid_flashcard = pd.Series(
        {
            "hanzi": "你好",
            "pinyin": "nǐ hǎo",
            "definition": "hello",
            "partofspeech": "greeting",
            "sentencehanzi": "你好吗？",
            "sentencepinyin": "Nǐ hǎo ma?",
            "sentencetranslation": "How are you?",
        }
    )
    assert validate_flashcard(valid_flashcard, "你好") is True

    invalid_flashcard_missing_column = valid_flashcard.drop("pinyin")
    assert validate_flashcard(invalid_flashcard_missing_column, "你好") is False

    invalid_flashcard_nan_value = valid_flashcard.copy()
    invalid_flashcard_nan_value["pinyin"] = None
    assert validate_flashcard(invalid_flashcard_nan_value, "你好") is False

    invalid_flashcard_wrong_hanzi = valid_flashcard.copy()
    invalid_flashcard_wrong_hanzi["hanzi"] = "世界"
    assert validate_flashcard(invalid_flashcard_wrong_hanzi, "你好") is False

    invalid_flashcard_hanzi_not_in_sentence = valid_flashcard.copy()
    invalid_flashcard_hanzi_not_in_sentence["sentencehanzi"] = "世界吗？"
    assert (
        validate_flashcard(invalid_flashcard_hanzi_not_in_sentence, "你好") is False
    )


@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_with_retry(mock_completion: MagicMock, tmp_path) -> None:
    """
    Tests the create_flashcards function with retry logic.
    """
    invalid_response_df = pd.DataFrame({"hanzi": ["你好"], "pinyin": [None]})
    valid_response_df = pd.DataFrame(
        {
            "hanzi": ["你好"],
            "pinyin": ["nǐ hǎo"],
            "definition": ["hello"],
            "partofspeech": ["greeting"],
            "sentencehanzi": ["你好吗？"],
            "sentencepinyin": ["Nǐ hǎo ma?"],
            "sentencetranslation": ["How are you?"],
        }
    )

    class MockResponse:
        def __init__(self, content):
            self.text = content

    mock_completion.return_value.generate_content.side_effect = [
        Exception("Invalid JSON"),
        MockResponse(
            json.dumps(valid_response_df.to_dict("records"))
        ),
    ]

    words = ["你好"]
    flashcards = create_flashcards(
        words, batch_size=1, retries=2, cache_dir=str(tmp_path)
    )
    assert len(flashcards) == 1
    assert mock_completion.return_value.generate_content.call_count == 2


@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_fails_after_retries(
    mock_completion: MagicMock,
    tmp_path,
) -> None:
    """
    Tests that create_flashcards fails after all retries.
    """
    mock_completion.return_value.generate_content.side_effect = Exception("Invalid JSON")

    words = ["你好"]
    flashcards = create_flashcards(
        words, batch_size=1, retries=3, cache_dir=str(tmp_path)
    )
    assert len(flashcards) == 0
    assert mock_completion.return_value.generate_content.call_count == 3  # 1 initial call + 2 retries


@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_with_caching(mock_completion: MagicMock, tmp_path) -> None:
    """
    Tests the caching mechanism in create_flashcards.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    valid_response_df = pd.DataFrame(
        {
            "hanzi": ["你好"],
            "pinyin": ["nǐ hǎo"],
            "definition": ["hello"],
            "partofspeech": ["greeting"],
            "sentencehanzi": ["你好吗？"],
            "sentencepinyin": ["Nǐ hǎo ma?"],
            "sentencetranslation": ["How are you?"],
        }
    )

    class MockResponse:
        def __init__(self, content):
            self.text = content

    mock_completion.return_value.generate_content.return_value = MockResponse(
        json.dumps(valid_response_df.to_dict("records"))
    )

    words = ["你好"]
    create_flashcards(words, batch_size=1, cache_dir=str(cache_dir))
    assert mock_completion.return_value.generate_content.call_count == 1

    # Run again and check that the cache is used
    create_flashcards(words, batch_size=1, cache_dir=str(cache_dir))
    assert mock_completion.return_value.generate_content.call_count == 1


@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_preserves_order(
    mock_completion: MagicMock,
    tmp_path,
) -> None:
    """
    Tests that create_flashcards preserves the order of the input words.
    """
    words = ["世界", "你好"]
    response_df = pd.DataFrame(
        {
            "hanzi": ["你好", "世界"],
            "pinyin": ["nǐ hǎo", "shì jiè"],
            "definition": ["hello", "world"],
            "partofspeech": ["greeting", "noun"],
            "sentencehanzi": ["你好吗？", "你好世界"],
            "sentencepinyin": ["Nǐ hǎo ma?", "Nǐ hǎo shì jiè"],
            "sentencetranslation": ["How are you?", "Hello world"],
        }
    )

    class MockResponse:
        def __init__(self, content):
            self.text = content

    mock_completion.return_value.generate_content.return_value = MockResponse(
        json.dumps(response_df.to_dict("records"))
    )

    flashcards = create_flashcards(words, batch_size=2, cache_dir=str(tmp_path))
    assert flashcards["hanzi"].tolist() == words


def test_save_flashcards(tmp_path) -> None:
    """
    Tests the save_flashcards function.
    """
    flashcards = pd.DataFrame(
        {
            "hanzi": ["你好"],
            "pinyin": ["nǐ hǎo"],
            "definition": ["hello"],
            "partofspeech": ["greeting"],
            "sentencehanzi": ["你好吗？"],
            "sentencepinyin": ["Nǐ hǎo ma?"],
            "sentencetranslation": ["How are you?"],
        }
    )
    output_path = tmp_path / "flashcards.tsv"
    save_flashcards(flashcards, str(output_path))
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
        assert (
            content.strip()
            == "你好\tnǐ hǎo\thello\tgreeting\t你好吗？\tNǐ hǎo ma?\tHow are you?"
        )


def test_main_vocab_only(tmp_path) -> None:
    """
    Tests the main function with the --vocab-only flag.
    """
    epub_path = "tests/test_book.epub"
    output_path = tmp_path / "output.txt"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    sys.argv = ["main.py", epub_path, str(output_path), "--vocab-only", "--cache-dir", str(cache_dir)]
    main()
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
        assert "第一章" in content
        assert "这是" in content


@patch("src.main.create_flashcards")
def test_flashcards_only(mock_create_flashcards, tmp_path):
    vocab_content = """
你好
世界
"""
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(vocab_content)

    output_path = tmp_path / "flashcards.txt"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    mock_flashcards = pd.DataFrame({
        "hanzi": ["你好", "世界"],
        "pinyin": ["nǐ hǎo", "shì jiè"],
        "definition": ["hello", "world"],
        "partofspeech": ["interjection", "noun"],
        "sentencehanzi": ["你好，世界", "你好，世界"],
        "sentencepinyin": ["nǐ hǎo, shì jiè", "nǐ hǎo, shì jiè"],
        "sentencetranslation": ["Hello, world", "Hello, world"],
    })
    mock_create_flashcards.return_value = mock_flashcards

    with patch("sys.argv", ["src/main.py", str(vocab_path), str(output_path), "--flashcards-only", "--cache-dir", str(cache_dir)]):
        main()

    assert os.path.exists(output_path)
    
    result_df = pd.read_csv(output_path, sep="\t", header=None)
    assert len(result_df) == 2
    assert result_df.iloc[0, 0] == "你好"


@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_with_custom_model(
    mock_completion: MagicMock,
    tmp_path,
) -> None:
    """
    Tests that the create_flashcards function uses the custom model.
    """
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(
        [
                {
                    "hanzi": "你好",
                    "pinyin": "nǐ hǎo",
                    "definition": "hello",
                    "partofspeech": "greeting",
                    "sentencehanzi": "你好吗？",
                    "sentencepinyin": "Nǐ hǎo ma?",
                    "sentencetranslation": "How are you?",
                }
        ]
    )
    mock_completion.return_value.generate_content.return_value = mock_response

    words = ["你好"]
    create_flashcards(
        words, batch_size=1, model="custom-model", cache_dir=str(tmp_path)
    )
    assert mock_completion.call_args[0][0] == "custom-model"


import toml

@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_uses_system_prompt_from_file(
    mock_generative_model: MagicMock,
    tmp_path,
) -> None:
    """
    Tests that create_flashcards uses the system prompt from the file.
    """
    with open("src/system_prompt.toml", "r") as f:
        prompt_data = toml.load(f)
        expected_prompt = prompt_data["system_prompt"]
        few_shot_examples = prompt_data["examples"]

    expected_messages = []
    for example in few_shot_examples:
        expected_messages.append({"role": "user", "parts": [example["input"]]})
        expected_messages.append({"role": "model", "parts": [example["output"]]})

    mock_response = MagicMock()
    mock_response.text = json.dumps(
        [
            {
                "hanzi": "你好",
                "pinyin": "nǐ hǎo",
                "definition": "hello",
                "partofspeech": "greeting",
                "sentencehanzi": "你好吗？",
                "sentencepinyin": "Nǐ hǎo ma?",
                "sentencetranslation": "How are you?",
            }
        ]
    )
    mock_generative_model.return_value.generate_content.return_value = mock_response

    words = ["你好"]
    create_flashcards(words, batch_size=1, cache_dir=str(tmp_path))

    # Check that the system prompt is set correctly
    assert mock_generative_model.call_args[1]["system_instruction"] == expected_prompt

    # Check that the messages are in the correct format
    actual_messages = mock_generative_model.return_value.generate_content.call_args[0][0]
    assert actual_messages[:len(expected_messages)] == expected_messages
    assert actual_messages[-1] == {"role": "user", "parts": [",".join(words)]}
    
    assert mock_generative_model.call_args[1]["generation_config"]["response_mime_type"] == "application/json"
    assert mock_generative_model.call_args[1]["generation_config"]["response_schema"]["type"] == "array"
