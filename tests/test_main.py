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
            "pinyin": "ní hǎo",
            "pinyinnumbered": "ni2 hao3",
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


def test_validate_flashcard_extended():
    """
    Tests the extended validation logic in validate_flashcard.
    """
    base_flashcard = {
        "hanzi": "行",
        "pinyin": "xíng | háng",
        "pinyinnumbered": "xing2 | hang2",
        "definition": "to walk; to go | a row; a line",
        "partofspeech": "verb | noun",
        "sentencehanzi": "好的，这样也行。",
        "sentencepinyin": "Hǎo de, zhèyàng yě xíng.",
        "sentencetranslation": "Ok, this way is also acceptable.",
    }

    # Valid flashcard
    valid_flashcard = pd.Series(base_flashcard)
    assert validate_flashcard(valid_flashcard, "行") is True

    # Invalid: pinyin and pinyinnumbered have different number of pipe-separated parts
    invalid_flashcard = pd.Series(base_flashcard.copy())
    invalid_flashcard["pinyinnumbered"] = "xing2"
    assert validate_flashcard(invalid_flashcard, "行") is False

    # Invalid: pinyin and definition have different number of pipe-separated parts
    invalid_flashcard = pd.Series(base_flashcard.copy())
    invalid_flashcard["definition"] = "to walk; to go"
    assert validate_flashcard(invalid_flashcard, "行") is False

    # Invalid: pinyin and pinyinnumbered have different number of semicolon-separated parts
    invalid_flashcard = pd.Series(base_flashcard.copy())
    invalid_flashcard["pinyin"] = "xíng; xing2 | háng"
    assert validate_flashcard(invalid_flashcard, "行") is False


@pytest.mark.parametrize("cache_tokens", [True, False])
@patch("src.main.validate_flashcard", side_effect=[False, False, True, True, True, True])
@patch("google.generativeai.caching.CachedContent")
@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_dynamic_batch_size(
    mock_completion: MagicMock,
    mock_cached_content: MagicMock,
    mock_validate_flashcard: MagicMock,
    cache_tokens: bool,
    tmp_path,
) -> None:
    """
    Tests that the batch size is adjusted dynamically.
    """
    words = ["你好", "世界", "我们", "他们"]
    response_df = pd.DataFrame(
        {
            "hanzi": ["你好", "世界", "我们", "他们"],
            "pinyin": ["ní hǎo", "shì jiè", "wǒ men", "tā men"],
            "pinyinnumbered": ["ni2 hao3", "shi4 jie4", "wo3 men5", "ta1 men5"],
            "definition": ["hello", "world", "we", "they"],
            "partofspeech": ["greeting", "noun", "pronoun", "pronoun"],
            "sentencehanzi": ["你好吗？", "你好世界", "我们是朋友", "他们是学生"],
            "sentencepinyin": ["Nǐ hǎo ma?", "Nǐ hǎo shì jiè", "Wǒmen shì péngyǒu", "Tāmen shì xuéshēng"],
            "sentencetranslation": ["How are you?", "Hello world", "We are friends", "They are students"],
        }
    )

    class MockResponse:
        def __init__(self, content):
            self.text = content

    mock_completion.return_value.generate_content.return_value = MockResponse(
        json.dumps(response_df.to_dict("records"))
    )

    flashcards = create_flashcards(
        words,
        initial_batch_size=2,
        batch_size_multiplier=2.0,
        retries=2,
        cache_dir=str(tmp_path),
        verbose=True,
        cache_tokens=cache_tokens,
    )
    assert len(flashcards) == 4
    assert mock_completion.return_value.generate_content.call_count == 3


@pytest.mark.parametrize("cache_tokens", [True, False])
@patch("google.generativeai.caching.CachedContent")
@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_fails_after_retries(
    mock_completion: MagicMock,
    mock_cached_content: MagicMock,
    cache_tokens: bool,
    tmp_path,
) -> None:
    """
    Tests that create_flashcards fails after all retries.
    """
    mock_completion.return_value.generate_content.side_effect = Exception("Invalid JSON")

    words = ["你好"]
    flashcards = create_flashcards(
        words, initial_batch_size=1, retries=3, cache_dir=str(tmp_path), cache_tokens=cache_tokens
    )
    assert len(flashcards) == 0
    assert mock_completion.return_value.generate_content.call_count == 3


@pytest.mark.parametrize("cache_tokens", [True, False])
@patch("google.generativeai.caching.CachedContent")
@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_with_caching(mock_completion: MagicMock, mock_cached_content: MagicMock, cache_tokens: bool, tmp_path) -> None:
    """
    Tests the caching mechanism in create_flashcards.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    valid_response_df = pd.DataFrame(
        {
            "hanzi": ["你好"],
            "pinyin": ["ní hǎo"],
            "pinyinnumbered": ["ni2 hao3"],
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
    create_flashcards(words, initial_batch_size=1, cache_dir=str(cache_dir), cache_tokens=cache_tokens)
    assert mock_completion.return_value.generate_content.call_count == 1

    # Run again and check that the cache is used
    create_flashcards(words, initial_batch_size=1, cache_dir=str(cache_dir), cache_tokens=cache_tokens)
    assert mock_completion.return_value.generate_content.call_count == 1


@pytest.mark.parametrize("cache_tokens", [True, False])
@patch("google.generativeai.caching.CachedContent")
@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_preserves_order(
    mock_completion: MagicMock,
    mock_cached_content: MagicMock,
    cache_tokens: bool,
    tmp_path,
) -> None:
    """
    Tests that create_flashcards preserves the order of the input words.
    """
    words = ["世界", "你好"]
    response_df = pd.DataFrame(
        {
            "hanzi": ["你好", "世界"],
            "pinyin": ["ní hǎo", "shì jiè"],
            "pinyinnumbered": ["ni2 hao3", "shi4 jie4"],
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

    flashcards = create_flashcards(words, initial_batch_size=2, cache_dir=str(tmp_path), cache_tokens=cache_tokens)
    assert flashcards["hanzi"].tolist() == words


def test_save_flashcards(tmp_path) -> None:
    """
    Tests the save_flashcards function.
    """
    flashcards = pd.DataFrame(
        {
            "hanzi": ["你好"],
            "pinyin": ["ní hǎo"],
            "pinyinnumbered": ["ni2 hao3"],
            "definition": ["hello"],
            "partofspeech": ["greeting"],
            "sentencehanzi": "你好吗？",
            "sentencepinyin": "Nǐ hǎo ma?",
            "sentencetranslation": "How are you?",
        }
    )
    output_path = tmp_path / "flashcards.tsv"
    save_flashcards(flashcards, str(output_path))
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
        assert (
            content.strip()
            == "你好\tní hǎo\tni2 hao3\thello\tgreeting\t你好吗？\tNǐ hǎo ma?\tHow are you?"
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
        "pinyin": ["ní hǎo", "shì jiè"],
        "pinyinnumbered": ["ni2 hao3", "shi4 jie4"],
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


@pytest.mark.parametrize("cache_tokens", [True, False])
@patch("google.generativeai.caching.CachedContent")
@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_with_custom_model(
    mock_completion: MagicMock,
    mock_cached_content: MagicMock,
    cache_tokens: bool,
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
                    "pinyin": "ní hǎo",
                    "pinyinnumbered": "ni2 hao3",
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
        words, initial_batch_size=1, model="custom-model", cache_dir=str(tmp_path), cache_tokens=cache_tokens
    )
    assert mock_completion.call_args[0][0] == "custom-model"


import toml

@pytest.mark.parametrize("cache_tokens", [True, False])
@patch("google.generativeai.caching.CachedContent")
@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_uses_system_prompt_from_file(
    mock_generative_model: MagicMock,
    mock_cached_content: MagicMock,
    cache_tokens: bool,
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
                "pinyin": "ní hǎo",
                "pinyinnumbered": "ni2 hao3",
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
    create_flashcards(words, initial_batch_size=1, cache_dir=str(tmp_path), cache_tokens=cache_tokens)

    # Check that the system prompt is set correctly
    assert mock_generative_model.call_args[1]["system_instruction"] == expected_prompt

    if cache_tokens:
        # Check that cached_content.create was called correctly
        mock_cached_content.create.assert_called_once_with(
            model=mock_generative_model.call_args[0][0],
            contents=expected_messages,
        )
        # Check that generate_content was called with cached_content and the batch
        mock_generative_model.return_value.generate_content.assert_called_once_with(
            contents=[{"role": "user", "parts": ["..".join(words)]}],
            cached_content=mock_cached_content.create.return_value,
        )
    else:
        # Check that cached_content.create was NOT called
        mock_cached_content.create.assert_not_called()
        # Check that generate_content was called with messages (few-shot + batch)
        actual_messages = mock_generative_model.return_value.generate_content.call_args[0][0]
        assert actual_messages[:len(expected_messages)] == expected_messages
        assert actual_messages[-1] == {"role": "user", "parts": [",".join(words)]}


import toml

def test_system_prompt_examples():
    """
    Tests that the examples in the system prompt are valid.
    """
    with open("src/system_prompt.toml", "r") as f:
        prompt_data = toml.load(f)
        examples = prompt_data["examples"]

    for example in examples:
        flashcards = json.loads(example["output"])
        for flashcard_dict in flashcards:
            flashcard = pd.Series(flashcard_dict)
            assert validate_flashcard(flashcard, flashcard["hanzi"]) is True


@pytest.mark.parametrize("cache_tokens", [True, False])
@patch("src.main.tqdm")
@patch("google.generativeai.caching.CachedContent")
@patch("google.generativeai.GenerativeModel")
def test_create_flashcards_progress_bar(
    mock_generative_model: MagicMock, mock_cached_content: MagicMock, mock_tqdm: MagicMock, cache_tokens: bool, tmp_path
) -> None:
    """
    Tests that the progress bar is updated correctly in create_flashcards.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Mock the response from the generative model
    valid_response_df = pd.DataFrame(
        {
            "hanzi": ["世界"],
            "pinyin": ["shì jiè"],
            "pinyinnumbered": ["shi4 jie4"],
            "definition": ["world"],
            "partofspeech": ["noun"],
            "sentencehanzi": ["你好世界"],
            "sentencepinyin": ["Nǐ hǎo shì jiè"],
            "sentencetranslation": ["Hello world"],
        }
    )

    class MockResponse:
        def __init__(self, content):
            self.text = content

    mock_generative_model.return_value.generate_content.return_value = MockResponse(
        json.dumps(valid_response_df.to_dict("records"))
    )

    # Create a cached flashcard
    cached_flashcard = pd.Series(
        {
            "hanzi": "你好",
            "pinyin": "ní hǎo",
            "pinyinnumbered": "ni2 hao3",
            "definition": "hello",
            "partofspeech": "greeting",
            "sentencehanzi": "你好吗？",
            "sentencepinyin": "Nǐ hǎo ma?",
            "sentencetranslation": "How are you?",
        }
    )
    cached_flashcard_path = cache_dir / "你好.json"
    cached_flashcard.to_json(cached_flashcard_path)

    words = ["你好", "世界"]
    create_flashcards(words, initial_batch_size=1, cache_dir=str(cache_dir), verbose=True, cache_tokens=cache_tokens)

    # Check that tqdm was called
    mock_tqdm.assert_called_once_with(total=len(words), desc="Creating flashcards")

    # Check that the progress bar was updated for both the cached and non-cached word
    mock_pbar = mock_tqdm.return_value
    assert mock_pbar.update.call_count == 2