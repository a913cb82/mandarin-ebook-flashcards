import json
from unittest.mock import MagicMock, patch

import pandas as pd

from flashcard import EXPECTED_COLUMNS, create_flashcards, save_flashcards, validate_flashcard


def test_save_flashcards_with_tabs(tmp_path):
    output_path = tmp_path / "output.tsv"
    card = {
        "hanzi": "你好",
        "pinyin": "ní hǎo",
        "pinyinnumbered": "ni2 hao3",
        "definition": "hello\tworld",  # Tab here
        "partofspeech": "greeting",
        "sentencehanzi": "你好吗？",
        "sentencepinyin": "sp",
        "sentencetranslation": "st",
    }
    df = pd.DataFrame([card])
    save_flashcards(df, str(output_path))

    content = output_path.read_text()
    lines = content.strip().split("\n")
    for line in lines:
        parts = line.split("\t")
        assert len(parts) == len(EXPECTED_COLUMNS)
        assert "hello world" in line  # Tab replaced by space


def test_validate_flashcard():
    card = pd.Series(
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
    assert validate_flashcard(card, "你好") is True
    assert validate_flashcard(card.drop("pinyin"), "你好") is False


def test_validate_flashcard_nu_er():
    card = pd.Series(
        {
            "hanzi": "女儿",
            "pinyin": "nǚ ér",
            "pinyinnumbered": "nv3 er2",
            "definition": "daughter",
            "partofspeech": "noun",
            "sentencehanzi": "这是我的女儿。",
            "sentencepinyin": "Zhè shì wǒ de nǚ'ér.",
            "sentencetranslation": "This is my daughter.",
        }
    )
    assert validate_flashcard(card, "女儿") is True


def test_validate_flashcard_failures():
    base_card = {
        "hanzi": "你好",
        "pinyin": "ní hǎo",
        "pinyinnumbered": "ni2 hao3",
        "definition": "hello",
        "partofspeech": "greeting",
        "sentencehanzi": "你好吗？",
        "sentencepinyin": "Nǐ hǎo ma?",
        "sentencetranslation": "How are you?",
    }

    # Hanzi mismatch
    card = base_card.copy()
    card["hanzi"] = "再见"
    assert validate_flashcard(card, "你好") is False

    # Pinyin mismatch
    card = base_card.copy()
    card["pinyin"] = "nǐ hǎo"  # Tone mark 3 vs 2
    assert validate_flashcard(card, "你好") is False

    # Empty value
    card = base_card.copy()
    card["definition"] = ""
    assert validate_flashcard(card, "你好") is False

    # Word not in sentence
    card = base_card.copy()
    card["sentencehanzi"] = "我很好。"
    assert validate_flashcard(card, "你好") is False

    # Pipe count mismatch
    card = base_card.copy()
    card["pinyin"] = "ní hǎo | nǐ hǎo"
    card["definition"] = "hello"  # Only 1 definition for 2 pinyin
    assert validate_flashcard(card, "你好") is False

    # Semicolon structure mismatch
    card = base_card.copy()
    card["pinyin"] = "ní; hǎo"
    card["pinyinnumbered"] = "ni2 hao3"  # No semicolon here
    assert validate_flashcard(card, "你好") is False


@patch("random.shuffle")
@patch("google.genai.Client")
def test_create_flashcards_shuffles_and_preserves_order(mock_client, mock_shuffle, tmp_path):
    words = ["你好", "世界"]
    mock_client.return_value.models.generate_content.return_value = MagicMock(
        text=json.dumps(
            [
                {
                    "hanzi": "你好",
                    "pinyin": "ní hǎo",
                    "pinyinnumbered": "ni2 hao3",
                    "definition": "hello",
                    "partofspeech": "noun",
                    "sentencehanzi": "你好世界",
                    "sentencepinyin": "Ní hǎo shì jiè.",
                    "sentencetranslation": "Hello world.",
                },
                {
                    "hanzi": "世界",
                    "pinyin": "shì jiè",
                    "pinyinnumbered": "shi4 jie4",
                    "definition": "world",
                    "partofspeech": "noun",
                    "sentencehanzi": "你好世界",
                    "sentencepinyin": "Ní hǎo shì jiè.",
                    "sentencetranslation": "Hello world.",
                },
            ]
        )
    )
    flashcards = create_flashcards(words, initial_batch_size=2, cache_dir=str(tmp_path))
    mock_shuffle.assert_called_once()
    assert set(flashcards["hanzi"].tolist()) == set(words)


@patch("google.genai.Client")
def test_batch_size_doubles_on_success(mock_client, tmp_path):
    words = [f"word{i}" for i in range(4)]

    def side_effect(*args, **kwargs):
        text = kwargs["contents"][-1].parts[0].text
        batch = text.split("..")
        return MagicMock(
            text=json.dumps(
                [
                    {
                        "hanzi": w,
                        "pinyin": "hǎo",
                        "pinyinnumbered": "hao3",
                        "definition": "d",
                        "partofspeech": "n",
                        "sentencehanzi": f"s {w}",
                        "sentencepinyin": "sp",
                        "sentencetranslation": "st",
                    }
                    for w in batch
                ]
            )
        )

    mock_client.return_value.models.generate_content.side_effect = side_effect
    with patch("builtins.print") as mock_print:
        create_flashcards(words, initial_batch_size=2, cache_dir=str(tmp_path), verbose=True)
        increase_calls = [c for c in mock_print.mock_calls if "increasing batch size" in str(c)]
        assert len(increase_calls) > 0


@patch("google.genai.Client")
def test_batch_size_decreases_on_failure(mock_client, tmp_path):
    words = [f"word{i}" for i in range(4)]

    # Mock success for first batch, then fail validation for second
    responses = [
        json.dumps(
            [
                {
                    "hanzi": w,
                    "pinyin": "hǎo",
                    "pinyinnumbered": "hao3",
                    "definition": "d",
                    "partofspeech": "n",
                    "sentencehanzi": f"s {w}",
                    "sentencepinyin": "sp",
                    "sentencetranslation": "st",
                }
                for w in ["word0", "word1"]
            ]
        ),
        json.dumps([]),  # Empty results will fail validation/succeeded check
    ]
    mock_client.return_value.models.generate_content.side_effect = [
        MagicMock(text=r) for r in responses
    ]

    with patch("builtins.print") as mock_print:
        create_flashcards(words, initial_batch_size=2, cache_dir=str(tmp_path), verbose=True)
        decrease_calls = [c for c in mock_print.mock_calls if "decreasing batch size" in str(c)]
        assert len(decrease_calls) > 0


@patch("time.sleep")
@patch("google.genai.Client")
def test_create_flashcards_handles_429(mock_client, mock_sleep, tmp_path):
    word = "你好"
    # First call returns 429 error, second returns success
    mock_client.return_value.models.generate_content.side_effect = [
        Exception("Resource has been exhausted (e.g. check quota). [429]"),
        MagicMock(
            text=json.dumps(
                [
                    {
                        "hanzi": word,
                        "pinyin": "ní hǎo",
                        "pinyinnumbered": "ni2 hao3",
                        "definition": "h",
                        "partofspeech": "n",
                        "sentencehanzi": "你好吗？",
                        "sentencepinyin": "sp",
                        "sentencetranslation": "st",
                    }
                ]
            )
        ),
    ]

    create_flashcards([word], initial_batch_size=1, cache_dir=str(tmp_path))
    assert mock_sleep.called
    assert mock_client.return_value.models.generate_content.call_count == 2


@patch("google.genai.Client")
def test_create_flashcards_hits_max_retries(mock_client, tmp_path):
    word = "你好"
    # Always return invalid data
    mock_client.return_value.models.generate_content.return_value = MagicMock(
        text=json.dumps([{"hanzi": "wrong"}])
    )

    with patch("builtins.print") as mock_print:
        flashcards = create_flashcards(
            [word], initial_batch_size=1, retries=2, cache_dir=str(tmp_path), verbose=True
        )
        assert flashcards.empty
        # Check for failure message
        failure_msg = f"Failed to create valid flashcard for word: {word}"
        failure_calls = [c for c in mock_print.mock_calls if failure_msg in str(c)]
        assert len(failure_calls) > 0


@patch("google.genai.Client")
def test_create_flashcards_with_caching(mock_client, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    word = "你好"
    card = {
        "hanzi": word,
        "pinyin": "ní hǎo",
        "pinyinnumbered": "ni2 hao3",
        "definition": "h",
        "partofspeech": "n",
        "sentencehanzi": "你好吗？",
        "sentencepinyin": "sp",
        "sentencetranslation": "st",
    }
    mock_client.return_value.models.generate_content.return_value = MagicMock(
        text=json.dumps([card])
    )

    create_flashcards([word], initial_batch_size=1, cache_dir=str(cache_dir))
    assert mock_client.return_value.models.generate_content.call_count == 1

    create_flashcards([word], initial_batch_size=1, cache_dir=str(cache_dir))
    assert mock_client.return_value.models.generate_content.call_count == 1
