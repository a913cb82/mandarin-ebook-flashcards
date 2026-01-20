import json
from unittest.mock import MagicMock, patch

import pandas as pd

from flashcard import create_flashcards, validate_flashcard


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
