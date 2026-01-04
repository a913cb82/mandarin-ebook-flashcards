import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.main import create_flashcards


@patch("google.generativeai.GenerativeModel")
def test_batch_size_doubles_on_success(mock_generative_model, tmp_path):
    """
    Tests that the batch size doubles on a successful batch, even if some words
    fail validation.
    """
    words = [f"word{i}" for i in range(10)]

    class MockResponse:
        def __init__(self, content):
            self.text = content

    def generate_content_side_effect(*args, **kwargs):
        # The number of words in the batch is the number of words in the prompt
        # separated by '..'

        # extract the user message
        if kwargs.get('cached_content'):
            messages = kwargs['contents']
        else:
            messages = args[0]
        user_message = messages[-1]

        # extract the words from the user message
        batch_words = user_message['parts'][0].split('..')

        response_df = pd.DataFrame(
            [
                {
                    "hanzi": word,
                    "pinyin": "pinyin",
                    "pinyinnumbered": "pinyin1",
                    "definition": "def",
                    "partofspeech": "pos",
                    "sentencehanzi": f"sent {word}",
                    "sentencepinyin": "sentpinyin",
                    "sentencetranslation": "senttrans",
                }
                for word in batch_words
            ]
        )
        return MockResponse(json.dumps(response_df.to_dict("records")))

    mock_generative_model.return_value.generate_content.side_effect = generate_content_side_effect

            with patch("src.main.validate_flashcard", side_effect=lambda flashcard, word, verbose: word != 'word1'):
                with patch("builtins.print") as mock_print:
                    create_flashcards(
                        words,
                        initial_batch_size=2,
                        cache_dir=str(tmp_path),
                        verbose=1,
                        batch_size_multiplier=2.0,
                    )
        
            mock_print.assert_any_call("Batch size set to 4")
@patch("google.generativeai.GenerativeModel")
def test_binary_search_on_failure(mock_generative_model, tmp_path):
    """
    Tests that the batch size is adjusted using binary search after a failure.
    """
    words = [f"word{i}" for i in range(100)]

    class MockResponse:
        def __init__(self, content):
            self.text = content

    def generate_content_side_effect(*args, **kwargs):
        # Batch size is inferred from the number of words in the prompt
        if kwargs.get('cached_content'):
            messages = kwargs['contents']
        else:
            messages = args[0]
        user_message = messages[-1]
        batch_size = len(user_message['parts'][0].split('..'))
        if batch_size > 20:
            raise Exception("Batch size too large")

        response_df = pd.DataFrame(
            [
                {
                    "hanzi": word,
                    "pinyin": "pinyin",
                    "pinyinnumbered": "pinyin1",
                    "definition": "def",
                    "partofspeech": "pos",
                    "sentencehanzi": f"sent {word}",
                    "sentencepinyin": "sentpinyin",
                    "sentencetranslation": "senttrans",
                }
                for word in user_message['parts'][0].split('..')
            ]
        )
        return MockResponse(json.dumps(response_df.to_dict("records")))

    mock_generative_model.return_value.generate_content.side_effect = generate_content_side_effect

    with patch("builtins.print") as mock_print:
        create_flashcards(
            words,
            initial_batch_size=4,
            cache_dir=str(tmp_path),
            verbose=1,
        )

        # Phase 1: Exponential growth
        mock_print.assert_any_call("Batch size set to 8")

    # Phase 2: Binary search
    mock_print.assert_any_call("Entering binary search for batch size between 16 and 32")
    mock_print.assert_any_call("Batch size set to 24")  # (16 + 32) // 2, fails
    mock_print.assert_any_call("Batch size set to 20")  # (16 + 24) // 2, fails
    # The final batch size should be around 19 or 20