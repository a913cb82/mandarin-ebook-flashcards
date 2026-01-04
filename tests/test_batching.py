import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.main import create_flashcards


@patch("google.genai.Client")
def test_batch_size_doubles_on_success(mock_client, tmp_path):
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
        if kwargs.get('contents'):
            messages = kwargs['contents']
        else:
            messages = kwargs.get('contents', args[0] if args else [])
        
        # In the new SDK, contents is a list of types.Content or dicts
        user_message = messages[-1]
        if hasattr(user_message, 'parts'):
            text = user_message.parts[0].text
        else:
            text = user_message['parts'][0].text if hasattr(user_message['parts'][0], 'text') else user_message['parts'][0]

        # extract the words from the user message
        batch_words = text.split('..')

        response_data = []
        for word in batch_words:
            if word == 'word1':
                 # Intentionally invalid data for word1
                 response_data.append({
                    "hanzi": word,
                    "pinyin": "invalid pinyin",
                    "pinyinnumbered": "hao3", # Mismatch
                    "definition": "def",
                    "partofspeech": "pos",
                    "sentencehanzi": f"sent {word}",
                    "sentencepinyin": "sentpinyin",
                    "sentencetranslation": "senttrans",
                 })
            else:
                response_data.append({
                    "hanzi": word,
                    "pinyin": "hǎo",
                    "pinyinnumbered": "hao3",
                    "definition": "def",
                    "partofspeech": "pos",
                    "sentencehanzi": f"sent {word}",
                    "sentencepinyin": "sentpinyin",
                    "sentencetranslation": "senttrans",
                })

        response_df = pd.DataFrame(response_data)
        return MockResponse(json.dumps(response_df.to_dict("records")))

    mock_client.return_value.models.generate_content.side_effect = generate_content_side_effect

    with patch("builtins.print") as mock_print:
        create_flashcards(
            words,
            initial_batch_size=2,
            cache_dir=str(tmp_path),
            verbose=1,
        )
    
        mock_print.assert_any_call("Batch succeeded, increasing batch size to 4")


@patch("google.genai.Client")
def test_binary_search_on_failure(mock_client, tmp_path):
    """
    Tests that the batch size is adjusted after a failure.
    """
    words = [f"word{i}" for i in range(100)]

    class MockResponse:
        def __init__(self, content):
            self.text = content

    def generate_content_side_effect(*args, **kwargs):
        # Batch size is inferred from the number of words in the prompt
        messages = kwargs.get('contents', args[0] if args else [])
        user_message = messages[-1]
        if hasattr(user_message, 'parts'):
            text = user_message.parts[0].text
        else:
            text = user_message['parts'][0].text if hasattr(user_message['parts'][0], 'text') else user_message['parts'][0]
            
        batch_size = len(text.split('..'))
        if batch_size > 20:
            raise Exception("Batch size too large")

        response_df = pd.DataFrame(
            [
                {
                    "hanzi": word,
                    "pinyin": "hǎo",
                    "pinyinnumbered": "hao3",
                    "definition": "def",
                    "partofspeech": "pos",
                    "sentencehanzi": f"sent {word}",
                    "sentencepinyin": "sentpinyin",
                    "sentencetranslation": "senttrans",
                }
                for word in text.split('..')
            ]
        )
        return MockResponse(json.dumps(response_df.to_dict("records")))

    mock_client.return_value.models.generate_content.side_effect = generate_content_side_effect

    with patch("builtins.print") as mock_print:
        create_flashcards(
            words,
            initial_batch_size=4,
            cache_dir=str(tmp_path),
            verbose=1,
        )

        # Phase 1: Exponential growth
        mock_print.assert_any_call("Batch succeeded, increasing batch size to 8")
        mock_print.assert_any_call("Batch succeeded, increasing batch size to 16")
        
        # When it tries 32, it fails
        mock_print.assert_any_call("Batch failed, reducing batch size to 16")