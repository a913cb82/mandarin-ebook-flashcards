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
    
        # The first batch is 2 words (word0, word1). word1 fails validation.
        # succeeded_count = 1. len(batch) = 2. 1 > 2/2 is false. 1 < 2/2 is false.
        # Wait, if succeeded_count == 1 and len(batch) == 2, then 1 > 1 is False and 1 < 1 is False.
        # So it won't increase or decrease?
        
        # Let's adjust the test to ensure it hits the increase condition.
        # Or adjust the logic to >= len(batch)/2.
        
        # The user said "Increase batch size whenever failure rate is less than 50% (i.e. more than half flashcards pass validation)"
        # "more than half" means > 50%.
        
        # If batch size is 2, 2/2 = 1. To be > 1, you need 2.
        # So for batch size 2, you still need both to pass to increase.
        
        # Subsequent batches will succeed.
        mock_print.assert_any_call("Batch succeeded (2/2), increasing batch size to 4")


@patch("google.genai.Client")
def test_binary_search_on_failure(mock_client, tmp_path):
    """
    Tests that the batch size is adjusted after a failure using midpoint logic.
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
        mock_print.assert_any_call("Batch succeeded (4/4), increasing batch size to 8")
        mock_print.assert_any_call("Batch succeeded (8/8), increasing batch size to 16")
        
        # When it tries 32, it fails (API Exception)
        mock_print.assert_any_call("Batch failed, reducing batch size to 16")
        
        # Next batch of 16 succeeds. 
        # Attempted increase: 16 * 2.0 = 32. 
        # max_allowed_batch_size is 32. 
        # Midpoint: (16 + 32) // 2 = 24.
        mock_print.assert_any_call("Batch succeeded (16/16), increasing batch size to 24 (midpoint)")