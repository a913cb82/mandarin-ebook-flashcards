import json
import pandas as pd
from unittest.mock import MagicMock, patch
from src.main import create_flashcards

@patch("random.shuffle")
@patch("google.genai.Client")
def test_create_flashcards_shuffles_and_preserves_order(
    mock_client: MagicMock,
    mock_shuffle: MagicMock,
    tmp_path,
) -> None:
    """
    Tests that create_flashcards shuffles words_to_process but preserves final output order.
    """
    words = ["word1", "word2", "word3", "word4"]
    
    # Mock shuffle to actually do something recognizable if it's called
    # But since it's in-place, we just want to verify it was called on a list
    # containing the expected words.
    
    class MockResponse:
        def __init__(self, content):
            self.text = content

    def generate_content_side_effect(*args, **kwargs):
        messages = kwargs.get('contents', args[0] if args else [])
        batch_text = messages[-1].parts[0].text
        batch_words = batch_text.split("..")
        response_df = pd.DataFrame(
            [
                {
                                            "hanzi": word,
                                            "pinyin": "pinyin",
                                            "pinyinnumbered": "pinyin5",
                                            "definition": "def",
                                            "partofspeech": "pos",
                                            "sentencehanzi": f"sent {word}",
                                            "sentencepinyin": "sentpinyin",
                                            "sentencetranslation": "senttrans",
                                        }                for word in batch_words
            ]
        )
        return MockResponse(json.dumps(response_df.to_dict("records")))

    mock_client.return_value.models.generate_content.side_effect = generate_content_side_effect

    flashcards = create_flashcards(
        words,
        initial_batch_size=2,
        cache_dir=str(tmp_path),
        verbose=False,
    )
    
    # 1. Verify random.shuffle was called
    mock_shuffle.assert_called_once()
    shuffled_list = mock_shuffle.call_args[0][0]
    assert set(shuffled_list) == set(words)
    
    # 2. Verify final order is preserved
    assert flashcards["hanzi"].tolist() == words
