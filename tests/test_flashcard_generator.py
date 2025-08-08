
import unittest
import os
import pandas as pd
from src.flashcard_generator import generate_flashcards

class TestFlashcardGenerator(unittest.TestCase):

    def test_generate_flashcards(self):
        words = ["你好", "世界"]
        # This test requires a Google API key to be set as an environment variable
        if "GOOGLE_API_KEY" not in os.environ:
            self.skipTest("GOOGLE_API_KEY environment variable not set")

        flashcards_df = generate_flashcards(words)

        self.assertIsInstance(flashcards_df, pd.DataFrame)
        self.assertEqual(list(flashcards_df.columns), ["hanzi", "pinyin", "definition", "partofspeech", "sentencehanzi", "sentencepinyin", "sentencetranslation"])
        self.assertEqual(len(flashcards_df), 2)

if __name__ == "__main__":
    unittest.main()
