
import unittest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.main import main

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.ebook_path = Path("/data/data/com.termux/files/home/projects/ebook_flashcards/tests/test_ebook_for_integration.txt")
        self.output_path = Path("/data/data/com.termux/files/home/projects/ebook_flashcards/tests/test_flashcards.tsv")

    def tearDown(self):
        if self.output_path.exists():
            self.output_path.unlink()

    @patch('src.flashcard_generator.generate_flashcards')
    def test_main_integration(self, mock_generate_flashcards):
        # Mock the flashcard generation
        mock_df = pd.DataFrame({
            "hanzi": ["你好", "世界", "Gemini"],
            "pinyin": ["nǐ hǎo", "shì jiè", "Gemini"],
            "definition": ["hello", "world", "Gemini"],
            "partofspeech": ["interjection", "noun", "noun"],
            "sentencehanzi": ["你好吗？", "探索新世界", "Gemini 是一个大型语言模型"],
            "sentencepinyin": ["nǐ hǎo ma?", "tàn suǒ xīn shì jiè", "Gemini shì yí ge dà xíng yǔ yán mó xíng"],
            "sentencetranslation": ["How are you?", "Explore a new world", "Gemini is a large language model"]
        })
        mock_generate_flashcards.return_value = mock_df

        # Run the main function
        main(self.ebook_path, self.output_path)

        # Check that the output file was created
        self.assertTrue(self.output_path.exists())

        # Check the content of the output file
        output_df = pd.read_csv(self.output_path, sep='\t', header=None)
        self.assertEqual(len(output_df), 3)
        self.assertEqual(output_df.iloc[0, 0], "你好")

if __name__ == '__main__':
    unittest.main()
