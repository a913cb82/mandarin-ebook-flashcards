
import unittest
from src.word_splitter import split_text_into_words

class TestWordSplitter(unittest.TestCase):

    def test_split_text_into_words(self):
        text = "你好世界"
        expected_words = ["你好", "世界"]
        self.assertEqual(split_text_into_words(text), expected_words)

if __name__ == "__main__":
    unittest.main()
