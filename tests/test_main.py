
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.main import main

class TestMain(unittest.TestCase):

    @patch("src.main.read_ebook")
    @patch("src.main.split_text_into_words")
    @patch("src.main.generate_flashcards")
    @patch("src.main.save_dataframe_to_tsv")
    def test_main(self, mock_save, mock_generate, mock_split, mock_read):
        ebook_path = Path("test_ebook.txt")
        output_path = Path("test_output.tsv")

        # Create a dummy ebook file
        ebook_path.touch()

        main(ebook_path, output_path)

        mock_read.assert_called_once_with(ebook_path)
        mock_split.assert_called_once()
        mock_generate.assert_called_once()
        mock_save.assert_called_once()

        # Clean up the dummy file
        ebook_path.unlink()

if __name__ == "__main__":
    unittest.main()
