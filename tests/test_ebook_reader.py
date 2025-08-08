
import unittest
from pathlib import Path
from src.ebook_reader import read_ebook

class TestEbookReader(unittest.TestCase):

    def test_read_ebook(self):
        # Create a dummy ebook file for testing
        ebook_content = "你好世界"
        ebook_path = Path("test_ebook.txt")
        ebook_path.write_text(ebook_content, encoding="utf-8")

        self.assertEqual(read_ebook(ebook_path), ebook_content)

        # Clean up the dummy file
        ebook_path.unlink()

if __name__ == "__main__":
    unittest.main()
