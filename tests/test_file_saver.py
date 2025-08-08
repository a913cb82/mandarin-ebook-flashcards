
import unittest
import pandas as pd
from pathlib import Path
from src.file_saver import save_dataframe_to_tsv

class TestFileSaver(unittest.TestCase):

    def test_save_dataframe_to_tsv(self):
        data = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data)
        output_path = Path("test_output.tsv")

        save_dataframe_to_tsv(df, output_path)

        self.assertTrue(output_path.exists())

        # Clean up the dummy file
        output_path.unlink()

if __name__ == "__main__":
    unittest.main()
