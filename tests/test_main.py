from unittest.mock import patch

import pandas as pd

from main import main


def test_main_vocab_only(tmp_path, test_book_path):
    output_path = tmp_path / "output.txt"
    with patch(
        "sys.argv",
        ["main.py", test_book_path, str(output_path), "--vocab-only"],
    ):
        main()
    assert output_path.exists()
    assert "第一章" in output_path.read_text()


@patch("main.create_flashcards")
def test_flashcards_only(mock_create, tmp_path):
    vocab = tmp_path / "v.txt"
    vocab.write_text("你好")
    out = tmp_path / "f.tsv"
    mock_create.return_value = pd.DataFrame(
        [
            {
                "hanzi": "你好",
                "pinyin": "p",
                "pinyinnumbered": "pn",
                "definition": "d",
                "partofspeech": "n",
                "sentencehanzi": "s",
                "sentencepinyin": "sp",
                "sentencetranslation": "st",
            }
        ]
    )
    with patch(
        "sys.argv", ["main.py", str(vocab), str(out), "--flashcards-only"]
    ):
        main()
    assert out.exists()
    assert "你好" in out.read_text()
