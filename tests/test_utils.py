from unittest.mock import MagicMock, patch

from utils import extract_vocabulary, read_epub


def test_read_epub(test_book_path) -> None:
    content = read_epub(test_book_path)
    assert isinstance(content, str)
    assert "第一章" in content
    assert (
        content.find("第一章")
        < content.find("第二章")
        < content.find("第三章")
    )


def test_extract_vocabulary() -> None:
    text = "你好世界你好, this is a test"
    words = extract_vocabulary(text)
    assert set(words) == {"你好", "世界"}


def test_extract_vocabulary_with_min_freq() -> None:
    text = "你好世界你好我们我们我们"
    words = extract_vocabulary(text, min_freq=2)
    assert set(words) == {"你好", "我们"}


@patch("builtins.print")
def test_extract_vocabulary_with_verbose(mock_print: MagicMock) -> None:
    text = "你好世界你好我们我们我们"
    extract_vocabulary(text, min_freq=3, verbose=True)
    mock_print.assert_any_call("Vocabulary size with min_freq=1: 3")


def test_extract_vocabulary_with_stop_words(tmp_path):
    text = "你好世界你好，我们"
    stop_words_path = tmp_path / "stop_words.txt"
    stop_words_path.write_text("你好\n我们")
    words = extract_vocabulary(text, stop_words_path=str(stop_words_path))
    assert words == ["世界"]
