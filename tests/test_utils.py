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
    # 你好=2 (core), 世界=1 (hapax). Core covers 66%, hapax needed for 98%.
    assert set(words) == {"你好", "世界"}


def test_extract_vocabulary_with_comprehension() -> None:
    text = "你好世界你好我们我们我们"
    words = extract_vocabulary(text, comprehension_pct=0.5)
    # 我们=3 alone covers 50% (3/6). Cutoff=1, only 我们 returned.
    assert words == ["我们"]


@patch("builtins.print")
def test_extract_vocabulary_with_verbose(mock_print: MagicMock) -> None:
    text = "你好世界你好我们我们我们"
    extract_vocabulary(text, comprehension_pct=0.8, verbose=True)
    # freq: 我们=3, 你好=2, 世界=1, total=6. 80% cutoff at 你好(freq=2).
    mock_print.assert_any_call("Total Chinese tokens: 6")
    mock_print.assert_any_call("Unique words: 3")
    mock_print.assert_any_call("Words at cutoff frequency: 2")
    mock_print.assert_any_call("Target coverage reached: 83.33%")
    mock_print.assert_any_call("Total words to learn: 2")


def test_extract_vocabulary_with_stop_words(tmp_path):
    text = "你好世界你好，我们"
    stop_words_path = tmp_path / "stop_words.txt"
    stop_words_path.write_text("你好\n我们")
    words = extract_vocabulary(text, stop_words_path=str(stop_words_path))
    assert words == ["世界"]
