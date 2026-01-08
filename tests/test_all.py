import json
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ebooklib import epub

from main import (
    create_flashcards,
    extract_vocabulary,
    main,
    read_epub,
    validate_flashcard,
    convert_pinyin
)

def create_test_epub(file_path):
    book = epub.EpubBook()
    book.set_identifier('id123456')
    book.set_title('Test Book')
    book.set_language('zh')
    book.add_author('Author')

    c1 = epub.EpubHtml(title='第一章', file_name='chap_1.xhtml', lang='zh')
    c1.content=u'<html><body><h1>第一章</h1><p>这是第一章。</p></body></html>'
    c2 = epub.EpubHtml(title='第二章', file_name='chap_2.xhtml', lang='zh')
    c2.content=u'<html><body><h1>第二章</h1><p>这是第二章。</p></body></html>'
    c3 = epub.EpubHtml(title='第三章', file_name='chap_3.xhtml', lang='zh')
    c3.content=u'<html><body><h1>第三章</h1><p>这是第三章。</p></body></html>'

    for c in [c1, c2, c3]: book.add_item(c)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ['nav', c1, c2, c3]
    epub.write_epub(file_path, book, {})

@pytest.fixture
def test_book_path(tmp_path):
    path = tmp_path / "test_book.epub"
    create_test_epub(str(path))
    return str(path)

# Test read_epub
def test_read_epub(test_book_path) -> None:
    content = read_epub(test_book_path)
    assert isinstance(content, str)
    assert "第一章" in content
    assert content.find("第一章") < content.find("第二章") < content.find("第三章")

# Test extract_vocabulary
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

# Test validation
def test_validate_flashcard():
    card = pd.Series({
        "hanzi": "你好", "pinyin": "ní hǎo", "pinyinnumbered": "ni2 hao3",
        "definition": "hello", "partofspeech": "greeting",
        "sentencehanzi": "你好吗？", "sentencepinyin": "Nǐ hǎo ma?", "sentencetranslation": "How are you?"
    })
    assert validate_flashcard(card, "你好") is True
    assert validate_flashcard(card.drop("pinyin"), "你好") is False

def test_validate_flashcard_nu_er():
    card = pd.Series({
        "hanzi": "女儿", "pinyin": "nǚ ér", "pinyinnumbered": "nv3 er2",
        "definition": "daughter", "partofspeech": "noun",
        "sentencehanzi": "这是我的女儿。", "sentencepinyin": "Zhè shì wǒ de nǚ'ér.", "sentencetranslation": "This is my daughter."
    })
    assert validate_flashcard(card, "女儿") is True

# Test Pinyin conversion
def test_convert_pinyin():
    assert convert_pinyin("hao3") == "hǎo"
    assert convert_pinyin("nv3") == "nǚ"
    assert convert_pinyin("Lü4") == "Lǜ"

# Test Batching and Shuffling
@patch("random.shuffle")
@patch("google.genai.Client")
def test_create_flashcards_shuffles_and_preserves_order(mock_client, mock_shuffle, tmp_path):
    words = ["你好", "世界"]
    mock_client.return_value.models.generate_content.return_value = MagicMock(text=json.dumps([
        {"hanzi": "你好", "pinyin": "ní hǎo", "pinyinnumbered": "ni2 hao3", "definition": "hello", "partofspeech": "noun",
         "sentencehanzi": "你好世界", "sentencepinyin": "Ní hǎo shì jiè.", "sentencetranslation": "Hello world."},
        {"hanzi": "世界", "pinyin": "shì jiè", "pinyinnumbered": "shi4 jie4", "definition": "world", "partofspeech": "noun",
         "sentencehanzi": "你好世界", "sentencepinyin": "Ní hǎo shì jiè.", "sentencetranslation": "Hello world."}
    ]))
    flashcards = create_flashcards(words, initial_batch_size=2, cache_dir=str(tmp_path))
    mock_shuffle.assert_called_once()
    assert set(flashcards["hanzi"].tolist()) == set(words)

@patch("google.genai.Client")
def test_batch_size_doubles_on_success(mock_client, tmp_path):
    words = [f"word{i}" for i in range(4)]
    def side_effect(*args, **kwargs):
        text = kwargs['contents'][-1].parts[0].text
        batch = text.split('..')
        return MagicMock(text=json.dumps([
            {"hanzi": w, "pinyin": "hǎo", "pinyinnumbered": "hao3", "definition": "d", "partofspeech": "n", 
             "sentencehanzi": f"s {w}", "sentencepinyin": "sp", "sentencetranslation": "st"} for w in batch
        ]))
    mock_client.return_value.models.generate_content.side_effect = side_effect
    with patch("builtins.print") as mock_print:
        create_flashcards(words, initial_batch_size=2, cache_dir=str(tmp_path), verbose=True)
        # Check if batch size increase was logged
        increase_calls = [c for c in mock_print.mock_calls if "increasing batch size" in str(c)]
        assert len(increase_calls) > 0

# Test Caching
@patch("google.genai.Client")
def test_create_flashcards_with_caching(mock_client, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    word = "你好"
    card = {"hanzi": word, "pinyin": "ní hǎo", "pinyinnumbered": "ni2 hao3", "definition": "h", "partofspeech": "n", 
            "sentencehanzi": "你好吗？", "sentencepinyin": "sp", "sentencetranslation": "st"}
    mock_client.return_value.models.generate_content.return_value = MagicMock(text=json.dumps([card]))
    
    create_flashcards([word], initial_batch_size=1, cache_dir=str(cache_dir))
    assert mock_client.return_value.models.generate_content.call_count == 1
    
    create_flashcards([word], initial_batch_size=1, cache_dir=str(cache_dir))
    assert mock_client.return_value.models.generate_content.call_count == 1

# Test Main Vocab Only
def test_main_vocab_only(tmp_path, test_book_path):
    output_path = tmp_path / "output.txt"
    with patch("sys.argv", ["main.py", test_book_path, str(output_path), "--vocab-only"]):
        main()
    assert output_path.exists()
    assert "第一章" in output_path.read_text()

# Test flashcards-only
@patch("main.create_flashcards")
def test_flashcards_only(mock_create, tmp_path):
    vocab = tmp_path / "v.txt"
    vocab.write_text("你好")
    out = tmp_path / "f.tsv"
    mock_create.return_value = pd.DataFrame([{"hanzi": "你好", "pinyin": "p", "pinyinnumbered": "pn", "definition": "d", 
                                             "partofspeech": "n", "sentencehanzi": "s", "sentencepinyin": "sp", "sentencetranslation": "st"}])
    with patch("sys.argv", ["main.py", str(vocab), str(out), "--flashcards-only"]):
        main()
    assert out.exists()
    assert "你好" in out.read_text()

