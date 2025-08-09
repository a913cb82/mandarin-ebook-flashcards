import pandas as pd
from main import main, read_epub, extract_words, create_flashcards, save_flashcards
import os
import pytest
from unittest.mock import patch, MagicMock

def mock_create_flashcards(words):
    data = {
        'hanzi': ['你好', '世界', '我们', '在', '打包', '东西'],
        'pinyin': ['nǐ hǎo', 'shìjiè', 'wǒmen', 'zài', 'dǎbāo', 'dōngxi'],
        'definition': ['hello', 'world', 'we/us', 'at/in', 'to pack', 'thing/stuff'],
        'partofspeech': ['greeting', 'noun', 'pronoun', 'preposition', 'verb', 'noun'],
        'sentencehanzi': ['你好吗？', '世界是圆的。', '我们是朋友。', '我在家。', '我在打包行李。', '这是什么东西？'],
        'sentencepinyin': ['Nǐ hǎo ma?', 'Shìjiè shì yuán de.', 'Wǒmen shì péngyǒu.', 'Wǒ zài jiā.', 'Wǒ zài dǎbāo xínglǐ.', 'Zhè shì shénme dōngxi?'],
        'sentencetranslation': ['How are you?', 'The world is round.', 'We are friends.', 'I am at home.', 'I am packing luggage.', 'What is this thing?']
    }
    return pd.DataFrame(data)

def test_read_epub():
    content = read_epub('src/tests/test_book.epub')
    assert isinstance(content, str)
    assert "第一章" in content
    assert "第二章" in content
    assert "第三章" in content
    assert content.find("第一章") < content.find("第二章") < content.find("第三章")

def test_extract_words():
    text = "你好世界"
    words = extract_words(text)
    assert isinstance(words, list)
    assert "你好" in words
    assert "世界" in words

@patch('main.completion')
def test_create_flashcards(mock_completion):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "hanzi\tpinyin\tdefinition\n你好\tnǐ hǎo\thello"
    mock_completion.return_value = mock_response

    words = ["你好"]
    flashcards = create_flashcards(words)
    assert isinstance(flashcards, pd.DataFrame)
    assert "hanzi" in flashcards.columns
    assert "pinyin" in flashcards.columns
    assert "definition" in flashcards.columns
    assert flashcards.iloc[0]['hanzi'] == "你好"

def test_save_flashcards():
    data = {'col1': ['a', 'b'], 'col2': ['c', 'd']}
    df = pd.DataFrame(data)
    output_path = 'src/tests/output.tsv'
    save_flashcards(df, output_path)

    assert os.path.exists(output_path)

    with open(output_path, 'r') as f:
        content = f.read()
        assert "a\tc" in content
        assert "b\td" in content

    os.remove(output_path)


def test_end_to_end(monkeypatch, capsys):
    monkeypatch.setattr('main.create_flashcards', mock_create_flashcards)

    epub_path = 'src/tests/test_book.epub'
    output_path = 'src/tests/output.tsv'

    import sys
    sys.argv = ['main.py', epub_path, output_path]

    main()

    assert os.path.exists(output_path)

    with open(output_path, 'r') as f:
        content = f.read()
        assert '你好' in content

    # Clean up the output file
    os.remove(output_path)