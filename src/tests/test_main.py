import pandas as pd
from main import main
import os

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
