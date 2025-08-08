
import jieba

def split_text_into_words(text: str) -> list[str]:
    """Splits a string of text into a list of words."""
    return list(jieba.cut(text))
