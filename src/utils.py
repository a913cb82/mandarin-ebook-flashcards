import os
from collections import Counter

import ebooklib
import jieba
from bs4 import BeautifulSoup
from ebooklib import epub


def read_epub(file_path: str) -> str:
    """Reads an EPUB file and returns its text content."""
    book = epub.read_epub(file_path)
    content = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        content.append(soup.get_text())
    return "\n".join(content)


def extract_vocabulary(
    text: str, stop_words_path: str | None = None, min_freq: int = 1, verbose: bool = False
) -> list[str]:
    """Extracts Chinese vocabulary from text using jieba."""
    stop_words = set()
    if stop_words_path and os.path.exists(stop_words_path):
        with open(stop_words_path) as f:
            stop_words = set(f.read().splitlines())

    words = [
        w
        for w in jieba.cut(text)
        if all("\u4e00" <= c <= "\u9fff" for c in w) and w not in stop_words
    ]
    counts = Counter(words)

    if verbose:
        for i in range(1, min_freq + 1):
            v_size = len([w for w, c in counts.items() if c >= i])
            print(f"Vocabulary size with min_freq={i}: {v_size}")

    return [w for w, c in counts.items() if c >= min_freq]
