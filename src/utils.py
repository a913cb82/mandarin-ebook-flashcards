import os
from collections import Counter
from pathlib import Path

import ebooklib
import jieba
from bs4 import BeautifulSoup
from ebooklib import epub

SUBTLEX_HEADER_ROWS = 3
SUBTLEX_COL_WORD = 0
SUBTLEX_COL_FREQ = 1


def load_subtlex_global_freqs(path: str | Path) -> dict[str, int]:
    """Load SUBTLEX-CH word frequencies into a dict[str, int].

    Returns {word: WCount} from the GBK-encoded SUBTLEX-CH-WF file.
    """
    freqs: dict[str, int] = {}
    with open(path, encoding="gbk") as f:
        for _ in range(SUBTLEX_HEADER_ROWS):
            next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) > SUBTLEX_COL_FREQ:
                word = parts[SUBTLEX_COL_WORD]
                try:
                    freq = int(parts[SUBTLEX_COL_FREQ])
                except ValueError:
                    continue
                freqs[word] = freq
    return freqs


def read_epub(file_path: str) -> str:
    """Reads an EPUB file and returns its text content."""
    book = epub.read_epub(file_path)
    content = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        content.append(soup.get_text())
    return "\n".join(content)


def extract_vocabulary(
    text: str,
    stop_words_path: str | None = None,
    comprehension_pct: float = 0.98,
    verbose: bool = False,
    global_freqs: dict[str, int] | None = None,
) -> list[str]:
    """Extracts Chinese words needed to reach a target comprehension level.

    Words are sorted by frequency descending. Within the same frequency
    level, words are ordered by global frequency (if available), then by
    the order first seen in the text.
    """
    stop_words = set()
    if stop_words_path and os.path.exists(stop_words_path):
        with open(stop_words_path) as f:
            stop_words = set(f.read().splitlines())

    if global_freqs is None:
        global_freqs = {}

    words = [
        w
        for w in jieba.cut(text)
        if all("\u4e00" <= c <= "\u9fff" for c in w) and w not in stop_words
    ]
    total = len(words)

    if total == 0 or comprehension_pct == 0.0:
        return []

    counts = Counter(words)

    # Group words by frequency so we can sort ties within each bucket
    by_freq: dict[int, list[str]] = {}
    for w, c in counts.items():
        by_freq.setdefault(c, []).append(w)

    for _freq, bucket in by_freq.items():
        bucket.sort(key=lambda w: global_freqs.get(w, 0), reverse=True)

    sorted_words = [
        w
        for _, bucket in sorted(by_freq.items(), reverse=True)
        for w in bucket
    ]

    covered = 0
    cutoff = 0
    for i, w in enumerate(sorted_words):
        covered += counts[w]
        if covered / total >= comprehension_pct:
            cutoff = i + 1
            break

    needed = sorted_words[:cutoff]

    if verbose:
        print(f"Total Chinese tokens: {total}")
        print(f"Unique words: {len(counts)}")
        print(f"Words at cutoff frequency: {counts[sorted_words[cutoff - 1]]}")
        print(f"Target coverage reached: {covered / total:.2%}")
        print(f"Total words to learn: {len(needed)}")

    return needed
