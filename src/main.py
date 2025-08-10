import argparse
from io import StringIO
from typing import List

import ebooklib
import jieba
import pandas as pd
from bs4 import BeautifulSoup
from ebooklib import epub
from litellm import completion

try:
    with open("src/promot.txt", "r") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are a helpful assistant that creates flashcards for language learning."


def read_epub(file_path: str) -> str:
    """
    Reads an EPUB file and returns its text content.

    Parameters
    ----------
    file_path : str
        The path to the EPUB file.

    Returns
    -------
    str
        The text content of the EPUB file.
    """
    book = epub.read_epub(file_path)
    content = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        content.append(soup.get_text())
    return "\n".join(content)


def extract_words(text: str, stop_words_path: str = None) -> List[str]:
    """
    Extracts unique words from a string of text.

    Parameters
    ----------
    text : str
        The text to extract words from.
    stop_words_path : str, optional
        The path to a file containing comma-separated stop words, by default None.

    Returns
    -------
    List[str]
        A list of unique words from the text.
    """
    if stop_words_path:
        with open(stop_words_path, "r") as f:
            stop_words = set(f.read().split(','))
    else:
        stop_words = set()

    seen = set()
    return [
        x
        for x in jieba.cut(text)
        if not (x in seen or seen.add(x)) and x not in stop_words
    ]


def create_flashcards(
    words: List[str],
    batch_size: int = 100,
    retries: int = 3,
    model: str = "gemini-pro",
) -> pd.DataFrame:
    """
    Creates flashcards from a list of words.

    Parameters
    ----------
    words : List[str]
        The list of words to create flashcards from.
    batch_size : int, optional
        The number of words to process in each batch, by default 100.
    retries : int, optional
        The number of times to retry a batch if it fails validation, by default 3.
    model : str, optional
        The name of the LLM model to use, by default "gemini-pro".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the flashcards.
    """
    all_flashcards = []
    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        for _ in range(retries):
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ",".join(batch)},
                ],
            )
            flashcards = pd.read_csv(
                StringIO(response.choices[0].message.content), sep="\t", header=0
            )
            if validate_flashcards_batch(flashcards, batch):
                all_flashcards.append(flashcards)
                break
        else:
            raise ValueError(f"Failed to create valid flashcards for batch: {batch}")
    return pd.concat(all_flashcards, ignore_index=True)


def validate_flashcards_batch(flashcards: pd.DataFrame, batch: List[str]) -> bool:
    """
    Validates a batch of flashcards.

    Parameters
    ----------
    flashcards : pd.DataFrame
        The DataFrame of flashcards to validate.
    batch : List[str]
        The list of words that the flashcards were generated from.

    Returns
    -------
    bool
        True if the batch is valid, False otherwise.
    """
    if len(flashcards) != len(batch):
        return False

    expected_columns = [
        "hanzi",
        "pinyin",
        "definition",
        "partofspeech",
        "sentencehanzi",
        "sentencepinyin",
        "sentencetranslation",
    ]
    if not all(col in flashcards.columns for col in expected_columns):
        return False

    if flashcards.isnull().values.any():
        return False

    if not all(flashcards["hanzi"] == batch):
        return False

    if not all(
        word in sentence
        for word, sentence in zip(flashcards["hanzi"], flashcards["sentencehanzi"])
    ):
        return False

    return True


def save_flashcards(flashcards: pd.DataFrame, file_path: str) -> None:
    """
    Saves a DataFrame of flashcards to a file.

    Parameters
    ----------
    flashcards : pd.DataFrame
        The DataFrame of flashcards to save.
    file_path : str
        The path to save the flashcards to.
    """
    flashcards.to_csv(file_path, sep="\t", index=False, header=False)

def main() -> None:
    """
    The main function for the script.
    """
    parser = argparse.ArgumentParser(
        description="Create Anki flashcards from a Chinese ebook."
    )
    parser.add_argument("ebook_path", type=str, help="The path to the ebook file.")
    parser.add_argument(
        "output_path", type=str, help="The path to save the flashcards."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="The number of words to process in each batch.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="The number of times to retry a batch if it fails validation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-pro",
        help="The name of the LLM model to use.",
    )
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="Only extract the vocabulary and save it to the output file.",
    )
    parser.add_argument(
        "--stop-words",
        type=str,
        default=None,
        help="The path to a file containing comma-separated stop words.",
    )
    args = parser.parse_args()

    content = read_epub(args.ebook_path)
    words = extract_words(content, args.stop_words)

    if args.vocab_only:
        with open(args.output_path, "w") as f:
            f.write("\n".join(words))
        return

    flashcards = create_flashcards(words, args.batch_size, args.retries, args.model)
    save_flashcards(flashcards, args.output_path)


if __name__ == "__main__":
    main()