import argparse
import os
from collections import Counter
from io import StringIO
from typing import Dict, List

from dotenv import load_dotenv
import ebooklib
import jieba
import pandas as pd
from bs4 import BeautifulSoup
from ebooklib import epub
from litellm import completion
from tqdm import tqdm

load_dotenv()

with open("src/system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


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


def extract_vocabulary(
    text: str,
    stop_words_path: str = None,
    min_freq: int = 1,
    verbose: bool = False,
) -> List[str]:
    """
    Extracts words from a string of text, filters them, and returns a list of words
    that meet the minimum frequency requirement.

    Parameters
    ----------
    text : str
        The text to extract words from.
    stop_words_path : str, optional
        The path to a file containing comma-separated stop words, by default None.
    min_freq : int, optional
        The minimum frequency for a word to be included, by default 1.
    verbose : bool, optional
        Whether to print verbose output, by default False.

    Returns
    -------
    List[str]
        A list of unique words from the text that meet the frequency requirement.
    """
    if stop_words_path:
        with open(stop_words_path, "r") as f:
            stop_words = set(f.read().split(','))
    else:
        stop_words = set()

    words = [word for word in jieba.cut(text) if word not in stop_words]

    def is_all_chinese(word):
        for char in word:
            if not "\u4e00" <= char <= "\u9fff":
                return False
        return True

    words = [word for word in words if is_all_chinese(word)]
    word_counts = Counter(words)

    if verbose:
        for i in range(1, min_freq + 1):
            vocab_size = len([word for word, count in word_counts.items() if count >= i])
            print(f"Vocabulary size with min_freq={i}: {vocab_size}")

    return [word for word, count in word_counts.items() if count >= min_freq]


def create_flashcards(
    words: List[str],
    batch_size: int = 100,
    retries: int = 3,
    model: str = "gemini-pro",
    verbose: bool = False,
    cache_dir: str = ".flashcard_cache",
) -> pd.DataFrame:
    """
    Creates flashcards from a list of words with caching and retries.

    Parameters
    ----------
    words : List[str]
        The list of words to create flashcards from.
    batch_size : int, optional
        The number of words to process in each batch, by default 100.
    retries : int, optional
        The number of times to retry a word if it fails validation, by default 3.
    model : str, optional
        The name of the LLM model to use, by default "gemini-pro".
    verbose : bool, optional
        Whether to print verbose output, by default False.
    cache_dir : str, optional
        The directory to cache flashcards, by default ".flashcard_cache".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the flashcards, in the same order as the input words.
    """
    os.makedirs(cache_dir, exist_ok=True)
    flashcards_map: Dict[str, pd.Series] = {}
    words_to_process = []

    for word in words:
        cache_path = os.path.join(cache_dir, f"{word}.json")
        if os.path.exists(cache_path):
            flashcards_map[word] = pd.read_json(cache_path, typ="series")
        else:
            words_to_process.append(word)

    pbar = (
        tqdm(total=len(words), desc="Creating flashcards")
        if verbose
        else None
    )
    if pbar:
        pbar.update(len(flashcards_map))

    retry_counts = {word: 0 for word in words_to_process}

    while words_to_process:
        batch = words_to_process[:batch_size]
        words_to_process = words_to_process[batch_size:]

        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ",".join(batch)},
                ],
            )
            response_df = pd.read_csv(
                StringIO(response.choices[0].message.content), sep="\t", header=0
            )
        except Exception as e:
            if verbose:
                print(f"Error processing batch: {e}")
            for word in batch:
                retry_counts[word] += 1
                if retry_counts[word] < retries:
                    words_to_process.append(word)
                elif verbose:
                    print(f"Failed to create valid flashcard for word: {word}")
                    if pbar:
                        pbar.update(1)
            continue

        response_map = {row["hanzi"]: row for _, row in response_df.iterrows()}

        for word in batch:
            if word in response_map and validate_flashcard(response_map[word], word):
                flashcard = response_map[word]
                flashcards_map[word] = flashcard
                cache_path = os.path.join(cache_dir, f"{word}.json")
                flashcard.to_json(cache_path)
                if pbar:
                    pbar.update(1)
            else:
                retry_counts[word] += 1
                if retry_counts[word] < retries:
                    words_to_process.append(word)
                elif verbose:
                    print(f"Failed to create valid flashcard for word: {word}")
                    if pbar:
                        pbar.update(1)

    if pbar:
        pbar.close()

    final_flashcards = [flashcards_map[word] for word in words if word in flashcards_map]
    if not final_flashcards:
        return pd.DataFrame()
    
    return pd.concat([card.to_frame().T for card in final_flashcards], ignore_index=True)



def validate_flashcard(flashcard: pd.Series, word: str) -> bool:
    """
    Validates a single flashcard.

    Parameters
    ----------
    flashcard : pd.Series
        The flashcard to validate.
    word : str
        The word that the flashcard was generated from.

    Returns
    -------
    bool
        True if the flashcard is valid, False otherwise.
    """
    expected_columns = [
        "hanzi",
        "pinyin",
        "definition",
        "partofspeech",
        "sentencehanzi",
        "sentencepinyin",
        "sentencetranslation",
    ]
    if not all(col in flashcard.index for col in expected_columns):
        return False

    if flashcard.isnull().any():
        return False

    if flashcard["hanzi"] != word:
        return False

    if word not in flashcard["sentencehanzi"]:
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
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="The minimum frequency for a word to be included.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output.",
    )
    args = parser.parse_args()

    content = read_epub(args.ebook_path)
    words = extract_vocabulary(
        content, args.stop_words, args.min_frequency, args.verbose
    )

    if args.vocab_only:
        with open(args.output_path, "w") as f:
            f.write("\n".join(words))
        return

    flashcards = create_flashcards(
        words, args.batch_size, args.retries, args.model, args.verbose
    )
    save_flashcards(flashcards, args.output_path)


if __name__ == "__main__":
    main()


