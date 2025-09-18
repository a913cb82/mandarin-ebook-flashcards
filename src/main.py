import argparse
import json
import os
import toml
import unicodedata
from collections import Counter
from io import StringIO
from typing import Dict, List

from dotenv import load_dotenv
import ebooklib
import jieba
import pandas as pd
from bs4 import BeautifulSoup
from ebooklib import epub
import google.generativeai as genai
from tqdm import tqdm

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

with open("src/system_prompt.toml", "r") as f:
    prompt_data = toml.load(f)
    SYSTEM_PROMPT = prompt_data["system_prompt"]
    FEW_SHOT_EXAMPLES = prompt_data["examples"]

EXPECTED_COLUMNS = [
    "hanzi",
    "pinyin",
    "pinyinnumbered",
    "definition",
    "partofspeech",
    "sentencehanzi",
    "sentencepinyin",
    "sentencetranslation",
]


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
        The path to a file containing newline-separated stop words, by default None.
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
            stop_words = set(f.read().splitlines())
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
    cache_dir: str,
    initial_batch_size: int = 100,
    batch_size_multiplier: float = 2.0,
    retries: int = 3,
    model: str = "gemini-pro",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Creates flashcards from a list of words with caching and retries.

    Parameters
    ----------
    words : List[str]
        The list of words to create flashcards from.
    initial_batch_size : int, optional
        The initial number of words to process in each batch, by default 100.
    batch_size_multiplier : float, optional
        The multiplier to use for adjusting the batch size, by default 2.0.
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

    pbar = (
        tqdm(total=len(words), desc="Creating flashcards")
        if verbose
        else None
    )

    for word in words:
        cache_path = os.path.join(cache_dir, f"{word}.json")
        if os.path.exists(cache_path):
            flashcards_map[word] = pd.read_json(cache_path, typ="series")
            if pbar:
                pbar.update(1)
        else:
            words_to_process.append(word)

    retry_counts = {word: 0 for word in words_to_process}

    flashcard_schema = {
        "type": "object",
        "properties": {
            "hanzi": {"type": "string"},
            "pinyin": {"type": "string"},
            "pinyinnumbered": {"type": "string"},
            "definition": {"type": "string"},
            "partofspeech": {"type": "string"},
            "sentencehanzi": {"type": "string"},
            "sentencepinyin": {"type": "string"},
            "sentencetranslation": {"type": "string"},
        },
        "required": [
            "hanzi",
            "pinyin",
            "pinyinnumbered",
            "definition",
            "partofspeech",
            "sentencehanzi",
            "sentencepinyin",
            "sentencetranslation",
        ],
    }

    messages = []
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "parts": [example["input"]]})
        messages.append({"role": "model", "parts": [example["output"]]})

    llm_model = genai.GenerativeModel(
        model,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "array",
                "items": flashcard_schema,
            },
        },
        system_instruction=SYSTEM_PROMPT,
        safety_settings={
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
    )

    batch_size = initial_batch_size

    while words_to_process:
        batch = words_to_process[:batch_size]
        words_to_process = words_to_process[batch_size:]
        response = None

        try:
            response = llm_model.generate_content(messages + [{"role": "user", "parts": ["..".join(batch)]}])
            response_df = pd.DataFrame(
                json.loads(response.text)
            )
        except Exception as e:
            if verbose:
                print(f"Error processing batch: {e}")
                print(f"{batch=}")
                if response:
                    print(f"{response.prompt_feedback=}")
                    if response.parts:
                        print(f"{response.parts=}")
            for word in batch:
                retry_counts[word] += 1
                if retry_counts[word] < retries:
                    words_to_process.append(word)
                else:
                    if verbose:
                        print(f"Failed to create valid flashcard for word: {word}")
                    if pbar:
                        pbar.update(1)
            batch_size = max(1, int(batch_size / batch_size_multiplier))
            if verbose:
                print(f"Batch failed, reducing batch size to {batch_size}")
            continue

        response_map = {row["hanzi"]: row for _, row in response_df.iterrows()}
        
        succeeded_count = 0
        for word in batch:
            if word in response_map and validate_flashcard(response_map[word], word):
                succeeded_count += 1
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
                else:
                    if verbose:
                        print(f"Failed to create valid flashcard for word: {word}")
                    if pbar:
                        pbar.update(1)
        
        if succeeded_count == len(batch):
            batch_size = int(batch_size * batch_size_multiplier)
            if verbose:
                print(f"Batch succeeded, increasing batch size to {batch_size}")

    if pbar:
        pbar.close()

    final_flashcards = [flashcards_map[word] for word in words if word in flashcards_map]
    if not final_flashcards:
        return pd.DataFrame()
    
    return pd.concat([card.to_frame().T for card in final_flashcards], ignore_index=True)


def are_pinyins_consistent(tone_marked_pinyin: str, numbered_pinyin: str) -> bool:
    """
    Checks if the tone-marked pinyin and numbered pinyin are consistent.

    Parameters
    ----------
    tone_marked_pinyin : str
        The pinyin with tone marks.
    numbered_pinyin : str
        The pinyin with numbers.

    Returns
    -------
    bool
        True if the pinyins are consistent, False otherwise.
    """

    def convert_s(s: str) -> str:
        s = unicodedata.normalize("NFD", s)
        res = []
        tone = "5"
        for c in s:
            if c == "ü":
                res.append("v")
            elif c == " ":
                res.append(tone)
                res.append(" ")
                tone = "5"
            elif c == "\u0304":
                tone = "1"
            elif c == "\u0301":
                tone = "2"
            elif c == "\u030c":
                tone = "3"
            elif c == "\u0300":
                tone = "4"
            elif not unicodedata.combining(c):
                res.append(c)
        res.append(tone)
        return "".join(res)

    tone_marked_parts = [p.strip() for p in tone_marked_pinyin.replace(";", "|").split("|")]
    numbered_parts = [p.strip() for p in numbered_pinyin.replace(";", "|").split("|")]

    if len(tone_marked_parts) != len(numbered_parts):
        return False

    for tm_part, n_part in zip(tone_marked_parts, numbered_parts):
        if convert_s(tm_part) != n_part:
            return False

    return True


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
    if not all(col in flashcard.index for col in EXPECTED_COLUMNS):
        return False

    if flashcard.isnull().any():
        return False

    if flashcard["hanzi"] != word:
        return False

    if word not in flashcard["sentencehanzi"]:
        return False

    if not are_pinyins_consistent(flashcard["pinyin"], flashcard["pinyinnumbered"]):
        return False

    pinyin_parts = flashcard["pinyin"].split("|")
    pinyin_numbered_parts = flashcard["pinyinnumbered"].split("|")
    definition_parts = flashcard["definition"].split("|")

    if len(pinyin_parts) != len(pinyin_numbered_parts) or len(pinyin_parts) != len(definition_parts):
        return False

    for pinyin_part, pinyin_numbered_part in zip(pinyin_parts, pinyin_numbered_parts):
        if len(pinyin_part.split(";")) != len(pinyin_numbered_part.split(";")):
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
    if not flashcards.empty:
        flashcards = flashcards[EXPECTED_COLUMNS]
    flashcards.to_csv(file_path, sep="\t", index=False, header=False)


def main() -> None:
    """
    The main function for the script.
    """
    parser = argparse.ArgumentParser(
        description="Create Anki flashcards from a Chinese ebook."
    )
    parser.add_argument("input_path", type=str, help="The path to the input file (ebook or vocab list).")
    parser.add_argument(
        "output_path", type=str, help="The path to save the flashcards."
    )
    parser.add_argument(
        "--initial-batch-size",
        type=int,
        default=100,
        help="The initial number of words to process in each batch.",
    )
    parser.add_argument(
        "--batch-size-multiplier",
        type=float,
        default=2.0,
        help="The multiplier to use for adjusting the batch size.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="The number of times to retry a flashcard if it fails validation.",
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
        help="The path to a file containing newline-separated stop words.",
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
    parser.add_argument(
        "--flashcards-only",
        action="store_true",
        help="Treat the input file as a vocab list and generate flashcards.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="The directory to cache flashcards.",
    )
    args = parser.parse_args()

    if args.flashcards_only:
        with open(args.input_path, "r") as f:
            words = [line.strip() for line in f]
    else:
        content = read_epub(args.input_path)
        words = extract_vocabulary(
            content, args.stop_words, args.min_frequency, args.verbose
        )

    if args.vocab_only:
        with open(args.output_path, "w") as f:
            f.write("\n".join(words))
        return

    flashcards = create_flashcards(
        words,
        args.cache_dir,
        args.initial_batch_size,
        args.batch_size_multiplier,
        args.retries,
        args.model,
        args.verbose,
    )
    save_flashcards(flashcards, args.output_path)


if __name__ == "__main__":
    main()
