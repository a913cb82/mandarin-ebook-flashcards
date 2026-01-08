import argparse
import json
import os
import random
import re
import time
import toml
import unicodedata
from collections import Counter, OrderedDict
from io import StringIO
from typing import Dict, List, Optional

from dotenv import load_dotenv
import ebooklib
import jieba
import pandas as pd
from bs4 import BeautifulSoup
from ebooklib import epub
from google import genai
from google.genai import types, errors
from tqdm import tqdm

load_dotenv()
# API key is loaded from environment variable when initializing the client

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(current_dir, "system_prompt.toml")

with open(prompt_path, "r") as f:
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
    verbose: int = 0,
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

    if verbose > 0:
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
    model: str = "gemini-3-pro-preview",
    verbose: int = 0,
    cache_tokens: bool = False,
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
        if verbose > 0
        else None
    )

    for word in words:
        cache_path = os.path.join(cache_dir, f"{word}.json")
        if os.path.exists(cache_path):
            cached_card = pd.read_json(cache_path, typ="series")
            if validate_flashcard(cached_card, word, verbose=verbose):
                flashcards_map[word] = cached_card
                if pbar:
                    pbar.update(1)
            else:
                if verbose > 1:
                    print(f"Cached card for '{word}' failed validation.")
                words_to_process.append(word)
        else:
            words_to_process.append(word)

    random.shuffle(words_to_process)
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
        messages.append(types.Content(role="user", parts=[types.Part.from_text(text=example["input"])]))
        messages.append(types.Content(role="model", parts=[types.Part.from_text(text=example["output"])]))

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    cached_content_name = None
    if cache_tokens:
        cache_config = types.CreateCachedContentConfig(
            contents=messages,
        )
        # Using the model name directly, but ensure it's compatible with cache creation if needed.
        # Often models need to be prefixed with 'models/' for caching.
        cache_model = model if model.startswith("models/") else f"models/{model}"
        
        try:
             cached_content = client.caches.create(
                model=cache_model,
                config=cache_config,
            )
             cached_content_name = cached_content.name
        except Exception as e:
            if verbose > 0:
                print(f"Failed to create cache: {e}")
            # Fallback to no caching if it fails? Or raise?
            # For now, let's proceed without caching if it fails, or maybe just log.
            # But the original code didn't try/except creation.
            # Let's let it raise if it fails, as per original behavior intent.
            raise e

    batch_size = initial_batch_size
    consecutive_rate_limits = 0
    max_allowed_batch_size = float("inf")

    while words_to_process:
        batch = words_to_process[:batch_size]
        words_to_process = words_to_process[batch_size:]
        response = None

        try:
            current_messages = []
            generate_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            k: types.Schema(type=types.Type.STRING) for k in flashcard_schema["properties"]
                        },
                        required=flashcard_schema["required"]
                    )
                ),
                system_instruction=SYSTEM_PROMPT,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ]
            )

            if cache_tokens and cached_content_name:
                generate_config.cached_content = cached_content_name
                current_messages = [types.Content(role="user", parts=[types.Part.from_text(text="..".join(batch))])]
            else:
                # If not using cache, we need to send the full history
                current_messages = messages + [types.Content(role="user", parts=[types.Part.from_text(text="..".join(batch))])]

            response = client.models.generate_content(
                model=model,
                contents=current_messages,
                config=generate_config
            )
            response_df = pd.DataFrame(json.loads(response.text))
            consecutive_rate_limits = 0
        except Exception as e:
            is_rate_limit = False
            wait_time = 0
            
            if isinstance(e, errors.APIError):
                if e.code == 429:
                    is_rate_limit = True
                    # Extract wait time from details if possible
                    try:
                        for detail in e.details.get('details', []):
                            if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                retry_delay = detail.get('retryDelay', '30s')
                                wait_time = float(retry_delay.rstrip('s'))
                                break
                    except:
                        pass
            elif "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                is_rate_limit = True

            if is_rate_limit:
                consecutive_rate_limits += 1
                if wait_time == 0:
                    wait_time = (2 ** consecutive_rate_limits) + (random.randint(0, 1000) / 1000)
                
                if verbose > 0:
                    print(f"Rate limit hit (429). Waiting {wait_time:.2f}s before retrying... (Consecutive: {consecutive_rate_limits})")
                
                time.sleep(wait_time)
                # Put the batch back at the front and retry
                words_to_process = batch + words_to_process
                continue

            if verbose > 0:
                print(f"Error processing batch: {e}")
                print(f"{batch=}")
                if response:
                    # response.prompt_feedback isn't directly available in the same way in new SDK, 
                    # usually check usage_metadata or candidates finish_reason
                    try:
                        print(f"{response.usage_metadata=}")
                        print(f"{response.candidates=}")
                    except:
                        pass
            for word in batch:
                retry_counts[word] += 1
                if retry_counts[word] < retries:
                    words_to_process.append(word)
                else:
                    if verbose > 0:
                        print(f"Failed to create valid flashcard for word: {word}")
                    if pbar:
                        pbar.update(1)
            
            max_allowed_batch_size = batch_size
            batch_size = max(1, int(batch_size / batch_size_multiplier))
            if verbose > 0:
                print(f"Batch failed, reducing batch size to {batch_size}")
            continue

        response_map = {row["hanzi"]: row for _, row in response_df.iterrows()}
        
        succeeded_count = 0
        for word in batch:
            if word in response_map and validate_flashcard(
                response_map[word], word, verbose=verbose
            ):
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
                    if verbose > 0:
                        print(f"Failed to create valid flashcard for word: {word}")
                    if pbar:
                        pbar.update(1)
        
        if succeeded_count > len(batch) / 2:
            if int(batch_size * batch_size_multiplier) >= max_allowed_batch_size:
                batch_size = (batch_size + max_allowed_batch_size) // 2
                if verbose > 0:
                    print(f"Batch succeeded ({succeeded_count}/{len(batch)}), increasing batch size to {batch_size} (midpoint)")
            else:
                batch_size = int(batch_size * batch_size_multiplier)
                if verbose > 0:
                    print(f"Batch succeeded ({succeeded_count}/{len(batch)}), increasing batch size to {batch_size}")
        elif succeeded_count < len(batch) / 2:
            batch_size = max(1, int(batch_size / batch_size_multiplier))
            if verbose > 0:
                print(f"Batch failed ({succeeded_count}/{len(batch)}), reducing batch size to {batch_size}")

    if pbar:
        pbar.close()

    final_flashcards = [flashcards_map[word] for word in words if word in flashcards_map]
    if not final_flashcards:
        return pd.DataFrame()
    
    return pd.concat([card.to_frame().T for card in final_flashcards], ignore_index=True)


import logging

logging.basicConfig(level=logging.INFO)

pinyinToneMarks = {
    'a': 'āáǎà', 'e': 'ēéěè', 'i': 'īíǐì',
    'o': 'ōóǒò', 'u': 'ūúǔù', 'ü': 'ǖǘǚǜ',
    'A': 'ĀÁǍÀ', 'E': 'ĒÉĚÈ', 'I': 'ĪÍǏÌ',
    'O': 'ŌÓǑÒ', 'U': 'ŪÚǓÙ', 'Ü': 'ǕǗǙǛ'
}

def convertPinyinCallback(m):
    tone=int(m.group(3))%5
    r=m.group(1).replace('v', 'ü').replace('V', 'Ü')
    # for multple vowels, use first one if it is a/e/o, otherwise use second one
    pos=0
    if len(r)>1 and not r[0] in 'aeoAEO':
        pos=1
    if tone != 0:
        r=r[0:pos]+pinyinToneMarks[r[pos]][tone-1]+r[pos+1:]
    return r+m.group(2)

def convertPinyin(s):
    return re.sub(r'([aeiouüvÜ]{1,3})(n?g?r?)([012345])', convertPinyinCallback, s, flags=re.IGNORECASE)


def are_pinyins_consistent(tone_marked_pinyin: str, numbered_pinyin: str) -> bool:
    tone_marked_pinyin = tone_marked_pinyin.replace(";", "|")
    numbered_pinyin = numbered_pinyin.replace(";", "|")
    tone_marked_parts = [p.strip().replace("'", "") for p in tone_marked_pinyin.split("|")]
    numbered_parts = [p.strip() for p in numbered_pinyin.split("|")]

    if len(tone_marked_parts) != len(numbered_parts):
        return False

    for tm_part, n_part in zip(tone_marked_parts, numbered_parts):
        converted_n_part = convertPinyin(n_part)
        if tm_part != converted_n_part:
            return False

    return True


def are_structures_identical(str1: str, str2: str) -> bool:
    def get_structure(s: str) -> List[int]:
        return [len(part.split(';')) for part in s.split('|')]
    return get_structure(str1) == get_structure(str2)

def are_pipe_counts_equal(str1: str, str2: str) -> bool:
    return len(str1.split('|')) == len(str2.split('|'))


def validate_flashcard(flashcard: pd.Series, word: str, verbose: int = 0) -> bool:
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
        if verbose > 1:
            print(f"Flashcard for '{word}' is missing columns.")
        return False

    if flashcard.isnull().any():
        if verbose > 1:
            print(f"Flashcard for '{word}' has null values.")
        return False

    if flashcard["hanzi"] != word:
        if verbose > 1:
            print(f"Flashcard for '{word}' has mismatched hanzi: {flashcard['hanzi']}")
        return False

    if word not in flashcard["sentencehanzi"]:
        if verbose > 1:
            print(f"Word '{word}' not in sentence: {flashcard['sentencehanzi']}")
        return False

    if not are_pinyins_consistent(flashcard["pinyin"], flashcard["pinyinnumbered"]):
        if verbose > 1:
            print(f"Pinyins for '{word}' are inconsistent: {flashcard['pinyin']} vs {flashcard['pinyinnumbered']}")
        return False

    if not are_structures_identical(flashcard["pinyin"], flashcard["pinyinnumbered"]):
        if verbose > 1:
            print(f"Inconsistent separator structures for '{word}' between pinyin and pinyinnumbered")
        return False
        
    if not are_pipe_counts_equal(flashcard["pinyin"], flashcard["definition"]):
        if verbose > 1:
            print(f"Inconsistent number of parts for '{word}' between pinyin and definition")
        return False

    if not are_pipe_counts_equal(flashcard["pinyin"], flashcard["partofspeech"]):
        if verbose > 1:
            print(f"Inconsistent number of parts for '{word}' between pinyin and partofspeech")
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
        default="gemini-3-pro-preview",
        help="The name of the LLM model to use (e.g., gemini-3-pro-preview, gemini-3-flash-preview).",
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
        type=int,
        default=0,
        help="Set the verbosity level (0, 1, or 2).",
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
    parser.add_argument(
        "--cache-tokens",
        action="store_true",
        help="Enable token caching.",
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
        args.cache_tokens,
    )
    save_flashcards(flashcards, args.output_path)


if __name__ == "__main__":
    main()
