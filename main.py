import argparse
import json
import os
import random
import re
import time
from collections import Counter
from typing import List, Optional, Any

import ebooklib
import jieba
import pandas as pd
import toml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ebooklib import epub
from google import genai
from google.genai import types
from tqdm import tqdm

# Constants
EXPECTED_COLUMNS = [
    "hanzi", "pinyin", "pinyinnumbered", "definition", "partofspeech",
    "sentencehanzi", "sentencepinyin", "sentencetranslation"
]

PINYIN_TONE_MARKS = {
    'a': 'āáǎà', 'e': 'ēéěè', 'i': 'īíǐì',
    'o': 'ōóǒò', 'u': 'ūúǔù', 'ü': 'ǖǘǚǜ',
    'A': 'ĀÁǍÀ', 'E': 'ĒÉĚÈ', 'I': 'ĪÍǏÌ',
    'O': 'ŌÓǑÒ', 'U': 'ŪÚǓÙ', 'Ü': 'ǕǗǙǛ'
}

def convert_pinyin_callback(m):
    tone = int(m.group(3)) % 5
    r = m.group(1).replace('v', 'ü').replace('V', 'Ü')
    pos = 1 if len(r) > 1 and r[0] not in 'aeoAEO' else 0
    if tone != 0:
        r = r[0:pos] + PINYIN_TONE_MARKS[r[pos]][tone - 1] + r[pos+1:]
    return r + m.group(2)

def convert_pinyin(s: str) -> str:
    return re.sub(r'([aeiouüvÜ]{1,3})(n?g?r?)([012345])', convert_pinyin_callback, s, flags=re.IGNORECASE)

def validate_flashcard(card: Any, word: str, verbose: int = 0) -> bool:
    card_dict = card.to_dict() if isinstance(card, pd.Series) else card

    if not all(col in card_dict for col in EXPECTED_COLUMNS):
        if verbose > 1: print(f"Missing columns for {word}")
        return False
    if any(pd.isna(card_dict.get(col)) or str(card_dict.get(col, "")).strip() == "" for col in EXPECTED_COLUMNS):
        if verbose > 1: print(f"Empty or NaN values for {word}")
        return False
    if card_dict["hanzi"] != word:
        if verbose > 1: print(f"Hanzi mismatch for {word}: {card_dict['hanzi']}")
        return False
    if word not in card_dict["sentencehanzi"]:
        if verbose > 1: print(f"Word {word} not in sentence: {card_dict['sentencehanzi']}")
        return False

    # Pinyin consistency check
    tm_parts = [p.strip() for p in str(card_dict["pinyin"]).replace(";", "|").replace("'", "").split("|")]
    n_parts = [p.strip() for p in str(card_dict["pinyinnumbered"]).replace(";", "|").split("|")]

    if len(tm_parts) != len(n_parts):
        if verbose > 1: print(f"Pinyin part count mismatch for {word}")
        return False
    for tm, n in zip(tm_parts, n_parts):
        if tm != convert_pinyin(n):
            if verbose > 1: print(f"Pinyin conversion mismatch for {word}: {tm} != {convert_pinyin(n)}")
            return False

    # Structure check (semicolons vs pipes)
    def get_struct(s): return [len(p.split(';')) for p in str(s).split('|')]
    if get_struct(card_dict["pinyin"]) != get_struct(card_dict["pinyinnumbered"]):
        if verbose > 1: print(f"Structure mismatch for {word}")
        return False
    
    pipe_count = lambda s: len(str(s).split('|'))
    if not (pipe_count(card_dict["pinyin"]) == pipe_count(card_dict["definition"]) == pipe_count(card_dict["partofspeech"])):
        if verbose > 1: print(f"Pipe count mismatch for {word}")
        return False
    return True

def read_epub(file_path: str) -> str:
    book = epub.read_epub(file_path)
    content = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        content.append(soup.get_text())
    return "\n".join(content)

def extract_vocabulary(text: str, stop_words_path: Optional[str] = None, min_freq: int = 1, verbose: bool = False) -> List[str]:
    stop_words = set()
    if stop_words_path and os.path.exists(stop_words_path):
        with open(stop_words_path, "r") as f:
            stop_words = set(f.read().splitlines())
    
    words = [w for w in jieba.cut(text) if all('\u4e00' <= c <= '\u9fff' for c in w) and w not in stop_words]
    counts = Counter(words)
    
    if verbose:
        for i in range(1, min_freq + 1):
            v_size = len([w for w, c in counts.items() if c >= i])
            print(f"Vocabulary size with min_freq={i}: {v_size}")
            
    return [w for w, c in counts.items() if c >= min_freq]

def create_flashcards(words: List[str], cache_dir: str = ".flashcard_cache", initial_batch_size: int = 100, 
                      batch_size_multiplier: float = 2.0, retries: int = 3, model: str = "gemini-1.5-flash", 
                      verbose: bool = False, cache_tokens: bool = False) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    with open("system_prompt.toml", "r") as f:
        prompt_data = toml.load(f)
    
    system_prompt = prompt_data["system_prompt"]
    examples = prompt_data["examples"]
    
    flashcards_map = {}
    to_process = []
    
    pbar = tqdm(total=len(words), desc="Creating flashcards") if verbose else None
    
    for word in words:
        cache_path = os.path.join(cache_dir, f"{word}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    card = json.load(f)
                if validate_flashcard(card, word, verbose=2 if verbose else 0):
                    flashcards_map[word] = card
                    if pbar: pbar.update(1)
                    continue
                elif verbose:
                    print(f"Cached card for '{word}' failed validation.")
            except Exception:
                pass
        to_process.append(word)

    if not to_process:
        if pbar: pbar.close()
        return pd.DataFrame([flashcards_map[w] for w in words if w in flashcards_map])

    random.shuffle(to_process)
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    batch_size = initial_batch_size
    retry_counts = {w: 0 for w in to_process}
    
    while to_process:
        batch = to_process[:batch_size]
        to_process = to_process[batch_size:]
        
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=system_prompt,
                response_schema=types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={k: types.Schema(type=types.Type.STRING) for k in EXPECTED_COLUMNS},
                        required=EXPECTED_COLUMNS
                    )
                )
            )
            
            contents = []
            for ex in examples:
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=ex["input"])]))
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=ex["output"])]))
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="..\n".join(batch))]))

            response = client.models.generate_content(model=model, contents=contents, config=config)
            results = json.loads(response.text)
            res_map = {r["hanzi"]: r for r in results}

            succeeded = 0
            for word in batch:
                if word in res_map and validate_flashcard(res_map[word], word):
                    flashcards_map[word] = res_map[word]
                    with open(os.path.join(cache_dir, f"{word}.json"), "w") as f:
                        json.dump(res_map[word], f)
                    if pbar: pbar.update(1)
                    succeeded += 1
                else:
                    retry_counts[word] += 1
                    if retry_counts[word] < retries:
                        to_process.append(word)
                    else:
                        if verbose: print(f"Failed to create valid flashcard for word: {word}")
                        if pbar: pbar.update(1)
            
            if succeeded > len(batch) / 2:
                if verbose: print(f"increasing batch size to {int(batch_size * batch_size_multiplier)}")
                batch_size = int(batch_size * batch_size_multiplier)
            else:
                if verbose: print(f"decreasing batch size to {max(1, int(batch_size / batch_size_multiplier))}")
                batch_size = max(1, int(batch_size / batch_size_multiplier))
                
        except Exception as e:
            if "429" in str(e):
                time.sleep(30)
                to_process = batch + to_process
            else:
                if verbose: print(f"Error: {e}")
                for word in batch:
                    retry_counts[word] += 1
                    if retry_counts[word] < retries:
                        to_process.append(word)
                    elif pbar: pbar.update(1)
                batch_size = max(1, int(batch_size / batch_size_multiplier))

    if pbar: pbar.close()
    final_list = [flashcards_map[w] for w in words if w in flashcards_map]
    return pd.DataFrame(final_list) if final_list else pd.DataFrame()

def save_flashcards(flashcards: pd.DataFrame, file_path: str) -> None:
    if not flashcards.empty:
        flashcards[EXPECTED_COLUMNS].to_csv(file_path, sep="\t", index=False, header=False)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Create Anki flashcards from Chinese text.")
    parser.add_argument("input_path", help="Path to input (EPUB or txt list)")
    parser.add_argument("output_path", help="Path to save flashcards (TSV)")
    parser.add_argument("--initial-batch-size", type=int, default=100)
    parser.add_argument("--batch-size-multiplier", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--model", default="gemini-1.5-flash")
    parser.add_argument("--vocab-only", action="store_true")
    parser.add_argument("--flashcards-only", action="store_true")
    parser.add_argument("--stop-words", help="Path to stop words file")
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cache-dir", default=".flashcard_cache")
    args = parser.parse_args()

    if args.flashcards_only:
        with open(args.input_path, "r") as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        text = read_epub(args.input_path) if args.input_path.endswith(".epub") else open(args.input_path).read()
        words = extract_vocabulary(text, args.stop_words, args.min_frequency, args.verbose)

    if args.vocab_only:
        with open(args.output_path, "w") as f:
            f.write("\n".join(words))
        return

    flashcards = create_flashcards(words, cache_dir=args.cache_dir, initial_batch_size=args.initial_batch_size,
                                  batch_size_multiplier=args.batch_size_multiplier, retries=args.retries,
                                  model=args.model, verbose=args.verbose)
    save_flashcards(flashcards, args.output_path)

if __name__ == "__main__":
    main()
