import json
import os
import random
import time
from typing import Any

import pandas as pd
import toml
from google import genai
from google.genai import types
from tqdm import tqdm

from pinyin import convert_pinyin

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


def validate_flashcard(card: Any, word: str, verbose: int = 0) -> bool:
    """Validates a flashcard's structure and content consistency."""
    card_dict = card.to_dict() if isinstance(card, pd.Series) else card

    if not all(col in card_dict for col in EXPECTED_COLUMNS):
        if verbose > 1:
            print(f"Missing columns for {word}")
        return False
    if any(
        pd.isna(card_dict.get(col)) or str(card_dict.get(col, "")).strip() == ""
        for col in EXPECTED_COLUMNS
    ):
        if verbose > 1:
            print(f"Empty or NaN values for {word}")
        return False
    if card_dict["hanzi"] != word:
        if verbose > 1:
            print(f"Hanzi mismatch for {word}: {card_dict['hanzi']}")
        return False
    if word not in card_dict["sentencehanzi"]:
        if verbose > 1:
            print(f"Word {word} not in sentence: {card_dict['sentencehanzi']}")
        return False

    # Pinyin consistency check
    tm_parts = [
        p.strip() for p in str(card_dict["pinyin"]).replace(";", "|").replace("'", "").split("|")
    ]
    n_parts = [p.strip() for p in str(card_dict["pinyinnumbered"]).replace(";", "|").split("|")]

    if len(tm_parts) != len(n_parts):
        if verbose > 1:
            print(f"Pinyin part count mismatch for {word}")
        return False
    for tm, n in zip(tm_parts, n_parts, strict=True):
        if tm != convert_pinyin(n):
            if verbose > 1:
                print(f"Pinyin conversion mismatch for {word}: {tm} != {convert_pinyin(n)}")
            return False

    # Structure check (semicolons vs pipes)
    def get_struct(s: Any) -> list[int]:
        return [len(p.split(";")) for p in str(s).split("|")]

    if get_struct(card_dict["pinyin"]) != get_struct(card_dict["pinyinnumbered"]):
        if verbose > 1:
            print(f"Structure mismatch for {word}")
        return False

    def get_pipe_count(s: Any) -> int:
        return len(str(s).split("|"))

    if not (
        get_pipe_count(card_dict["pinyin"])
        == get_pipe_count(card_dict["definition"])
        == get_pipe_count(card_dict["partofspeech"])
    ):
        if verbose > 1:
            print(f"Pipe count mismatch for {word}")
        return False
    return True


def create_flashcards(
    words: list[str],
    cache_dir: str = ".flashcard_cache",
    initial_batch_size: int = 100,
    batch_size_multiplier: float = 2.0,
    retries: int = 3,
    model: str = "gemini-1.5-flash",
    verbose: bool = False,
) -> pd.DataFrame:
    """Creates flashcards for a list of words using the Gemini API, with caching and batching."""
    os.makedirs(cache_dir, exist_ok=True)
    with open("system_prompt.toml") as f:
        prompt_data = toml.load(f)

    system_prompt = prompt_data["system_prompt"]
    examples = prompt_data["examples"]

    flashcards_map = {}
    to_process = []

    pbar = tqdm(total=len(words), desc="Creating flashcards")

    for word in words:
        cache_path = os.path.join(cache_dir, f"{word}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    card = json.load(f)
                if validate_flashcard(card, word, verbose=2 if verbose else 0):
                    flashcards_map[word] = card
                    pbar.update(1)
                    continue
                elif verbose:
                    print(f"Cached card for '{word}' failed validation.")
            except Exception:
                pass
        to_process.append(word)

    if not to_process:
        pbar.close()
        return pd.DataFrame([flashcards_map[w] for w in words if w in flashcards_map])

    random.shuffle(to_process)
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    batch_size = initial_batch_size
    max_batch_size = 1000000
    retry_counts = dict.fromkeys(to_process, 0)

    while to_process:
        pbar.set_postfix(batch_size=batch_size)
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
                        properties={
                            k: types.Schema(type=types.Type.STRING) for k in EXPECTED_COLUMNS
                        },
                        required=EXPECTED_COLUMNS,
                    ),
                ),
            )

            contents = []
            for ex in examples:
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=ex["input"])])
                )
                contents.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=ex["output"])])
                )
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text="..\n".join(batch))])
            )

            response = client.models.generate_content(model=model, contents=contents, config=config)
            if response.text is None:
                raise ValueError("API returned empty response text")

            results = json.loads(response.text)
            res_map = {r["hanzi"]: r for r in results}

            succeeded = 0
            for word in batch:
                if word in res_map and validate_flashcard(res_map[word], word):
                    flashcards_map[word] = res_map[word]
                    with open(os.path.join(cache_dir, f"{word}.json"), "w") as f:
                        json.dump(res_map[word], f)
                    pbar.update(1)
                    succeeded += 1
                else:
                    retry_counts[word] += 1
                    if retry_counts[word] < retries:
                        to_process.append(word)
                    else:
                        if verbose:
                            print(f"Failed to create valid flashcard for word: {word}")
                        pbar.update(1)

            if succeeded > len(batch) / 2:
                new_batch_size = int(
                    min((batch_size + max_batch_size) // 2, batch_size * batch_size_multiplier)
                )
                if verbose:
                    print(f"increasing batch size to {new_batch_size}")
                batch_size = new_batch_size
            else:
                if verbose:
                    print(
                        f"decreasing batch size to "
                        f"{int(max(1, batch_size // batch_size_multiplier))}"
                    )
                batch_size = int(max(1, batch_size // batch_size_multiplier))

        except Exception as e:
            if verbose:
                print(f"Exception occurred: {e}")
            if "429" in str(e):
                time.sleep(30)
                to_process = batch + to_process
            else:
                if verbose:
                    print(f"Error: {e}")
                for word in batch:
                    retry_counts[word] += 1
                    if retry_counts[word] < retries:
                        to_process.append(word)
                    else:
                        pbar.update(1)
                if verbose:
                    print(
                        f"decreasing batch size to "
                        f"{int(max(1, batch_size // batch_size_multiplier))}"
                    )
                max_batch_size = batch_size
                batch_size = int(max(1, batch_size // batch_size_multiplier))

    pbar.close()
    final_list = [flashcards_map[w] for w in words if w in flashcards_map]
    return pd.DataFrame(final_list) if final_list else pd.DataFrame()


def save_flashcards(flashcards: pd.DataFrame, file_path: str) -> None:
    """Saves flashcards to a TSV file."""
    if not flashcards.empty:
        flashcards[EXPECTED_COLUMNS].to_csv(file_path, sep="\t", index=False, header=False)
