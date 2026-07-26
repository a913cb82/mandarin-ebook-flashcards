import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import tomli
from tqdm import tqdm

from pinyin import convert_pinyin

ROLE_PATH = Path.home() / ".config" / "aichat" / "roles" / "flashcard.md"
TOML_PATH = Path(__file__).parent.parent / "system_prompt.toml"


def build_role(
    toml_path: Path = TOML_PATH,
    role_path: Path = ROLE_PATH,
) -> None:
    """Generate aichat role file from system_prompt.toml."""
    with open(toml_path, "rb") as f:
        data = tomli.load(f)

    parts = [
        "---",
        "temperature: 0",
        "---",
        data["system_prompt"].rstrip("\n"),
    ]

    for ex in data["examples"]:
        parts.append("")
        parts.append("### INPUT:")
        parts.append(ex["input"])
        parts.append("### OUTPUT:")
        parsed = json.loads(ex["output"])
        compact = json.dumps(
            parsed, indent=None, ensure_ascii=False, separators=(",", ":")
        )
        parts.append(f"```json\n{compact}\n```")

    role_path.parent.mkdir(parents=True, exist_ok=True)
    role_path.write_text("\n".join(parts) + "\n")


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
        pd.isna(card_dict.get(col))
        or str(card_dict.get(col, "")).strip() == ""
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

    tm_parts = [
        p.strip()
        for p in str(card_dict["pinyin"])
        .replace(";", "|")
        .replace("'", "")
        .split("|")
    ]
    n_parts = [
        p.strip()
        for p in str(card_dict["pinyinnumbered"]).replace(";", "|").split("|")
    ]

    if len(tm_parts) != len(n_parts):
        if verbose > 1:
            print(f"Pinyin part count mismatch for {word}")
        return False
    for tm, n in zip(tm_parts, n_parts, strict=True):
        if tm != convert_pinyin(n):
            if verbose > 1:
                msg = (
                    f"Pinyin conversion mismatch for {word}: "
                    f"{tm} != {convert_pinyin(n)}"
                )
                print(msg)
            return False

    def get_struct(s: Any) -> list[int]:
        return [len(p.split(";")) for p in str(s).split("|")]

    if get_struct(card_dict["pinyin"]) != get_struct(
        card_dict["pinyinnumbered"]
    ):
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


def parse_aichat_response(stdout: bytes) -> list[dict[str, Any]]:
    """Parse aichat stdout to extract flashcard JSON array.

    Handles thinking text that may precede the JSON.
    """
    text = stdout.decode()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        result = json.loads(text[start:end])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def run_aichat(
    words: list[str],
    model: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Run aichat subprocess for a batch of words.

    Returns (parsed_cards, stderr_text or None).
    """
    batch_text = "\n".join(words)
    result = subprocess.run(
        ["aichat", "-r", "flashcard", "-m", model, batch_text],
        capture_output=True,
    )
    stderr_text = result.stderr.decode() if result.returncode != 0 else None
    if result.returncode != 0:
        return [], stderr_text
    return parse_aichat_response(result.stdout), stderr_text


def create_flashcards(
    words: list[str],
    cache_dir: str = ".flashcard_cache",
    batch_size: int = 20,
    retries: int = 3,
    model: str = "ollama:qwen3:8b",
    verbose: bool = False,
) -> pd.DataFrame:
    """Creates flashcards via aichat subprocess."""
    os.makedirs(cache_dir, exist_ok=True)

    flashcards_map: dict[str, dict[str, Any]] = {}
    to_process: list[str] = []

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
        return pd.DataFrame(
            [flashcards_map[w] for w in words if w in flashcards_map]
        )

    random.shuffle(to_process)
    retry_counts = dict.fromkeys(to_process, 0)

    while to_process:
        batch = to_process[:batch_size]
        del to_process[:batch_size]

        cards, stderr_text = run_aichat(batch, model)

        if stderr_text and verbose:
            print(f"aichat stderr: {stderr_text}")

        res_map = {c["hanzi"]: c for c in cards}
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
                if verbose:
                    print(f"Retry {retry_counts[word]}/{retries} for: {word}")
                if retry_counts[word] < retries:
                    to_process.append(word)
                else:
                    if verbose:
                        print(
                            f"Failed to create valid flashcard"
                            f" for word: {word}"
                        )
                    pbar.update(1)

        # Adaptive batch sizing (TODO: tune for subprocess overhead)
        if succeeded > len(batch) / 2:
            new_batch_size = min(batch_size * 2, batch_size + 100)
            if verbose:
                print(f"increasing batch size to {new_batch_size}")
            batch_size = new_batch_size
        else:
            new_batch_size = max(1, batch_size // 2)
            if verbose:
                print(f"decreasing batch size to {new_batch_size}")
            batch_size = new_batch_size

    pbar.close()
    final_list = [flashcards_map[w] for w in words if w in flashcards_map]
    return pd.DataFrame(final_list) if final_list else pd.DataFrame()


def save_flashcards(flashcards: pd.DataFrame, file_path: str) -> None:
    """Saves flashcards to a TSV file."""
    if not flashcards.empty:
        flashcards_clean = flashcards[EXPECTED_COLUMNS].copy()
        for col in EXPECTED_COLUMNS:
            flashcards_clean[col] = flashcards_clean[col].apply(
                lambda x: str(x).replace("\t", " ")
            )
        flashcards_clean.to_csv(file_path, sep="\t", index=False, header=False)
