import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
import tomli
from tqdm import tqdm

from pinyin import convert_pinyin

ROLE_PATH = Path.home() / ".config" / "aichat" / "roles" / "flashcard.md"
TOML_PATH = Path(__file__).parent.parent / "system_prompt.toml"


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self) -> None:
        """Block until a request slot is available."""
        if self.rpm <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait_time = self.interval - (now - self.last_call)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call = time.monotonic()


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


def validate_flashcard(
    card: pd.Series | dict[str, str],
    word: str,
    verbose: int = 0,
) -> bool:
    """Validates a flashcard's structure and content consistency."""
    if isinstance(card, pd.Series):
        raw = card.to_dict()
        card_dict: dict[str, str] = {}
        for k, v in raw.items():
            if v is None:
                if verbose > 1:
                    print(f"Empty or NaN values for {word}")
                return False
            card_dict[str(k)] = str(v)
    else:
        card_dict = card

    if not all(col in card_dict for col in EXPECTED_COLUMNS):
        if verbose > 1:
            print(f"Missing columns for {word}")
        return False
    if any(
        card_dict.get(col) is None or card_dict.get(col, "").strip() == ""
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
        for p in card_dict["pinyin"]
        .replace(";", "|")
        .replace("'", "")
        .split("|")
    ]
    n_parts = [
        p.strip()
        for p in card_dict["pinyinnumbered"].replace(";", "|").split("|")
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

    def get_struct(s: str) -> list[int]:
        return [len(p.split(";")) for p in s.split("|")]

    if get_struct(card_dict["pinyin"]) != get_struct(
        card_dict["pinyinnumbered"]
    ):
        if verbose > 1:
            print(f"Structure mismatch for {word}")
        return False

    def get_pipe_count(s: str) -> int:
        return len(s.split("|"))

    if not (
        get_pipe_count(card_dict["pinyin"])
        == get_pipe_count(card_dict["definition"])
        == get_pipe_count(card_dict["partofspeech"])
    ):
        if verbose > 1:
            print(f"Pipe count mismatch for {word}")
        return False
    return True


def parse_aichat_response(stdout: bytes) -> list[dict[str, str]]:
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


def is_rate_limit_error(stderr: str) -> bool:
    """Check if stderr contains a rate limit error (429).

    Detects:
    - 429 (HTTP status)
    - RESOURCE_EXHAUSTED (Gemini gRPC status)
    - rate limit / rate_limit (generic)
    - too many requests (generic)
    """
    lower = stderr.lower()
    return (
        "429" in lower
        or "resource_exhausted" in lower
        or "resource exhausted" in lower
        or "rate limit" in lower
        or "rate_limit" in lower
        or "too many requests" in lower
    )


def run_aichat(
    words: list[str],
    model: str,
) -> tuple[list[dict[str, str]], str | None, bool]:
    """Run aichat subprocess for a batch of words.

    Returns (parsed_cards, stderr_text or None, is_rate_limit_error).
    """
    batch_text = "\n".join(words)
    result = subprocess.run(
        ["aichat", "-r", "flashcard", "-m", model, batch_text],
        capture_output=True,
    )
    stderr_text = result.stderr.decode() if result.returncode != 0 else None
    rate_limited = stderr_text is not None and is_rate_limit_error(stderr_text)
    if result.returncode != 0:
        return [], stderr_text, rate_limited
    return parse_aichat_response(result.stdout), stderr_text, rate_limited


def create_flashcards(
    words: list[str],
    cache_dir: str = ".flashcard_cache",
    batch_size: int = 20,
    retries: int = 3,
    model: str = "ollama:qwen3:8b",
    verbose: bool = False,
    rpm: int = 10,
    workers: int = 1,
) -> pd.DataFrame:
    """Creates flashcards via aichat subprocess."""
    os.makedirs(cache_dir, exist_ok=True)

    flashcards_map: dict[str, dict[str, str]] = {}
    flashcards_lock = threading.Lock()
    to_process: list[str] = []
    process_lock = threading.Lock()
    retry_counts: dict[str, int] = {}
    retry_lock = threading.Lock()

    pbar = tqdm(total=len(words), desc="Creating flashcards")

    for word in words:
        cache_path = os.path.join(cache_dir, f"{word}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    card: dict[str, str] = json.load(f)
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

    rate_limiter = RateLimiter(rpm)
    batch_size_state = {"size": batch_size}

    def process_batch(batch: list[str]) -> None:
        """Process a single batch of words."""
        rate_limiter.wait()
        cards, stderr_text, rate_limited = run_aichat(batch, model)

        if stderr_text and verbose:
            prefix = "RATE LIMITED" if rate_limited else "aichat stderr"
            print(f"{prefix}: {stderr_text}")

        res_map = {c["hanzi"]: c for c in cards}
        succeeded = 0
        for word in batch:
            if word in res_map and validate_flashcard(res_map[word], word):
                with flashcards_lock:
                    flashcards_map[word] = res_map[word]
                with open(os.path.join(cache_dir, f"{word}.json"), "w") as f:
                    json.dump(res_map[word], f)
                pbar.update(1)
                succeeded += 1
            else:
                with retry_lock:
                    retry_counts[word] += 1
                    current_retry = retry_counts[word]
                if verbose:
                    print(f"Retry {current_retry}/{retries} for: {word}")
                if current_retry < retries:
                    with process_lock:
                        to_process.append(word)
                else:
                    if verbose:
                        print(
                            f"Failed to create valid flashcard"
                            f" for word: {word}"
                        )
                    pbar.update(1)

        # Adaptive batch sizing (TODO: tune for subprocess overhead)
        with process_lock:
            if succeeded > len(batch) / 2:
                new_batch_size = min(
                    batch_size_state["size"] * 2,
                    batch_size_state["size"] + 100,
                )
                if verbose:
                    print(f"increasing batch size to {new_batch_size}")
                batch_size_state["size"] = new_batch_size
            else:
                new_batch_size = max(1, batch_size_state["size"] // 2)
                if verbose:
                    print(f"decreasing batch size to {new_batch_size}")
                batch_size_state["size"] = new_batch_size

    if workers <= 1:
        # Single-threaded mode (original behavior)
        while to_process:
            with process_lock:
                batch = to_process[: batch_size_state["size"]]
                del to_process[: batch_size_state["size"]]
            process_batch(batch)
    else:
        # Multi-threaded mode
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as executor:
            while to_process:
                with process_lock:
                    batch = to_process[: batch_size_state["size"]]
                    del to_process[: batch_size_state["size"]]
                executor.submit(process_batch, batch)

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
