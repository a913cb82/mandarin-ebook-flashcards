import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import tomli

from flashcard import (
    EXPECTED_COLUMNS,
    TOML_PATH,
    RateLimiter,
    create_flashcards,
    is_rate_limit_error,
    parse_aichat_response,
    save_flashcards,
    validate_flashcard,
)


def _make_aichat_stdout(cards: list[dict[str, str]]) -> bytes:
    """Build fake aichat stdout with optional thinking text."""
    thinking = "<think>Okay, I need to generate flashcards.</think>\n\n"
    return (thinking + json.dumps(cards)).encode()


def _make_run_result(
    stdout: bytes, returncode: int = 0, stderr: bytes = b""
) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


def test_save_flashcards_with_tabs(tmp_path):
    output_path = tmp_path / "output.tsv"
    card = {
        "hanzi": "你好",
        "pinyin": "ní hǎo",
        "pinyinnumbered": "ni2 hao3",
        "definition": "hello\tworld",
        "partofspeech": "greeting",
        "sentencehanzi": "你好吗？",
        "sentencepinyin": "sp",
        "sentencetranslation": "st",
    }
    df = pd.DataFrame([card])
    save_flashcards(df, str(output_path))

    content = output_path.read_text()
    lines = content.strip().split("\n")
    for line in lines:
        parts = line.split("\t")
        assert len(parts) == len(EXPECTED_COLUMNS)
        assert "hello world" in line


def test_validate_flashcard():
    card = pd.Series(
        {
            "hanzi": "你好",
            "pinyin": "ní hǎo",
            "pinyinnumbered": "ni2 hao3",
            "definition": "hello",
            "partofspeech": "greeting",
            "sentencehanzi": "你好吗？",
            "sentencepinyin": "Nǐ hǎo ma?",
            "sentencetranslation": "How are you?",
        }
    )
    assert validate_flashcard(card, "你好") is True
    assert validate_flashcard(card.drop("pinyin"), "你好") is False


def test_validate_flashcard_nu_er():
    card = pd.Series(
        {
            "hanzi": "女儿",
            "pinyin": "nǚ ér",
            "pinyinnumbered": "nv3 er2",
            "definition": "daughter",
            "partofspeech": "noun",
            "sentencehanzi": "这是我的女儿。",
            "sentencepinyin": "Zhè shì wǒ de nǚ'ér.",
            "sentencetranslation": "This is my daughter.",
        }
    )
    assert validate_flashcard(card, "女儿") is True


def test_validate_flashcard_failures():
    base_card = {
        "hanzi": "你好",
        "pinyin": "ní hǎo",
        "pinyinnumbered": "ni2 hao3",
        "definition": "hello",
        "partofspeech": "greeting",
        "sentencehanzi": "你好吗？",
        "sentencepinyin": "Nǐ hǎo ma?",
        "sentencetranslation": "How are you?",
    }

    card = base_card.copy()
    card["hanzi"] = "再见"
    assert validate_flashcard(card, "你好") is False

    card = base_card.copy()
    card["pinyin"] = "nǐ hǎo"
    assert validate_flashcard(card, "你好") is False

    card = base_card.copy()
    card["definition"] = ""
    assert validate_flashcard(card, "你好") is False

    card = base_card.copy()
    card["sentencehanzi"] = "我很好。"
    assert validate_flashcard(card, "你好") is False

    card = base_card.copy()
    card["pinyin"] = "ní hǎo | nǐ hǎo"
    card["definition"] = "hello"
    assert validate_flashcard(card, "你好") is False

    card = base_card.copy()
    card["pinyin"] = "ní; hǎo"
    card["pinyinnumbered"] = "ni2 hao3"
    assert validate_flashcard(card, "你好") is False


def test_parse_aichat_response_valid():
    cards = [
        {
            "hanzi": "你好",
            "pinyin": "ní hǎo",
            "pinyinnumbered": "ni2 hao3",
            "definition": "hello",
            "partofspeech": "greeting",
            "sentencehanzi": "你好吗？",
            "sentencepinyin": "Nǐ hǎo ma?",
            "sentencetranslation": "How are you?",
        }
    ]
    stdout = json.dumps(cards).encode()
    result = parse_aichat_response(stdout)
    assert result == cards


def test_parse_aichat_response_with_thinking():
    cards = [{"hanzi": "你好"}]
    stdout = _make_aichat_stdout(cards)
    result = parse_aichat_response(stdout)
    assert result == cards


def test_parse_aichat_response_no_json():
    stdout = b"just some text with no json"
    result = parse_aichat_response(stdout)
    assert result == []


def test_parse_aichat_response_invalid_json():
    stdout = b"text before [invalid json] text after"
    result = parse_aichat_response(stdout)
    assert result == []


@patch("flashcard.subprocess.run")
def test_create_flashcards_basic(mock_run, tmp_path):
    word = "你好"
    cards = [
        {
            "hanzi": word,
            "pinyin": "ní hǎo",
            "pinyinnumbered": "ni2 hao3",
            "definition": "hello",
            "partofspeech": "greeting",
            "sentencehanzi": "你好吗？",
            "sentencepinyin": "Nǐ hǎo ma?",
            "sentencetranslation": "How are you?",
        }
    ]
    mock_run.return_value = _make_run_result(_make_aichat_stdout(cards))

    flashcards = create_flashcards(
        [word], cache_dir=str(tmp_path), batch_size=1
    )
    assert len(flashcards) == 1
    assert flashcards.iloc[0]["hanzi"] == word


@patch("flashcard.subprocess.run")
def test_create_flashcards_multiple_words(mock_run, tmp_path):
    words = ["你好", "世界"]
    cards = [
        {
            "hanzi": "你好",
            "pinyin": "ní hǎo",
            "pinyinnumbered": "ni2 hao3",
            "definition": "hello",
            "partofspeech": "greeting",
            "sentencehanzi": "你好世界",
            "sentencepinyin": "Nǐ hǎo shì jiè.",
            "sentencetranslation": "Hello world.",
        },
        {
            "hanzi": "世界",
            "pinyin": "shì jiè",
            "pinyinnumbered": "shi4 jie4",
            "definition": "world",
            "partofspeech": "noun",
            "sentencehanzi": "你好世界",
            "sentencepinyin": "Nǐ hǎo shì jiè.",
            "sentencetranslation": "Hello world.",
        },
    ]
    mock_run.return_value = _make_run_result(_make_aichat_stdout(cards))

    flashcards = create_flashcards(
        words, cache_dir=str(tmp_path), batch_size=2
    )
    assert set(flashcards["hanzi"].tolist()) == set(words)


@patch("flashcard.subprocess.run")
def test_create_flashcards_retries_on_invalid(mock_run, tmp_path):
    word = "你好"
    valid_card = {
        "hanzi": word,
        "pinyin": "ní hǎo",
        "pinyinnumbered": "ni2 hao3",
        "definition": "hello",
        "partofspeech": "greeting",
        "sentencehanzi": "你好吗？",
        "sentencepinyin": "Nǐ hǎo ma?",
        "sentencetranslation": "How are you?",
    }
    invalid_card = {"hanzi": "wrong"}
    mock_run.side_effect = [
        _make_run_result(_make_aichat_stdout([invalid_card])),
        _make_run_result(_make_aichat_stdout([valid_card])),
    ]

    flashcards = create_flashcards(
        [word], cache_dir=str(tmp_path), batch_size=1, retries=2, rpm=0
    )
    assert len(flashcards) == 1
    assert mock_run.call_count == 2


@patch("flashcard.subprocess.run")
def test_create_flashcards_hits_max_retries(mock_run, tmp_path):
    word = "你好"
    invalid_card = {"hanzi": "wrong"}
    mock_run.return_value = _make_run_result(
        _make_aichat_stdout([invalid_card])
    )

    with patch("builtins.print") as mock_print:
        flashcards = create_flashcards(
            [word],
            cache_dir=str(tmp_path),
            batch_size=1,
            retries=2,
            verbose=True,
            rpm=0,
        )
        assert flashcards.empty
        failure_msg = f"Failed to create valid flashcard for word: {word}"
        failure_calls = [
            c for c in mock_print.mock_calls if failure_msg in str(c)
        ]
        assert len(failure_calls) > 0


@patch("flashcard.subprocess.run")
def test_create_flashcards_handles_nonzero_exit(mock_run, tmp_path):
    word = "你好"
    valid_card = {
        "hanzi": word,
        "pinyin": "ní hǎo",
        "pinyinnumbered": "ni2 hao3",
        "definition": "h",
        "partofspeech": "n",
        "sentencehanzi": "你好吗？",
        "sentencepinyin": "sp",
        "sentencetranslation": "st",
    }
    mock_run.side_effect = [
        _make_run_result(
            stdout=b"",
            returncode=1,
            stderr=b"Error: rate limited",
        ),
        _make_run_result(_make_aichat_stdout([valid_card])),
    ]

    flashcards = create_flashcards(
        [word],
        cache_dir=str(tmp_path),
        batch_size=1,
        retries=2,
        rpm=0,
    )
    assert len(flashcards) == 1
    assert mock_run.call_count == 2


@patch("flashcard.subprocess.run")
def test_create_flashcards_with_caching(mock_run, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    word = "你好"
    card = {
        "hanzi": word,
        "pinyin": "ní hǎo",
        "pinyinnumbered": "ni2 hao3",
        "definition": "h",
        "partofspeech": "n",
        "sentencehanzi": "你好吗？",
        "sentencepinyin": "sp",
        "sentencetranslation": "st",
    }
    mock_run.return_value = _make_run_result(_make_aichat_stdout([card]))

    create_flashcards([word], batch_size=1, cache_dir=str(cache_dir))
    assert mock_run.call_count == 1

    create_flashcards([word], batch_size=1, cache_dir=str(cache_dir))
    assert mock_run.call_count == 1


@patch("flashcard.subprocess.run")
def test_create_flashcards_preserves_output_order(mock_run, tmp_path):
    words = ["你好", "世界"]
    cards = [
        {
            "hanzi": "世界",
            "pinyin": "shì jiè",
            "pinyinnumbered": "shi4 jie4",
            "definition": "world",
            "partofspeech": "noun",
            "sentencehanzi": "你好世界",
            "sentencepinyin": "Nǐ hǎo shì jiè.",
            "sentencetranslation": "Hello world.",
        },
        {
            "hanzi": "你好",
            "pinyin": "ní hǎo",
            "pinyinnumbered": "ni2 hao3",
            "definition": "hello",
            "partofspeech": "greeting",
            "sentencehanzi": "你好世界",
            "sentencepinyin": "Nǐ hǎo shì jiè.",
            "sentencetranslation": "Hello world.",
        },
    ]
    mock_run.return_value = _make_run_result(_make_aichat_stdout(cards))

    flashcards = create_flashcards(
        words, cache_dir=str(tmp_path), batch_size=2
    )
    assert flashcards["hanzi"].tolist() == words


# --- Rate limit error detection ---


def test_is_rate_limit_error_429():
    assert is_rate_limit_error("Error 429: Too many requests") is True


def test_is_rate_limit_error_resource_exhausted():
    assert is_rate_limit_error("429 RESOURCE_EXHAUSTED") is True


def test_is_rate_limit_error_gemini_message():
    stderr = "Resource has been exhausted (e.g. check quota). (status: 429)"
    assert is_rate_limit_error(stderr) is True


def test_is_rate_limit_error_not_rate_limit():
    assert is_rate_limit_error("Connection timeout") is False
    assert is_rate_limit_error("") is False
    assert is_rate_limit_error("Some other error") is False


def test_is_rate_limit_error_rate_limit_text():
    assert is_rate_limit_error("Rate limit exceeded") is True
    assert is_rate_limit_error("rate_limit_reached") is True


# --- RateLimiter ---


def test_rate_limiter_allows_immediate_first_call():
    limiter = RateLimiter(rpm=60)
    import time

    start = time.monotonic()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed < 0.1


def test_rate_limiter_blocks_when_too_fast():
    limiter = RateLimiter(rpm=60)  # 1 per second
    limiter.wait()  # First call
    with patch("flashcard.time.sleep") as mock_sleep:
        limiter.wait()  # Should call sleep
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] >= 0.9


def test_rate_limiter_disabled_when_rpm_zero():
    limiter = RateLimiter(rpm=0)
    import time

    start = time.monotonic()
    for _ in range(10):
        limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed < 0.1


# --- Multithreading ---


@patch("flashcard.subprocess.run")
def test_create_flashcards_multithreaded(mock_run, tmp_path):
    words = ["你好", "世界", "中国", "美国"]
    cards = [
        {
            "hanzi": w,
            "pinyin": f"{w[0]}i",
            "pinyinnumbered": f"{w[0]}i",
            "definition": f"word {w}",
            "partofspeech": "n",
            "sentencehanzi": f"{w}好",
            "sentencepinyin": "sp",
            "sentencetranslation": "st",
        }
        for w in words
    ]
    mock_run.return_value = _make_run_result(_make_aichat_stdout(cards))

    flashcards = create_flashcards(
        words,
        cache_dir=str(tmp_path),
        batch_size=2,
        workers=2,
        rpm=0,
    )
    assert len(flashcards) == 4
    assert set(flashcards["hanzi"].tolist()) == set(words)


@patch("flashcard.subprocess.run")
def test_create_flashcards_rate_limit_detected(mock_run, tmp_path):
    word = "你好"
    valid_card = {
        "hanzi": word,
        "pinyin": "ní hǎo",
        "pinyinnumbered": "ni2 hao3",
        "definition": "hello",
        "partofspeech": "greeting",
        "sentencehanzi": "你好吗？",
        "sentencepinyin": "Nǐ hǎo ma?",
        "sentencetranslation": "How are you?",
    }
    mock_run.side_effect = [
        _make_run_result(
            stdout=b"",
            returncode=1,
            stderr=b"429 RESOURCE_EXHAUSTED",
        ),
        _make_run_result(_make_aichat_stdout([valid_card])),
    ]

    with patch("builtins.print") as mock_print:
        flashcards = create_flashcards(
            [word],
            cache_dir=str(tmp_path),
            batch_size=1,
            retries=2,
            rpm=0,
            verbose=True,
        )
        assert len(flashcards) == 1
        rate_limit_calls = [
            c for c in mock_print.mock_calls if "RATE LIMITED" in str(c)
        ]
        assert len(rate_limit_calls) > 0


# --- Manual integration test (not run by default) ---


@pytest.mark.manual
@pytest.mark.timeout(120)
def test_manual_aichat_integration():
    """Run a real aichat subprocess.

    Run with:
        pytest -m manual -s tests/test_flashcard.py
        MANUAL_MODEL=gemini:gemini-3.6-flash \
            pytest -m manual -s tests/test_flashcard.py
    """
    model = os.environ.get("MANUAL_MODEL", "ollama:qwen3:8b")
    words = ["你好", "世界"]
    with tempfile.TemporaryDirectory() as cache_dir:
        cards = create_flashcards(
            words,
            cache_dir=cache_dir,
            batch_size=2,
            model=model,
        )
        print(cards.to_string())
        assert len(cards) > 0
        assert set(cards["hanzi"].tolist()) == set(words)


def test_system_prompt_examples_validate():
    """All example JSONs in system_prompt.toml must pass validation."""
    with open(TOML_PATH, "rb") as f:
        data = tomli.load(f)

    for i, ex in enumerate(data["examples"]):
        cards = json.loads(ex["output"])
        for card in cards:
            assert validate_flashcard(card, card["hanzi"]), (
                f"Example {i + 1}: {card['hanzi']} failed validation"
            )
