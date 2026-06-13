"""Tests for comprehension-based vocabulary extraction."""

import os
import tempfile
from unittest.mock import patch

from utils import extract_vocabulary


@patch("jieba.cut")
def test_cutoff_at_boundary(mock_cut) -> None:
    """Cutoff lands at a clean frequency boundary."""
    mock_cut.return_value = ["走"] * 5 + ["跑"] * 5 + ["喝"] * 3 + ["吃"] * 2
    # freq groups: 走=5, 跑=5, 喝=3, 吃=2, total=15
    # 50% → after 走+跑 (10/15 >= 0.5), cutoff=2
    words = extract_vocabulary("dummy", comprehension_pct=0.5)
    assert set(words) == {"走", "跑"}


@patch("jieba.cut")
def test_cutoff_mid_bucket_no_global(mock_cut) -> None:
    """Cutoff splits a frequency bucket — insertion order used."""
    mock_cut.return_value = ["走"] * 2 + ["跑"] * 2 + ["喝"] * 2 + ["吃"] * 2
    # All freq=2, total=8. 50% → after 2 words (4/8 >= 0.5), cutoff=2
    # Insertion order: 走, 跑 first → those are included
    words = extract_vocabulary("dummy", comprehension_pct=0.5)
    assert words == ["走", "跑"]


@patch("jieba.cut")
def test_cutoff_mid_bucket_with_global(mock_cut) -> None:
    """Global freqs override insertion order within the cutoff bucket."""
    mock_cut.return_value = ["走"] * 2 + ["跑"] * 2 + ["喝"] * 2 + ["吃"] * 2
    # All freq=2, total=8. Need 2 of them.
    # Global: 吃=100, 喝=50, 跑=10. 走 not in global (defaults 0).
    # Sorted: 吃, 喝, 跑, 走 → first 2: 吃, 喝
    words = extract_vocabulary(
        "dummy",
        comprehension_pct=0.5,
        global_freqs={"吃": 100, "喝": 50, "跑": 10},
    )
    assert words == ["吃", "喝"]


@patch("jieba.cut")
def test_zero_comprehension(mock_cut) -> None:
    mock_cut.return_value = ["走", "跑"]
    words = extract_vocabulary("dummy", comprehension_pct=0.0)
    assert words == []


@patch("jieba.cut")
def test_full_comprehension(mock_cut) -> None:
    """100% returns all words."""
    mock_cut.return_value = ["走"] * 3 + ["跑"] * 1
    words = extract_vocabulary("dummy", comprehension_pct=1.0)
    assert set(words) == {"走", "跑"}


@patch("jieba.cut")
def test_empty_text(mock_cut) -> None:
    mock_cut.return_value = []
    words = extract_vocabulary("")
    assert words == []


@patch("jieba.cut")
def test_single_word_repeated(mock_cut) -> None:
    mock_cut.return_value = ["走"] * 20
    words = extract_vocabulary("dummy", comprehension_pct=0.98)
    assert words == ["走"]


@patch("jieba.cut")
def test_stop_words_excluded(mock_cut) -> None:
    mock_cut.return_value = ["走"] * 2 + ["跑"] * 5 + ["吃"] * 1
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("走\n")
        sw = f.name
    # After filtering: 跑=5, 吃=1, total=6. 80% → 跑 alone: 5/6=83% >= 80%
    words = extract_vocabulary(
        "dummy", stop_words_path=sw, comprehension_pct=0.8
    )
    os.unlink(sw)
    assert words == ["跑"]
