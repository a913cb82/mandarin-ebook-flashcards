# Mandarin Ebook Flashcards

Generate Anki-ready flashcards from Chinese ebooks using Google Gemini.

## Commands

```bash
pip install ".[dev]"             # install everything
python src/main.py input.epub cards.tsv      # ebook -> flashcards
python src/main.py vocab.txt cards.tsv --flashcards-only  # vocab list -> flashcards
python src/main.py input.epub vocab.txt --vocab-only      # extract vocab only
pytest                           # run tests (no network)
ruff check src                   # lint
mypy src                         # typecheck
```

## Project Structure

- `src/main.py` — CLI entry point (argparse)
- `src/flashcard.py` — Gemini API calls, validation, caching, TSV output
- `src/pinyin.py` — numbered pinyin to tone-marked conversion
- `src/utils.py` — EPUB reading (ebooklib) + Chinese vocab extraction (jieba)
- `system_prompt.toml` — Gemini system prompt + few-shot examples
- `tests/` — 18 tests, all mocked (no network), disable-socket enforced

## Architecture

- EPUB text is extracted and segmented with jieba to find vocabulary
- `extract_vocabulary` returns the top words covering X% of tokens (default 98%, configurable via `--comprehension`)
- Words are sent to Gemini in adaptive batches (JSON structured output via `response_schema`)
- Results are cached per-word in `{cache_dir}/{word}.json` and validated before saving
- Output is tab-separated, 8 columns (hanzi, pinyin, pinyinnumbered, definition, partofspeech, sentence_hanzi, sentence_pinyin, sentence_english), no header

## Conventions

- Python 3.10+, strict type annotations (`disallow_untyped_defs`), ruff with line-length 79
- Pre-commit runs ruff, mypy, pytest before commits
- `.env` file for `GOOGLE_API_KEY` (gitignored)
- No `__init__.py` in `src/` — pytest adds `src/` to `pythonpath`
