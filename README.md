# Mandarin Ebook Flashcards

Generate Anki-ready flashcards from Chinese ebooks using aichat.

## Usage

```bash
pip install -e ".[dev]"                         # install

python src/main.py input.epub cards.tsv         # ebook -> flashcards
python src/main.py vocab.txt cards.tsv --flashcards-only  # vocab list -> flashcards
python src/main.py input.epub vocab.txt --vocab-only      # extract vocab only
```

Run `python src/main.py --help` for all options and output format.

## Development

```bash
pip install -e ".[dev]"       # install with dev deps
pytest                        # run tests (no network)
ruff check src                # lint
uvx ty check                  # typecheck
```

### How it works

1. Extracts vocabulary from EPUB/text using jieba word segmentation
2. Sends words to aichat in adaptive batches for flashcard generation
3. Validates and caches results, outputs tab-separated TSV

### Key features

- Adaptive batch sizing based on success rate
- Multithreaded with configurable rate limiting (`--rpm`, `--workers`)
- Per-word caching in `{cache_dir}/{word}.json`
- System prompt defined in `system_prompt.toml` (single source of truth)

### Requirements

- Python 3.10+
- aichat installed and configured (`aichat --list-models`)

## Data, books, and cache

- `data/SUBTLEX-CH-WF`: SUBTLEX-CH word-frequency list (Cai & Brysbaert;
  GBK-encoded, tab-separated, 3 header rows). Not in git (see `.gitignore`);
  obtain it from the SUBTLEX-CH release and place it at this path
  (override with `--global-freqs`). Used to rank vocabulary by global frequency.
- `books/`: drop your `.epub` files here. Not in git; e.g.
  `python src/main.py books/<name>.epub cards.tsv`.
- `.flashcard_cache/`: per-word JSON cache (`{word}.json`, default `--cache-dir`).
  Not in git. Safe to delete to force regeneration (stale entries are simply
  re-fetched); `show-cache.sh` inspects it.

Regenerate the pinned snapshot with `.venv/bin/pip freeze > requirements.snapshot.txt`.
Copy `.env.example` to `.env` and fill in values.
