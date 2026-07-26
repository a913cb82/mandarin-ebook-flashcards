# Mandarin Ebook Flashcards

Generate Anki-ready flashcards from Chinese ebooks using aichat.

## Usage

```bash
pip install -e ".[dev]"                         # install

python src/main.py input.epub cards.tsv         # ebook -> flashcards
python src/main.py vocab.txt cards.tsv --flashcards-only  # vocab list -> flashcards
python src/main.py input.epub vocab.txt --vocab-only      # extract vocab only
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `ollama:qwen3:8b` | aichat model (run `aichat --list-models`) |
| `--rpm` | `10` | Requests per minute limit (0 = unlimited) |
| `--workers` | `1` | Concurrent threads |
| `--batch-size` | `20` | Words per batch (adaptive) |
| `--retries` | `3` | Max retries per word |
| `--comprehension` | `0.98` | Vocabulary coverage threshold (0-1) |
| `--cache-dir` | `.flashcard_cache` | Per-word cache location |
| `--verbose` | | Print retry/error details |

### Output

Tab-separated, 8 columns, no header: `hanzi`, `pinyin`, `pinyinnumbered`, `definition`, `partofspeech`, `sentencehanzi`, `sentencepinyin`, `sentencetranslation`

## Development

```bash
pip install -e ".[dev]"       # install with dev deps
pytest                        # run tests (no network)
ruff check src                # lint
mypy src                      # typecheck
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
