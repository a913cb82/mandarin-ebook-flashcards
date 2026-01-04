# Ebook Flashcards

This project extracts unique words from Chinese ebooks (or lists) and generates comprehensive flashcards using the Google Gemini API, saving them as Anki-ready TSV files.

## Features

*   **Extraction:** Reads `.epub` files or raw text lists, segmenting Chinese words via `jieba`.
*   **Generation:** Uses LLMs to generate Pinyin (tone-marked & numbered), definitions, parts of speech, and example sentences.
*   **Validation:** rigorous consistency checks for pinyin and data structure.
*   **Efficiency:** Caches results locally and handles API rate limits with dynamic batching.

## Usage

1.  **Setup:**
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

2.  **Run:**
    ```bash
    # Basic Ebook processing
    python3 src/main.py input.epub output.tsv --cache-dir .flashcard_cache
    ```

    For a full list of options, run:
    ```bash
    python3 src/main.py --help
    ```

## Workflow: Updating Existing Anki Cards

To backfill data for existing cards:

1.  **Export:** Export your Anki deck to a text file (e.g., `anki_export.txt`).
2.  **Extract:** Create a file with just the Hanzi, one per line (e.g., `vocab.txt`).
3.  **Generate:**
    ```bash
    python3 src/main.py vocab.txt new_data.tsv --flashcards-only --cache-dir .flashcard_cache
    ```
4.  **Import:** Import `new_data.tsv` into Anki, matching the Hanzi column to update your notes.

## Testing

Run tests:

```bash
PYTHONPATH=. ./.venv/bin/pytest
```
