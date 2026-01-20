# Mandarin Ebook Flashcards

Generate Anki-ready flashcards from Chinese ebooks using Gemini.

## Setup

1.  **Install dependencies from `pyproject.toml`:**

    ```bash
    pip install .
    ```

2.  **Set up your Google API Key:**

    ```bash
    export GOOGLE_API_KEY="your_key"
    ```

## Usage

```bash
# Process ebook
python src/main.py input.epub output.tsv

# Process vocab list
python src/main.py vocab.txt output.tsv --flashcards-only
```

## Development

1.  **Install development dependencies and pre-commit hooks:**

    ```bash
    pip install ".[dev]"
    pre-commit install
    ```

2.  **Run checks:**

    ```bash
    pytest
    ruff check src
    mypy src
    ```