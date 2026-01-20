# Mandarin Ebook Flashcards

Generate Anki-ready flashcards from Chinese ebooks using Gemini.

## Setup

1.  **Install the project:**

    ```bash
    pip install .
    ```

    Or for development:

    ```bash
    pip install -e ".[dev]"
    ```

2.  **Set up your Google API Key:**

    ```bash
    export GOOGLE_API_KEY="your_key"
    ```

## Usage

The project installs a CLI tool named `mandarin-ebook-flashcards`.

```bash
# Process ebook
mandarin-ebook-flashcards input.epub output.tsv

# Process vocab list
mandarin-ebook-flashcards vocab.txt output.tsv --flashcards-only
```

## Development

Run tests:

```bash
pytest
```

Run linting and type checking:

```bash
ruff check src
mypy src
```
