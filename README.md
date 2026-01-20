# Mandarin Ebook Flashcards

Generate Anki-ready flashcards from Chinese ebooks using Gemini.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    Wait, we removed requirements.txt. I should probably tell them to use pip install . or just list them.
    Actually, since we have pyproject.toml, they can still do `pip install .` to get dependencies even if they don't use the entry point.
    But they wanted to get rid of "install in venv things".

    If they want to run `python src/main.py`, they just need the dependencies in their environment.

    Let's update to:

    ```bash
    pip install ebooklib jieba google-genai pandas beautifulsoup4 python-dotenv tqdm toml
    ```

    Or better:

    ```bash
    pip install .
    ```
    (This still installs the package metadata but ensures dependencies are met).

    Actually, let's keep it simple:

    ```bash
    pip install -e .
    ```

    No, they said "get rid of that install in venv things".

    Okay, I will list the dependencies.

## Usage

```bash
# Process ebook
python src/main.py input.epub output.tsv

# Process vocab list
python src/main.py vocab.txt output.tsv --flashcards-only
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
