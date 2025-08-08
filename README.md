# Ebook Flashcards

This project is a Python script that extracts all the unique words from a Chinese ebook and generates flashcards for them. The flashcards are saved as a TSV file that can be imported into Anki.

## Features

*   Reads `.epub` ebooks
*   Extracts all unique words from the ebook
*   Generates flashcards with pinyin, definition, part of speech, and an example sentence
*   Saves the flashcards as a TSV file

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/ebook-flashcards.git
    ```

2.  Install the dependencies:

    ```bash
    pip install -e .
    ```

## Usage

1.  Set your Gemini API key as an environment variable:

    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    ```

2.  Run the script:

    ```bash
    python3 src/main.py my_ebook.epub my_flashcards.tsv
    ```

## Testing

To run the tests, run the following command:

```bash
pytest
```
