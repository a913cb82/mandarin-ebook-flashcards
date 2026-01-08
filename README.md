# Mandarin Ebook Flashcards

Generate Anki-ready flashcards from Chinese ebooks using Gemini.

## Setup
```bash
export GOOGLE_API_KEY="your_key"
pip install -r requirements.txt
```

## Usage
```bash
# Process ebook
python3 main.py input.epub output.tsv

# Process vocab list
python3 main.py vocab.txt output.tsv --flashcards-only
```

## Test
```bash
pytest
```