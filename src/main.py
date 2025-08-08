import ebooklib
from ebooklib import epub
import jieba
import pandas as pd
from litellm import completion
from bs4 import BeautifulSoup
import argparse
from io import StringIO

SYSTEM_PROMPT = "TODO read prompt.txt"

def read_epub(file_path):
    book = epub.read_epub(file_path)
    content = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        content.append(soup.get_text())
    return "\n".join(content)

def extract_words(text):
    return list(jieba.cut(text))

def create_flashcards(words):
    response = completion(
        model="gemini/gemini-pro",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ",".join(words)},
        ],
    )
    # The response is a TSV string, so we can use pandas to parse it
    return pd.read_csv(StringIO(response.choices[0].message.content), sep='\t', header=None)

def save_flashcards(flashcards, file_path):
    flashcards.to_csv(file_path, sep='\t', index=False, header=False)

def main():
    parser = argparse.ArgumentParser(description='Create Anki flashcards from a Chinese ebook.')
    parser.add_argument('ebook_path', type=str, help='The path to the ebook file.')
    parser.add_argument('output_path', type=str, help='The path to save the flashcards.')
    args = parser.parse_args()

    content = read_epub(args.ebook_path)
    words = extract_words(content)
    flashcards = create_flashcards(words)
    save_flashcards(flashcards, args.output_path)

if __name__ == '__main__':
    main()
