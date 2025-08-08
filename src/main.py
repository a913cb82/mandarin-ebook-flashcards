
from pathlib import Path
from src.ebook_reader import read_ebook
from src.word_splitter import split_text_into_words
from src.flashcard_generator import generate_flashcards
from src.file_saver import save_dataframe_to_tsv

def main(ebook_path: Path, output_path: Path):
    """Main function to run the ebook flashcard generator."""
    ebook_content = read_ebook(ebook_path)
    words = split_text_into_words(ebook_content)
    unique_words = sorted(list(set(words)))
    flashcards_df = generate_flashcards(unique_words)
    save_dataframe_to_tsv(flashcards_df, output_path)

if __name__ == "__main__":
    # This is where you would parse command line arguments
    # and call the main function.
    # For now, we'll just call it with some dummy paths.
    main(Path("ebook.txt"), Path("flashcards.tsv"))
