import argparse
import os

from dotenv import load_dotenv

from flashcard import create_flashcards, save_flashcards
from utils import extract_vocabulary, load_subtlex_global_freqs, read_epub


def main() -> None:
    """CLI entry point for mandarin-ebook-flashcards."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Create Anki flashcards from Chinese text."
    )
    parser.add_argument("input_path", help="Path to input (EPUB or txt list)")
    parser.add_argument("output_path", help="Path to save flashcards (TSV)")
    parser.add_argument("--initial-batch-size", type=int, default=100)
    parser.add_argument("--batch-size-multiplier", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--vocab-only", action="store_true")
    parser.add_argument("--flashcards-only", action="store_true")
    parser.add_argument("--stop-words", help="Path to stop words file")
    parser.add_argument(
        "--comprehension",
        type=float,
        default=0.98,
        help="Vocabulary coverage threshold (default: 0.98)",
    )
    parser.add_argument(
        "--global-freqs",
        default="data/SUBTLEX-CH-WF",
        help="Path to SUBTLEX-CH-WF file (default: data/SUBTLEX-CH-WF)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cache-dir", default=".flashcard_cache")
    parser.add_argument(
        "--use-gemini-cache",
        action="store_true",
        help="Enable Gemini context caching (reduces per-request tokens)",
    )
    args = parser.parse_args()

    global_freqs = {}
    if os.path.exists(args.global_freqs):
        global_freqs = load_subtlex_global_freqs(args.global_freqs)

    if args.flashcards_only:
        with open(args.input_path) as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        text = (
            read_epub(args.input_path)
            if args.input_path.endswith(".epub")
            else open(args.input_path).read()
        )
        words = extract_vocabulary(
            text,
            stop_words_path=args.stop_words,
            comprehension_pct=args.comprehension,
            verbose=args.verbose,
            global_freqs=global_freqs,
        )

    if args.vocab_only:
        with open(args.output_path, "w") as f:
            f.write("\n".join(words))
        return

    flashcards = create_flashcards(
        words,
        cache_dir=args.cache_dir,
        initial_batch_size=args.initial_batch_size,
        batch_size_multiplier=args.batch_size_multiplier,
        retries=args.retries,
        model=args.model,
        verbose=args.verbose,
        use_gemini_cache=args.use_gemini_cache,
    )
    save_flashcards(flashcards, args.output_path)


if __name__ == "__main__":
    main()
