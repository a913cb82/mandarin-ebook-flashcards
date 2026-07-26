#!/usr/bin/env bash
set -euo pipefail

cache_dir="${1:-.flashcard_cache}"
n="${2:-5}"

total=$(find "$cache_dir" -name '*.json' | wc -l)
recent_1m=$(find "$cache_dir" -name '*.json' -mmin -1 | wc -l)
recent_5m=$(find "$cache_dir" -name '*.json' -mmin -5 | wc -l)
recent_60m=$(find "$cache_dir" -name '*.json' -mmin -60 | wc -l)
recent_1d=$(find "$cache_dir" -name '*.json' -mmin -1440 | wc -l)

printf "Total: %d  |  +%-3d (1m)  +%-3d (5m)  +%-3d (60m)  +%-3d (1d)\n\n" \
    "$total" "$recent_1m" "$recent_5m" "$recent_60m" "$recent_1d"

printf "Newest %d cards:\n\n" "$n"

{ printf "Hanzi\tPinyin\tDefinition\tPOS\tSentence\n"
  ls -t "$cache_dir"/*.json | head -n "$n" | while read -r f; do
      jq -r '[.hanzi, .pinyin, .definition, .partofspeech, .sentencehanzi] | @tsv' "$f"
  done
} | column -t -s $'\t'
