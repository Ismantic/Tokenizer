#!/usr/bin/env python3
"""Filter raw_count.txt to keep only pure ASCII-letter tokens."""

import sys

def is_all_english(word: str) -> bool:
    return all(c.isascii() and c.isalpha() for c in word)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.txt> <output.txt>", file=sys.stderr)
        sys.exit(1)

    kept = 0
    total = 0
    with open(sys.argv[1], "r", encoding="utf-8") as fin, \
         open(sys.argv[2], "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and len(parts[0]) > 1 and is_all_english(parts[0]):
                fout.write(line)
                kept += 1

    print(f"Kept {kept}/{total} tokens", file=sys.stderr)

if __name__ == "__main__":
    main()
