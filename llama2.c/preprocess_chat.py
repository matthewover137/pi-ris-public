"""
Preprocesses pre-split chat history .txt files into tokenized .bin files
with an index of turn boundary positions.

Usage:
    python preprocess_chat.py --input_dir ../private --output_dir ../private

Expects:
    chat_history_train.txt
    chat_history_val.txt

Outputs:
    chat_history_train.bin   - tokenized training data (uint16)
    chat_history_val.bin     - tokenized validation data (uint16)
    turn_offsets_train.npy   - array of token positions where 'I:'/'M:' turns start (training)
    turn_offsets_val.npy     - array of token positions where 'I:'/'M:' turns start (validation)
"""

import argparse
import os
import numpy as np

from sentencepiece import SentencePieceProcessor


def tokenize_file(input_path, sp):
    """Tokenize a single .txt file and return tokens + turn offsets."""
    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"  {len(lines)} lines. Tokenizing...")

    all_tokens = []
    turn_offsets = []

    for line in lines:
        current_offset = len(all_tokens)

        if line.startswith("I: ") or line.startswith("M: "):
            turn_offsets.append(current_offset)

        tokens = sp.encode(line)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    turn_offsets = np.array(turn_offsets, dtype=np.int64)

    return all_tokens, turn_offsets


def preprocess(input_dir, output_dir, tokenizer_path="tokenizer.model"):
    sp = SentencePieceProcessor(model_file=tokenizer_path)
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val"]:
        txt_path = os.path.join(input_dir, f"chat_history_{split}.txt")
        if not os.path.exists(txt_path):
            print(f"Skipping {split}: {txt_path} not found")
            continue

        tokens, offsets = tokenize_file(txt_path, sp)

        tokens.tofile(os.path.join(output_dir, f"chat_history_{split}.bin"))
        np.save(os.path.join(output_dir, f"turn_offsets_{split}.npy"), offsets)

        print(f"  chat_history_{split}.bin  ({len(tokens)} tokens)")
        print(f"  turn_offsets_{split}.npy  ({len(offsets)} turn positions)")

        max_seq_len = 256
        usable = offsets[offsets + max_seq_len < len(tokens)]
        print(f"  Usable samples (offset + {max_seq_len} fits): {len(usable)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with chat_history_train.txt and chat_history_val.txt")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to input_dir)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.model", help="Path to tokenizer.model")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    preprocess(args.input_dir, output_dir, args.tokenizer)
