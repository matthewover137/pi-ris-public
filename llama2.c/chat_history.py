#ADDED CODE - MLi
import argparse
import os
import numpy as np

from tokenizer import Tokenizer

train_file = "./../private/chat_history_train.txt"
val_file = "./../private/chat_history_val.txt"

def pretokenize():
    enc = Tokenizer()
    with open(train_file, "r") as f:
        train_tokens = enc.encode(f.read(), bos=False, eos=False)
    with open(val_file, "r") as f:
        val_tokens = enc.encode(f.read(), bos=False, eos=False)
    
    train_tokens = np.array(train_tokens, dtype=np.uint16)
    val_tokens = np.array(val_tokens, dtype=np.uint16)

    train_bin = train_file.replace(".txt", ".bin")
    val_bin = val_file.replace(".txt", ".bin")

    with open(train_bin, "wb") as f:
        f.write(train_tokens.tobytes())
    with open(val_bin, "wb") as f:
        f.write(val_tokens.tobytes())
    
    print(f"Saved tokenized train: {train_bin} and val: {val_bin}")

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python chat_history.py pretokenize
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "pretokenize":
        pretokenize()
    else:
        raise ValueError(f"Unknown stage {args.stage}")
