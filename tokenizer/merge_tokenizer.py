import os
import json
from transformers import AutoTokenizer
import argparse
from pathlib import Path

def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()  

    # path
    parser.add_argument("--orig_tokenizer_path", type=str, default="./token-phi-ko")
    parser.add_argument("--merged_tokenizer_save_path", type=str, default="./tokenizer-phi-2-merged")
    parser.add_argument("--added_tokenizer_json", type=str, default="korean_tokenizer_bbpe.json")

    args = parser.parse_known_args()
    return args


def main(args):

    orig_tokenizer_path = args.orig_tokenizer_path
    merged_tokenizer_save_path = args.merged_tokenizer_save_path
    added_tokenizer_json = args.added_tokenizer_json

    tokenizer = AutoTokenizer.from_pretrained(orig_tokenizer_path)
    tokenizer.save_pretrained(merged_tokenizer_save_path)

    with open(f"./{orig_tokenizer_path}/tokenizer.json", "r") as json_file:
        data_orig = json.load(json_file)

    with open(added_tokenizer_json, "r") as json_file:
        data_new = json.load(json_file)

    vocab_orig = data_orig['model']['vocab']
    vocab_new = data_new['model']['vocab']
    vocab_merge = vocab_orig.copy()
    idx = len(vocab_orig)

    for word in vocab_new.keys():
        if word not in vocab_orig.keys():
            vocab_merge[word] = idx
            idx += 1

    merges_orig = data_orig['model']['merges']
    merges_new = data_new['model']['merges']

    data_merge = data_orig.copy()
    data_merge['model']['vocab'] = vocab_merge
    data_merge['model']['merges'] = merges_orig + merges_new

    with open(f"{merged_tokenizer_save_path}/tokenizer.json", "w") as jsonFile:
        json.dump(data_merge, jsonFile, indent=2, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(merged_tokenizer_save_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_tokenizer_save_path)

if __name__ == '__main__':
    args, _ = parse_args()
    print(args)
    main(args)
