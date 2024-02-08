import os
import pandas as pd
import urllib.request
from tokenizers import SentencePieceBPETokenizer
from pathlib import Path
from datasets import load_dataset
import time

def save_dataset_to_txt(txt_column, txt_dir, hf_dataset_id):
    dataset = load_dataset(hf_dataset_id)
    os.makedirs(txt_dir, exist_ok=True)
    for split_key in dataset.keys():
        doc_path = f"{txt_dir}/{split_key}.txt"
        with open(doc_path, 'w') as f:
            for doc in dataset[split_key][txt_column]:
                f.write(doc+'\n')

def main():

    path_corpus = path_corpus1 + path_corpus2 + path_corpus23
    tokenizer = SentencePieceBPETokenizer(fuse_unk=True)
    #tokenizer = ByteLevelBPETokenizer(unicode_normalizer="nfkc", trim_offsets=True)


    vocab_size = 18000
    limit_alphabet = 1000
    min_frequency = 30
    t1 = time.time()

    ### For BPE
    tokenizer.train(
        files=path_corpus,
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>"],
        min_frequency=min_frequency, 
        limit_alphabet=limit_alphabet,
        show_progress=True
    )
    
    ### For BBPE
    # tokenizer.train(
    #     files=path_corpus,
    #     vocab_size=vocab_size,
    #     special_tokens=["<|endoftext|>"],
    #     min_frequency=min_frequency, 
    #     show_progress=True
    # )
    

    tokenizer.save('korean_tokenizer.json') 
    print("Elapsed time:", time.time() - t1)

if __name__ == '__main__':
    main()