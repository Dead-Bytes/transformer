import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn as nn

from datasets import load_dataset
from tokenizer import Tokenizer
from tokenizer.models import WordLevel
from tokenizer.trainers import WordLevelTrainer
from tokenizer.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_senetences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
    
def get_or_build_tokenizer(config, ds, lang):

    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = "<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[EOS]", "[SOS]", "[PAD]", "<unk>"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_senetences(ds, lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_row = load_dataset('opus_books', f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')

    #build tokenizer

    src_tokenizer = get_or_build_tokenizer(config, ds_row, config['src_lang'])
    tgt_tokenizer = get_or_build_tokenizer(config, ds_row, config['tgt_lang'])

    # split in 90:10 for training & Validation

    train_ds_size = int(len(ds_row) * 0.9)
    train_ds, val_ds = random_split(ds_row, [train_ds_size, len(ds_row) - train_ds_size])