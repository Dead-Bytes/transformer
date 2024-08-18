import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn as nn
from dataset import BilingualDataset, causal_mask
from models import build_transformer
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tokenizer import Tokenizer
from tokenizer.models import WordLevel
from tokenizer.trainers import WordLevelTrainer
from tokenizer.pre_tokenizers import Whitespace
from config import get_config, get_weights_file_path
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
    train_ds_raw, val_ds_raw = random_split(ds_row, [train_ds_size, len(ds_row) - train_ds_size])

    train_ds = BilingualDataset(train_ds_raw, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in train_ds:
        src_ids = src_tokenizer.encode(item['translation'][config['src_lang']]).ids
        tgt_ids = tgt_tokenizer.encode(item['translation'][config['tgt_lang']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source language: {max_len_src}")
    print(f"Max length of target language: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def get_model(config, vocab_src_size, vocab_tgt_size):
    model = build_transformer(vocab_src_size, vocab_tgt_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    Path(config['model_folder']).mkdir(parents= True, exist_ok = True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config)

    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    writer  = SummaryWriter(log_dir = config['experiment_name'])

    optimizer  =  torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename  = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id("[PAD]"), label_smoothing= 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"processing epoch {epoch}", total = len(train_dataloader))
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_MASK = batch['encoder_mask'].to(device)
            decoder_MASK = batch['decoder_mask'].to(device)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_MASK) # (B, SeqLen, d_model)
            decoder_output = model.decode(encoder_output, encoder_MASK, decoder_input, decoder_MASK)
            projection_output = model.project(decoder_output) # (B, SeqLen, tgt_vocab_size)

            label = batch['label'].to(device) # (B, SeqLen)
            loss = loss_fn(projection_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f"loss": loss.item()})

            # log the tenorboard