import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn as nn
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import get_config, get_weights_file_path
from pathlib import Path
import warnings


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, src_tokenizer, tgt_tokenizer, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    #size of the control window

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size should be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, tgt_tokenizer, max_len, device)
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Expected: {target_text}")
            print_msg(f"Predicted: {model_out_text}")

            if count >= num_examples:
                break

    if writer:
        # Torch Metrics , charError Rate, BLEU Score, word ERror rate
        pass

def get_all_senetences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
    
def get_or_build_tokenizer(config, ds, lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang))
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

    for item in ds_row:
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
        batch_iterator = tqdm(train_dataloader, desc = f"processing epoch {epoch}", total = len(train_dataloader))
        for batch in batch_iterator:
            model.train()

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
            writer.add_scalar(' train loss', loss.item(), global_step)
            writer.flush()

            # backpropagation
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            # def run_validation(model, validation_ds, src_tokenizer, tgt_tokenizer, max_len, device, print_msg, global_state, writer, num_examples=2):

            run_validation(model, val_dataloader, src_tokenizer, tgt_tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=2)

            global_step += 1

        # save the model
    model_filename = get_weights_file_path(config, epoch)
    torch.save({
        'epoch' : epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
  
    train_model(config)
