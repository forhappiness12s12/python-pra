import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k
import spacy

# Load English and German tokenizers
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Define the fields
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

# Load the dataset
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

# Build the vocabulary
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Create iterators
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                 d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1, max_len=100):
        super(Transformer, self).__init__()

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.trg_tok_emb = nn.Embedding(trg_vocab_size, d_model)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(d_model, max_len))

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout)

        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def _generate_positional_encoding(self, d_model, max_len):
        positional_encoding = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                positional_encoding[pos, i] = torch.sin(pos / (10000 ** (i / d_model)))
                positional_encoding[pos, i + 1] = torch.cos(pos / (10000 ** ((i + 1) / d_model)))
        return positional_encoding.unsqueeze(0)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = trg.transpose(0, 1) == self.trg_pad_idx
        trg_len = trg.shape[0]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
        trg_mask = trg_pad_mask.unsqueeze(1) | trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_positions = torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(device)
        trg_positions = torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(device)

        src = self.src_tok_emb(src) + self.positional_encoding[:, :src_seq_len, :]
        trg = self.trg_tok_emb(trg) + self.positional_encoding[:, :trg_seq_len, :]

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        output = self.transformer(src, trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)
        output = self.fc_out(output)

        return output

# Training the model
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:-1, :])
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:-1, :])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
