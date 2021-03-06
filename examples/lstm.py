
import os
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

data_dir = Path(os.getcwd()) / '..' / 'data' / 'wikitext-2-raw'
train_path = data_dir / 'wiki.test.raw'

text = open(train_path, encoding='utf8').read()
device = 'cuda'


class CharDataset(Dataset):

    def __init__(self, text, seq_len):
        chars = list(sorted(set(text)))
        chars.insert(0, "\0")

        self.vocab_size = len(chars)

        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}

        idx_text = [self.char2idx[char] for char in text]

        seqs = [[idx_text[i+j] for i in range(seq_len)] for j in range(len(idx_text) - seq_len)]
        targets = [idx_text[j + seq_len] for j in range(len(idx_text) - seq_len)]

        self.seqs = torch.as_tensor(seqs)
        self.targets = torch.as_tensor(targets)

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, item):
        return self.seqs[item], self.targets[item]


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # Layer dimensions
        self.in_dim = input_dim
        self.hidden_dim = hidden_dim

        # Forget gate
        self.forget_gate = nn.Linear(self.in_dim + self.hidden_dim, self.hidden_dim)
        self.forget_activation = nn.Sigmoid()

        # Input gate
        self.input_gate = nn.Linear(self.in_dim + self.hidden_dim, self.hidden_dim)
        self.input_activation = nn.Sigmoid()

        # Candidate gate
        self.candidate_gate = nn.Linear(self.in_dim + self.hidden_dim, self.hidden_dim)
        self.candidate_activation = nn.Tanh()

        # Output gate
        self.output_gate = nn.Linear(self.in_dim + self.hidden_dim, self.hidden_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, sequence, h_init=None, c_init=None):
        bs, seq_len, _ = sequence.shape
        # Initialise the hidden state with all zeros if no initial state is given
        h = torch.autograd.Variable(torch.zeros(bs, self.hidden_dim)).to(device) if h_init is None else h_init
        c = torch.autograd.Variable(torch.zeros(bs, self.hidden_dim)).to(device) if c_init is None else c_init

        # Go through the sequence
        for i in range(seq_len):
            inh = torch.cat((h, sequence[:, i]), 1)
            f = self.forget_activation(self.forget_gate(inh))
            i = self.input_activation(self.input_gate(inh))
            c_cand = self.input_activation(self.input_gate(inh))
            c = f * c + i * c_cand
            o = self.input_activation(self.input_gate(inh))
            h = o * torch.tanh(c)

        return h, c


class CharModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = RNN(embed_dim, hidden_dim)
        self.hidden_2_out = nn.Linear(hidden_dim, vocab_size)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, sequence, h=None, c=None):
        hidden_state, c_state = self.rnn(self.embedding(sequence), h, c)
        return self.out(self.hidden_2_out(hidden_state)), hidden_state, c_state


def get_next(start_char, dataset, model, length):
    model.eval()

    txt = start_char
    h, c = None, None
    for _ in range(length):
        char_seq = torch.as_tensor([dataset.char2idx[txt[-1]]]).to(device).view(1, -1)
        pred, h, c = model(char_seq, h, c)
        probs = np.exp(pred.detach().to('cpu').numpy())[0]
        i = np.random.choice(model.vocab_size, p=probs)
        txt += dataset.idx2char[i]

    return txt


learning_rate = 1e-2
decay = 1e-6
num_epochs = 200
batch_size = 1000
seq_len = 30
grad_clip = 5

dataset = CharDataset(text, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CharModel(dataset.vocab_size, 42, 256).to(device)

print(get_next('a', dataset, model, 100))

loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay, momentum=0.9, nesterov=True)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

losses = []
for epoch in range(num_epochs):
    curr_loss = 0
    model.train()
    start = time.time()
    for i_batch, batch in enumerate(dataloader):
        seq, target = batch
        seq, target = seq.to(device), target.to(device)
        pred, _, _ = model(seq)
        loss = loss_fn(pred, target)
        curr_loss = 0.9 * curr_loss + 0.1 * loss if curr_loss != 0 else loss
        losses.append(curr_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step(epoch)
    duration = time.time() - start
    print(f'{epoch}, {curr_loss.item():.3f}, {duration:.2f}')

    model.eval()
    print(get_next('A', dataset, model, 100))
    print()
plt.plot(losses)
plt.show()

