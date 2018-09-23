
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


class Char3Dataset(Dataset):

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


class CharModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 42)
        self.hidden_dim = 256

        self.input_fc = nn.Linear(42, self.hidden_dim)
        self.in_activation = nn.ReLU(inplace=True)

        self.hidden_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation = nn.Tanh()

        self.out_fc = nn.Linear(self.hidden_dim, vocab_size)
        self.out_activation = nn.LogSoftmax(dim=-1)

    def forward(self, seq):
        if not model.training:
            seq = seq.view(1, -1)
        bs, seq_len = seq.shape
        h = torch.autograd.Variable(torch.zeros(bs, self.hidden_dim)).to(device)
        for i in range(seq_len):
            input = self.in_activation(self.input_fc(self.embedding(seq[:, i])))
            # h_in = torch.cat((h, input), -1)
            h_in = input + h
            h = self.activation(self.hidden_fc(h_in))

        return self.out_activation(self.out_fc(h))

    def ramble(self, seq, length):
        seq = seq.view(1, -1)
        bs, seq_len = seq.shape
        h = torch.autograd.Variable(torch.zeros(bs, self.hidden_dim)).to(device)

        for i in range(seq_len):
            last = seq[:, i]
            input = self.in_activation(self.input_fc(self.embedding(seq[:, i])))
            # h_in = torch.cat((h, input), -1)
            h_in = h + input
            h = self.activation(self.hidden_fc(h_in))

        rambling = []
        for i in range(length):
            input = self.in_activation(self.input_fc(self.embedding(last)))
            # h_in = torch.cat((h, input), -1)
            h_in = h + input
            h = self.activation(self.hidden_fc(h_in))
            probs = np.exp(self.out_activation(self.out_fc(h)).cpu().detach().numpy()[0])
            i = np.random.choice(self.vocab_size, p=probs)
            rambling.append(i)
            last = torch.as_tensor(i).to(device).view(-1)

        return rambling

def get_next(txt, dataset, model, length):
    model.eval()
    c = torch.as_tensor([dataset.char2idx[char] for char in txt]).to(device)
    pred = model.ramble(c, length)
    for c in pred:
        txt = txt + dataset.idx2char[c]
    return txt


learning_rate = 1e-2
decay = 1e-6
num_epochs = 200
batch_size = 1000
seq_len = 30
grad_clip = 5

dataset = Char3Dataset(text, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CharModel(dataset.vocab_size).to(device)

loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay, momentum=0.9, nesterov=True)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
print(get_next('a', dataset, model, 100))

losses = []
for epoch in range(num_epochs):
    curr_loss = 0
    model.train()
    start = time.time()
    for i_batch, batch in enumerate(dataloader):
        seq, target = batch
        seq, target = seq.to(device), target.to(device)
        pred = model(seq)
        loss = loss_fn(pred, target)
        curr_loss = 0.9 * curr_loss + 0.1 * loss if curr_loss != 0 else loss
        losses.append(curr_loss)
        optimizer.zero_grad()
        # nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        loss.backward()
        optimizer.step()
    lr_scheduler.step(epoch)
    duration = time.time() - start
    print(f'{epoch}, {curr_loss.item():.3f}, {duration:.2f}')

    model.eval()
    print(get_next('A', dataset, model, 100))
    # print(get_next('He was a ', dataset, model, 200))
    # print(get_next('In the', dataset, model, 200))
    # print(get_next('Beginning', dataset, model, 200))
    print()
plt.plot(losses)
plt.show()
