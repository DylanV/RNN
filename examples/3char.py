
import os
from pathlib import Path
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

data_dir = Path(os.getcwd()) / '..' / 'data' / 'wikitext-2-raw'
train_path = data_dir / 'wiki.test.raw'

text = open(train_path, encoding='utf8').read()
device = 'cuda'


class Char3Dataset(Dataset):

    def __init__(self, text):
        chars = list(sorted(set(text)))
        chars.insert(0, "\0")

        self.vocab_size = len(chars)

        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}

        idx_text = [self.char2idx[char] for char in text]

        cs = 3
        self.c1 = torch.as_tensor([idx_text[i] for i in range(0, len(idx_text) - cs, cs)])
        self.c2 = torch.as_tensor([idx_text[i + 1] for i in range(0, len(idx_text) - cs, cs)])
        self.c3 = torch.as_tensor([idx_text[i + 2] for i in range(0, len(idx_text) - cs, cs)])
        self.c4 = torch.as_tensor([idx_text[i + 3] for i in range(0, len(idx_text) - cs, cs)])

    def __len__(self):
        return self.c1.shape[0]

    def __getitem__(self, item):
        return self.c1[item], self.c2[item], self.c3[item], self.c4[item]


class Char3Model(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 42)
        self.hidden_dim = 256

        self.input_fc = nn.Linear(42, 256)
        self.in_activation = nn.ReLU(inplace=True)

        self.hidden_fc = nn.Linear(256, 256)
        self.activation = nn.Tanh()

        self.out_fc = nn.Linear(256, vocab_size)
        self.out_activation = nn.LogSoftmax(dim=-1)
        self.hidden_state = torch.autograd.Variable(torch.zeros(self.hidden_dim)).to(device)

    def forward(self, *seq):
        # in1 = F.relu(self.input_fc(self.embedding(c1)))
        # in2 = F.relu(self.input_fc(self.embedding(c2)))
        # in3 = F.relu(self.input_fc(self.embedding(c3)))

        # h = torch.autograd.Variable(torch.zeros(in1.size()))
        # h = self.activation(self.hidden_fc(h+in1))
        # h = self.activation(self.hidden_fc(h+in2))
        # h = self.activation(self.hidden_fc(h+in3))

        if self.training:
            h = torch.autograd.Variable(torch.zeros(self.hidden_dim)).to(device)
        else:
            h = self.hidden_state

        for x in seq:
            input = self.in_activation(self.input_fc(self.embedding(x)))
            h = self.activation(self.hidden_fc(h+input))

        if not self.training:
            self.hidden_state = h

        return self.out_activation(self.out_fc(h))

def get_next(txt, dataset, model):
    c = torch.as_tensor([dataset.char2idx[char] for char in txt]).to(device)
    c1, c2, c3 = c
    pred = model(c1, c2, c3)
    p = np.exp(pred.to('cpu').detach().numpy())
    i = np.random.choice(model.vocab_size, p=p)
    return dataset.idx2char[i]

def get_max_next(txt, dataset, model):
    c = torch.as_tensor([dataset.char2idx[char] for char in txt]).cuda()
    c1, c2, c3 = c
    pred = model(c1, c2, c3)
    i = np.argmax(pred.to('cpu').detach().numpy())
    # i = np.random.choice(model.vocab_size, p=p)
    return dataset.idx2char[i]

dataset = Char3Dataset(text)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
model = Char3Model(dataset.vocab_size)
model.to(device)

learning_rate = 1e-2
decay = 1e-6
num_epochs = 200
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

import matplotlib.pyplot as plt

clip = 5
losses = []
for epoch in range(num_epochs):
    curr_loss = 0
    model.train()
    start = time.time()
    for i_batch, batch in enumerate(dataloader):
        c1, c2, c3, c4 = batch
        c1, c2, c3, c4 = c1.to(device), c2.to(device), c3.to(device), c4.to(device)
        pred = model(c1, c2, c3)
        loss = loss_fn(pred, c4)
        curr_loss = 0.99 * curr_loss + 0.01 * loss if curr_loss != 0 else loss
        losses.append(curr_loss)
        optimizer.zero_grad()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        loss.backward()
        optimizer.step()
    lr_scheduler.step(epoch)
    duration = time.time() - start
    print(f'{epoch}, {curr_loss.item():.3f}, {duration:.2f}')

    model.eval()
    txt = 'and'
    for i in range(300):
        txt += get_next(txt[-3:], dataset, model)
    print(txt)
    print()
plt.plot(losses)
plt.show()
