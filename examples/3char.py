
import os
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

data_dir = Path(os.getcwd()) / '..' / 'data' / 'wikitext-2-raw'
train_path = data_dir / 'wiki.test.raw'

text = open(train_path, encoding='utf8').read()


class Char3Dataset(Dataset):

    def __init__(self, text):
        chars = list(sorted(set(text)))
        chars.insert(0, "\0")

        self.vocab_size = len(chars)
        print(self.vocab_size)

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

    def forward(self, *seq):
        # in1 = F.relu(self.input_fc(self.embedding(c1)))
        # in2 = F.relu(self.input_fc(self.embedding(c2)))
        # in3 = F.relu(self.input_fc(self.embedding(c3)))

        # h = torch.autograd.Variable(torch.zeros(in1.size()))
        # h = self.activation(self.hidden_fc(h+in1))
        # h = self.activation(self.hidden_fc(h+in2))
        # h = self.activation(self.hidden_fc(h+in3))

        h = torch.autograd.Variable(torch.zeros(self.hidden_dim).cuda())
        for x in seq:
            input = self.in_activation(self.input_fc(self.embedding(x)))
            h = self.activation(self.hidden_fc(h+input))

        return self.out_activation(self.out_fc(h))

def get_next(txt, dataset, model):
    c = torch.as_tensor([dataset.char2idx[char] for char in txt]).cuda()
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
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
model = Char3Model(dataset.vocab_size)
model.cuda()

learning_rate = 1e-3
decay = 1e-5
# loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

# print(get_next('hel', dataset, model))
import matplotlib.pyplot as plt

num_epochs = 50
losses = []
for epoch in range(num_epochs):
    curr_loss = 0
    model.train()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    for i_batch, batch in enumerate(dataloader):
        c1, c2, c3, c4 = batch
        c1, c2, c3, c4 = c1.cuda(), c2.cuda(), c3.cuda(), c4.cuda()
        pred = model(c1, c2, c3)
        loss = loss_fn(pred, c4)
        curr_loss = 0.99 * curr_loss + 0.01 * loss if curr_loss != 0 else loss
        losses.append(curr_loss)
        optimizer.zero_grad()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        loss.backward()
        optimizer.step()
    # plt.plot(losses)
    # plt.show()
    print(curr_loss.item())

    model.eval()
    txt = 'and'
    for i in range(100):
        txt += get_next(txt[-3:], dataset, model)
    print(txt)
    print()
    txt = 'and'
    for i in range(100):
        txt += get_max_next(txt[-3:], dataset, model)
    print(txt)

print(get_next('hel', dataset, model))
print(get_next(' th', dataset, model))
print(get_next('and', dataset, model))

