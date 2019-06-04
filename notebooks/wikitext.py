#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass

#%%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#%% [markdown]
# ## Load wikitext-2 dataset

#%%
import os

data_dir = os.path.join(os.getcwd(), '..', 'data', 'wikitext-2-raw')
train_path = os.path.join(data_dir, 'wiki.test.raw')


#%%
text = open(train_path, encoding='utf8').read()


#%%
text[:500]


#%%
chars = list(sorted(set(text)))
# for padding
chars.insert(0, "\0")
''.join(chars)


#%%
vocab_size = len(chars)
print(f'Total characters: {vocab_size}')


#%%
# Maps
char_idx = {char: idx for idx, char in enumerate(chars)}
idx_char = {idx: char for idx, char in enumerate(chars)}


#%%
idx_text = [char_idx[char] for char in text]


#%%
print(idx_text[:100])
print([idx_char[idx] for idx in idx_text[:100]])


#%%
import torch
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, text):
        
        chars = list(sorted(set(text)))
        chars.insert(0, "\0")
        
        vocab_size = len(chars)
        # Maps
        char_idx = {char: idx for idx, char in enumerate(chars)}
        idx_char = {idx: char for idx, char in enumerate(chars)}
        
        idx_text = [char_idx[char] for char in text]
        
        cs=3
        c1_dat = [idx_text[i]   for i in range(0, len(idx_text)-cs, cs)]
        c2_dat = [idx_text[i+1] for i in range(0, len(idx_text)-cs, cs)]
        c3_dat = [idx_text[i+2] for i in range(0, len(idx_text)-cs, cs)]
        c4_dat = [idx_text[i+3] for i in range(0, len(idx_text)-cs, cs)]
        
        x = np.vstack((np.stack(c1_dat), np.stack(c2_dat), np.stack(c3_dat))).T
        y = np.stack(c4_dat)
        
        self.x = torch.from_numpy(x).type(torch.LongTensor)
        self.y = torch.from_numpy(y)
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


#%%
dataset = Dataset(text)
data_generator = data.DataLoader(dataset, batch_size=12, shuffle=True)


#%%
n_hidden = 256
n_fac = 42
from torch import nn
import torch.nn.functional as F

class Char3Model(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)

        # The 'green arrow' from our diagram - the layer operation from input to hidden
        self.l_in = nn.Linear(n_fac, n_hidden)

        # The 'orange arrow' from our diagram - the layer operation from hidden to hidden
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        
        # The 'blue arrow' from our diagram - the layer operation from hidden to output
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, x):
        c1 = x.narrow(1, 0, 1)
        c2 = x.narrow(1, 1, 1)
        c3 = x.narrow(1, 2, 1)
        print(c1.shape)
        print(self.e(c1).shape)
        in1 = F.relu(self.l_in(self.e(c1)))
        print(in1.shape)
        in2 = F.relu(self.l_in(self.e(c2)))
        in3 = F.relu(self.l_in(self.e(c3)))
        
        h = torch.autograd.Variable(torch.zeros(in1.size()))
        h = F.tanh(self.l_hidden(h+in1))
        h = F.tanh(self.l_hidden(h+in2))
        h = F.tanh(self.l_hidden(h+in3))
        
        return F.log_softmax(self.l_out(h))


#%%
model = Char3Model(vocab_size, n_fac)
loss_fn = F.nll_loss
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#%%
max_epochs = 10
for epoch in range(max_epochs):
    for x, y in data_generator:
#         x = torch.autograd.Variable(x)
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        print(t, loss.item())

        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()


#%%



#%%



