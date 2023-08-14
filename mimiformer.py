# %%
!python3 --version
import random
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm
import datetime
import numpy as np
from datasets import load_dataset
import time

dataset = load_dataset("roneneldan/TinyStories")

# Generate the same random numbers
torch.manual_seed(42)

# %%
sample_data = dataset["train"]['text'][:1000]

with open('sentences.txt', 'w') as f:
    for sentence in sample_data:
        if sentence.strip():  # Check if the stripped sentence is not empty
            f.write(sentence)  # Write the sentence to the file with a newline
            
with open("sentences.txt", "r") as f:
    lines = f.readlines()

# Remove empty lines
lines = [line for line in lines if line.strip()]

with open("sentences.txt", "w") as f:
    f.writelines(lines)

# %%
input_file = 'sentences.txt' 
prefix = 'sentences'
vocab_size = 2000

# 

spm.SentencePieceTrainer.train(
    input=input_file, 
    model_prefix=prefix, 
    vocab_size=vocab_size
)

# %%
class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{prefix}.model')
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, name):
        return self.sp.encode_as_ids(name)

    def decode(self, tokens):
        return self.sp.decode_ids(tokens)
    
tokenizer = Tokenizer()

# for i in range(10):
#     print(tokenizer.decode([i]))

print(tokenizer.vocab_size)


# %%

class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    with open('sentences.txt', 'r') as f:
      self.names = f.read().split('\n')
    self.tokenizer = Tokenizer()

  def __len__(self):
    # Return number of names
    return len(self.names)

  def __getitem__(self, idx):
    # Get name at index
    name = self.names[idx]
    # Return encoded name
    return torch.tensor(self.tokenizer.encode(name), dtype=torch.long)


ds = Dataset()
# dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=)
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

# %%
class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.stories = data["story"]
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        return torch.tensor(self.tokenizer.encode(story))

# train_dataset = TinyStoriesDataset(dataset["train"])
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)


# %% [markdown]
# BESFORMER

# %%
embedding_size = 32
hidden_size = 11
mask_dimensions = 1000
dropout_rate = 0.1

class BesSimpleTransformer(torch.nn.Module):
  def __init__(self):
    super(BesSimpleTransformer, self).__init__()
    # Embedding part of the model - 7 is the embedding size
    self.embedding    = torch.nn.Embedding(tokenizer.vocab_size, embedding_size)
    self.pos_emb      = self.get_pos_matrix()
    self.dropout = torch.nn.Dropout(dropout_rate)
    # Mask tensor trick - if batch size is one, we might not need it - research it!
    self.register_buffer('mask', torch.tril(torch.ones(mask_dimensions, mask_dimensions)))
    # First decoder block
    # 11 could be anything, if we have heads or batch_size this might change
    self.layer_00_key = torch.nn.Linear(embedding_size, 11)
    self.layer_00_qry = torch.nn.Linear(embedding_size, 11)
    self.layer_00_val = torch.nn.Linear(embedding_size, 11)
    self.layer_00_ffw = torch.nn.Linear(11, embedding_size)
    self.layer_norm0 = torch.nn.LayerNorm(embedding_size)
    # Second decoder block
    self.layer_01_key = torch.nn.Linear(embedding_size, 11)
    self.layer_01_qry = torch.nn.Linear(embedding_size, 11)
    self.layer_01_val = torch.nn.Linear(embedding_size, 11)
    self.layer_01_ffw = torch.nn.Linear(11, embedding_size)
    self.layer_norm1 = torch.nn.LayerNorm(embedding_size)
    # Output of the model
    self.layer_norm2 = torch.nn.LayerNorm(embedding_size)
    self.map_to_vocab = torch.nn.Linear(embedding_size, tokenizer.vocab_size)

  def forward(self, x):
    emb = self.embedding(x)
    pos = self.pos_emb[0:x.shape[0], :]
    emb = emb + pos
    
    # normalise
    emb = self.layer_norm0(emb)

    key = self.layer_00_key(emb)
    qry = self.layer_00_qry(emb)
    val = self.layer_00_val(emb)
    att = torch.mm(qry, key.t())

    att = att / math.sqrt(key.shape[1])

    # mask from 0 to token end (square mask)
    msk = self.mask[0:x.shape[0], 0:x.shape[0]]
    # mask over tensor (same as adding it)
    att = att.masked_fill(msk == 0, float('-inf'))
    
    att = torch.nn.functional.softmax(att, dim=1)
    att_00 = att
    res = torch.mm(att, val)
    # this is the feed forward layer
    res = self.dropout(res)
    res = self.layer_00_ffw(res)
    # add residual
    res = res + emb
    res1 = res
    # normalise
    res = self.layer_norm1(res)

    # do it all again with new q, k, v
    key = self.layer_01_key(res)
    qry = self.layer_01_qry(res)
    val = self.layer_01_val(res)
    att = torch.mm(qry, key.t())
    att = att / math.sqrt(key.shape[1])

    msk = self.mask[0:x.shape[0], 0:x.shape[0]]
    att = att.masked_fill(msk == 0, float('-inf'))
    
    att = torch.nn.functional.softmax(att, dim=1)
    att_01 = att
    res = torch.mm(att, val)
    res = self.dropout(res)
    res = self.layer_01_ffw(res)
    # add and normalise
    res = res1 + self.layer_norm2(res)

    # map back to our 29 vocab (alphabet + pos, eos, sos)
    out = self.map_to_vocab(res)
    return out, [att_00, att_01]

  def get_pos_matrix(self):
    store = torch.zeros(mask_dimensions, embedding_size)
    for pos in range(mask_dimensions):
      # why do we do this range thing here?
      for i in range(0, 7, 2):
        denominator = 10000 ** (2 * i / 7)
        store[pos, i] = math.sin(pos / denominator)
        if i + 1 < 7: store[pos, i + 1] = math.cos(pos / denominator)
    return store


m = BesSimpleTransformer()

# SDG instead of Adam, why?
opt = torch.optim.Adam(m.parameters(), lr=0.01)

# %%
loss_history = []
num_epochs = 10
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"models/{timestamp}_mimiformer.pth"

start = time.time()

for epoch in range(num_epochs):
  for idx, batch in enumerate(dl):

    # sos = torch.tensor([2])
    # eos = torch.tensor([1])
    # Derive sos and eos from tokenizer
    sos = torch.tensor([tokenizer.sp.piece_to_id('<s>')], dtype=torch.long)
    eos = torch.tensor([tokenizer.sp.piece_to_id('</s>')], dtype=torch.long)

    # for each row in batch
    x = batch[0]
    # add sos to beginning of row
    x = torch.cat([sos, x])
    # In target tensor, add eos to end of each row and remove sos from start
    y = torch.cat([x[1:], eos])

    
    # run our batch through the whole transformer (attention1, ffw, attention2, ffw, linear)
    p, _ = m(x)
    # calculate cross-entropy loss between predicted token and target token
    l = torch.nn.functional.cross_entropy(p, y)
    # print loss every 1000 rows of dataset
    if idx % 1000 == 0: 
      loss_history.append(l.item())
      print("Loss:", l.item())
    # backpropogate for every row
    l.backward()
    # What is the optimiser doing?  Why is it called after we have done backward pass?
    opt.step()
    opt.zero_grad()
  
  # save model after each epoch
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  torch.save(m.state_dict(), save_path)

end = time.time()
print(f"Training took {end - start:.2f} seconds")
  
  

# %%
xaxis = len(loss_history) + 1
print(xaxis)
plt.plot(range(1, xaxis), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss every 1000 Datapoints')
plt.grid(True)
plt.show()

# %%

tokens = []
# generate a random start token for testing
# Get all the tokens/pieces from SentencePiece
all_tokens = [tokenizer.sp.id_to_piece(i) for i in range(tokenizer.vocab_size)]

# Pick a random token/piece
random_token = random.choice(all_tokens)

# Convert the token to its corresponding ID
x = torch.tensor([tokenizer.sp.piece_to_id(random_token)])
print(tokenizer.decode(x.tolist()))

sos = torch.tensor([tokenizer.sp.piece_to_id('<s>')])
print("sos", sos.tolist())
print("sos", sos)
x = torch.cat([sos, x])

top_p_threshold = 0.5

while True:
  tokens.append(x[-1].tolist())
  print("token", tokens)
  # run our random start through transformer and get attention matricies out
  p, attention = m(x)
  # create probabilities from 29 token options
  p = torch.nn.functional.softmax(p, dim=1)
  probs = p[-1].tolist()
  # choose the best prediction (most probable next token according to tranformer)
  p = torch.argmax(p, dim=1)

  #choose from the top 5 most probable tokens
  # sorted_probs = sorted(probs, reverse=True)
  # top_p = 0
  # for i in range(len(sorted_probs)):
  #   top_p += sorted_probs[i]
  #   if top_p > top_p_threshold:
  #     break
  # top_p = sorted_probs[:i+1]
  # top_p = torch.tensor(top_p)
  # p = torch.multinomial(top_p, 1)
  # p = torch.multinomial(p, 1)

  # unsqueeze to get the right dimensions for torch.cat
  # dimensions of x = [1, 1] and dimensions of p = [1]
  x = torch.cat([x, p[-1].unsqueeze(0)])
  if p[-1] == 1 or len(p.tolist()) == 17: break
print("Generate:", tokenizer.decode(x.tolist()))

# %%

def plot_attention_heatmap(attention, tokens):
    decoded_tokens = [tokenizer.decode([tok]) for tok in tokens]
    num_layers = len(attention)
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))

    for idx, att in enumerate(attention):
        sns.heatmap(att.detach().numpy(), annot=False, cmap='viridis', ax=axes[idx], xticklabels=decoded_tokens, yticklabels=decoded_tokens)
        axes[idx].set_title(f'Attention Layer {idx + 1}')
        axes[idx].set_xticks(np.arange(len(decoded_tokens))+.5, minor=False)
        axes[idx].set_yticks(np.arange(len(decoded_tokens))+.5, minor=False)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.show()


# Assuming that you've tokenized 'x' and the tokens are stored in a variable called 'tokens'
plot_attention_heatmap(attention, tokens)

print(attention[0].shape)


