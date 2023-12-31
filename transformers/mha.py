# %% 
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
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Ensure Deterministic Behavior
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %% [markdown]
# Download Dataset

# %%
# Load Dataset
dataset = load_dataset("roneneldan/TinyStories")

num_lines = 100000

sample_data = dataset["train"]['text'][:num_lines]

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

# %% [markdown]
# Tokenize

# %%
input_file = 'sentences.txt' 
prefix = 'sentences'
vocab_size = 5000

spm.SentencePieceTrainer.train(
    input=input_file, 
    model_prefix=prefix, 
    vocab_size=vocab_size
)


class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{prefix}.model')
        self.vocab_size = self.sp.get_piece_size()

        self.SOS_TOKEN = self.sp.piece_to_id('<s>')
        self.EOS_TOKEN = self.sp.piece_to_id('</s>')

    def encode(self, name):
        return self.sp.encode(name, out_type=int)

    def decode(self, tokens):
        return self.sp.decode(tokens, out_type=str)
    
tokenizer = Tokenizer()

# %% [markdown]
# Create Dataset

# %%
batch_size = 128
embedding_size = 512
hidden_size = 64
mask_dimensions = 1000
dropout_rate = 0.1
n_heads = 4
num_blocks = 1

class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    with open('sentences.txt', 'r') as f:
      self.stories = f.readlines()
    self.tokenizer = Tokenizer()

  def __len__(self):
    return len(self.stories)

  def __getitem__(self, idx):
    story = self.stories[idx]
    return torch.tensor(self.tokenizer.encode(story), dtype=torch.long)
  
  def getMaxLen(self):
    return max(len(story) for story in self.stories)


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    max_len = len(batch[0])
    padded_batch = []
    for sequence in batch:
        padded_sequence = torch.cat([sequence, torch.tensor([0] * (max_len - len(sequence)), dtype=torch.long)])
        padded_batch.append(padded_sequence)
    return torch.stack(padded_batch)

ds = Dataset()
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print("Longest item in dataset:", ds.getMaxLen())

# %% [markdown]
# Multi-headed Attention Mimi-Former

# %%

class Head(nn.Module):
    def __init__(self, embedding_size, n_heads, hidden_size, dropout_rate):
        super(Head, self).__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.head_size = embedding_size // n_heads

        # Note: For each head, we have its own set of weights
        self.keys = nn.ModuleList([nn.Linear(self.embedding_size, self.head_size, bias=False) for _ in range(n_heads)])
        self.queries = nn.ModuleList([nn.Linear(self.embedding_size, self.head_size, bias=False) for _ in range(n_heads)])
        self.values = nn.ModuleList([nn.Linear(self.embedding_size, self.head_size, bias=False) for _ in range(n_heads)])
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input): 
        # Split the input across the heads
        outputs = []
        for i in range(self.n_heads):
            q = self.queries[i](input)
            k = self.keys[i](input)
            v = self.values[i](input)

            dim_k = q.shape[-1]

            att = torch.matmul(q, k.transpose(-2, -1))
            att = att / (dim_k ** 0.5)
            att = self.apply_attention_mask(att)
            att = F.softmax(att, dim=-1)
            att = torch.matmul(att, v)
            outputs.append(att)

        # Concatenate the outputs along the last dimension
        return torch.cat(outputs, dim=-1)

    def apply_attention_mask(self, attention_scores):
        # Generate a mask for the lower triangular part of each matrix in the batch
        batch_size = attention_scores.size(0)
        size = attention_scores.size(1)
        mask = torch.tril(torch.ones(batch_size, size, size), diagonal=0).to(attention_scores.device)
    
        # Create a tensor of -inf values with the same shape as attention_scores
        negative_inf = torch.full_like(attention_scores, float('-inf'))
    
        # Use torch.where to fill masked positions with -inf
        masked_attention_scores = torch.where(mask.bool(), attention_scores, negative_inf)
    
        return masked_attention_scores

class Transformer(nn.Module):
    def __init__(self, embedding_size, n_heads, hidden_size, dropout_rate):
        super(Transformer, self).__init__() 

        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.heads = nn.ModuleList([Head(embedding_size, n_heads, hidden_size, dropout_rate) for _ in range(n_heads)])
        
        # Assuming you want to retain the original embedding size after concatenating all head outputs
        self.linear = nn.Linear(embedding_size * n_heads, embedding_size)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )

        self.a_norm = nn.LayerNorm(embedding_size)
        self.ffn_norm = nn.LayerNorm(embedding_size)

        self.final_layer = nn.Linear(embedding_size, vocab_size)

    def get_pos_matrix(self, x):
        # Positional encoding
        batch_size, sequence_length = x.shape
        store = torch.zeros((batch_size, sequence_length, self.embedding_size))
        for pos in range(sequence_length):
            for i in range(0, self.embedding_size, 2):
                denominator = 10000 ** (i / self.embedding_size)
                angles = torch.tensor([pos / denominator]) 
                store[:, pos, i] = torch.sin(angles)
                if i + 1 < self.embedding_size:
                    store[:, pos, i + 1] = torch.cos(angles)
        return store.to(x.device)

    def forward(self, input):
        embedded_input = self.embedding(input)
        pos_encoded_input = embedded_input + self.get_pos_matrix(input)
        pos_encoded_input = self.a_norm(pos_encoded_input)
        heads_outputs = [head(pos_encoded_input) for head in self.heads]
        concatenated_outputs = torch.cat(heads_outputs, dim=-1)
        output = self.linear(concatenated_outputs)
        output = self.ffn_norm(output + pos_encoded_input)
        output = self.ffn(output)
        output = self.final_layer(output)
        return output

# %%
# Model admin

m = Transformer(embedding_size, n_heads, hidden_size, dropout_rate).to(device)

num_params = sum(p.numel() for p in m.parameters())
print(f'The model has {num_params:,} parameters')


# %%

learning_rate = 0.001

optimiser = "Adam"
opt = torch.optim.Adam(m.parameters(), lr = learning_rate)

num_epochs = 1


# title = str(input("Are you ready to run???: "))

title = f"Ep:{num_epochs}, Ba:{batch_size}, Em:{embedding_size}, Hi:{hidden_size}, Dr:{dropout_rate}, He:{n_heads}, Bl:{num_blocks}, T:{device.type}, Ds:{num_lines}, O: {optimiser}, Lr:{learning_rate}"

wandb.init(
  project="Mimi's_Mimiformer",
  name= title,
  config={
  "dataset": "sentences.txt",
  "epochs": num_epochs,
  "batch_size": batch_size,
  "model_params": num_params,
  "optimiser": optimiser, 
  "learning_rate": learning_rate,
  "embedding_size": embedding_size,
  "hidden_size": hidden_size,
  "dropout_rate": dropout_rate,
  "nheads": n_heads,
  "num_blocks": num_blocks,
  "GPU" : device.type, 
  "Dataset": f"TinyStories{num_lines}"
  }
)

# %% [markdown]
# Train model

# %%
# Derive sos and eos from tokenizer
sos = torch.tensor([tokenizer.SOS_TOKEN], dtype=torch.long).to(device)
eos  = torch.tensor([tokenizer.EOS_TOKEN], dtype=torch.long).to(device)

scaler = GradScaler()

start = time.time()

# Training loop
for epoch in range(num_epochs):
  for idx, batch in enumerate(dl):
    # Prepend sos and append eos tokens to sequences
    batch = batch.to(device)
    x = torch.stack([torch.cat([sos, b]) for b in batch]).to(device)
    y = torch.stack([torch.cat([b, eos]) for b in batch]).to(device)

    start_step = time.time()
    # Forward pass
    with autocast():
      p = m(x)
      l = torch.nn.functional.cross_entropy(p.view(-1, p.size(-1)), y.view(-1))
    
    # Backpropagation and optimization
    # Scales the loss, and calls backward() to create scaled gradients
    scaler.scale(l).backward()
    # Unscales gradients and calls or skips optimizer.step()
    scaler.step(opt)
    # Updates the scale for next iteration
    scaler.update()
    opt.zero_grad()

    end_step = time.time()
    step_duration = end_step - start_step

    # Log and print
    wandb.log({"epoch": epoch, "train_loss": l.item(), "step_duration": step_duration})



# Save the model and finish
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"models/{timestamp}_mimiformer.pth"
torch.save(m.state_dict(), save_path)


print(f"Training took {time.time() - start:.2f} seconds")
wandb.log({"Training time": time.time() - start})
wandb.save(save_path)
wandb.finish()

# %%
def sample(probs, temperature=1.0):
    probs = torch.pow(probs, 1/temperature)
    probs = probs / probs.sum()
    return torch.multinomial(probs, 1)

def top_p_sample(probs, top_p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    indices_to_remove = cumulative_probs > top_p
    sorted_probs[indices_to_remove] = 0
    sorted_probs /= sorted_probs.sum()
    return torch.multinomial(sorted_probs, 1)

  # print("Input:", tokenizer.decode(x.tolist()), "Prediction:", tokenizer.decode(p.tolist()) )
  # x = torch.cat([x, p[-1].unsqueeze(0)])

# %%

tokens = []
# generate a random start token for testing
# Get all the tokens/pieces from SentencePiece
all_tokens = [tokenizer.sp.id_to_piece(i) for i in range(tokenizer.vocab_size)]

batched_random = []

for b in range(batch_size): 
  # Pick a random token/piece
  random_token = random.choice(all_tokens)

  # Convert the token to its corresponding ID
  x = torch.tensor([tokenizer.sp.piece_to_id(random_token)]).to(device)
  # print(tokenizer.decode(x.tolist()))

  x = torch.cat([sos, x]) 
  batched_random.append(x)


x = torch.stack(batched_random).to(device)

top_k = 5
p_index = batch_size

while True:
  # run our random start through transformer and get attention matricies out
  p = m(x)
  # create probabilities from 29 token options
  p = torch.nn.functional.softmax(p, dim=-1)
  probs = p[:, -1]

  # mask out all the padding tokens
  probs[:, tokenizer.sp.piece_to_id('<pad>')] = 0

  # choose the best prediction (most probable next token according to tranformer)
  p = sample(probs, 0.9)
  # prediction = torch.argmax(p, dim=-1)[:, -1]
  
  # unsqueeze to get the right dimensions for torch.cat
  x = torch.cat([x, p], dim=1)
  if x.size(1) ==50: break
for i in range(batch_size): 
  generated_text = tokenizer.decode((x[i].tolist()))
  tokens = x[i].tolist()
print(tokenizer.decode(x[0].tolist()))



# %%
