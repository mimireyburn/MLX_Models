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

# Ensure Deterministic Behavior
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load Dataset
dataset = load_dataset("roneneldan/TinyStories")

# %%
sample_data = dataset["train"]['text'][:2000]

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
vocab_size = 1000

# 

spm.SentencePieceTrainer.train(
    input=input_file, 
    model_prefix=prefix, 
    vocab_size=vocab_size,
    user_defined_symbols="<pad>,<sos>,<eos>"
)

# %%
class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{prefix}.model')
        self.vocab_size = self.sp.get_piece_size()

        self.PAD_TOKEN = self.sp.piece_to_id('<pad>')
        self.SOS_TOKEN = self.sp.piece_to_id('<sos>')
        self.EOS_TOKEN = self.sp.piece_to_id('<eos>')

    def encode(self, name):
        return self.sp.encode_as_ids(name)

    def decode(self, tokens):
        return self.sp.decode_ids(tokens)
    
tokenizer = Tokenizer()

# %%
def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    max_len = len(batch[0])
    padded_batch = []
    for sequence in batch:
        padded_sequence = torch.cat([sequence, torch.tensor([tokenizer.PAD_TOKEN] * (max_len - len(sequence)), dtype=torch.long)])
        padded_batch.append(padded_sequence)
    return torch.stack(padded_batch)

# %%
batch_size = 10

class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    with open('sentences.txt', 'r') as f:
      self.names = f.read().split('\n')
    self.tokenizer = Tokenizer()

  def __len__(self):
    return len(self.names)

  def __getitem__(self, idx):
    name = self.names[idx]
    return torch.tensor(self.tokenizer.encode(name), dtype=torch.long)


ds = Dataset()
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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

class DecoderBlock(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_rate):
        super(DecoderBlock, self).__init__()
        
        # Self-Attention Mechanism
        self.key = torch.nn.Linear(embedding_size, hidden_size)
        self.qry = torch.nn.Linear(embedding_size, hidden_size)
        self.val = torch.nn.Linear(embedding_size, hidden_size)
        
        # Feed-forward Network
        self.ffw = torch.nn.Linear(hidden_size, embedding_size)
        
        # Layer Norm & Dropout
        self.layer_norm = torch.nn.LayerNorm(embedding_size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        # Self-Attention
        key = self.key(x)
        qry = self.qry(x)
        val = self.val(x)
        
        att = torch.matmul(qry, key.transpose(-2, -1)) / math.sqrt(key.shape[1])
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=1)
        
        res = torch.matmul(att, val)
        res = self.dropout(self.ffw(res))
        
        # Add & Norm
        x = self.layer_norm(x + res)
        
        return x, att
    

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, mask_dimensions, dropout_rate, batch_size):
        super(Transformer, self).__init__()

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.decoder1 = DecoderBlock(embedding_size, hidden_size, dropout_rate)
        self.decoder2 = DecoderBlock(embedding_size, hidden_size, dropout_rate)
        self.final_layer_norm = torch.nn.LayerNorm(embedding_size)
        self.map_to_vocab = torch.nn.Linear(embedding_size, vocab_size)
        
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
        return store


    def forward(self, x):
        sequence_length = x.shape[1]
        mask = torch.tril(torch.ones(sequence_length, sequence_length)).to(x.device)
        x = self.embedding(x) + self.get_pos_matrix(x)
        x, att_00 = self.decoder1(x, mask)
        x, att_01 = self.decoder2(x, mask)
        
        x = self.final_layer_norm(x)
        out = self.map_to_vocab(x)
        
        return out, [att_00, att_01]
        

m = Transformer(vocab_size, embedding_size, hidden_size, mask_dimensions, dropout_rate, batch_size)

# SDG instead of Adam, why?
opt = torch.optim.SGD(m.parameters(), lr=0.01)

print("Parameters: ", m.parameters())

# %%
loss_history = []
num_epochs = 10
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"models/{timestamp}_mimiformer.pth"


# Derive sos and eos from tokenizer
sos = torch.tensor([tokenizer.SOS_TOKEN], dtype=torch.long)
eos  = torch.tensor([tokenizer.EOS_TOKEN], dtype=torch.long)

start = time.time()

for epoch in range(num_epochs):
  for idx, batch in enumerate(dl):

    # Add sos to the beginning of each sequence in the batch
    x = [torch.cat([sos, b]) for b in batch]
    y = [torch.cat([b, eos]) for b in batch]
    # convert batch to tensor
    x = torch.stack(x)
    y = torch.stack(y)

    # run our batch through the whole transformer (attention1, ffw, attention2, ffw, linear)
    p, _ = m(x)
    # calculate cross-entropy loss between predicted token and target token
    l = torch.nn.functional.cross_entropy(p.view(-1, p.size(-1)), y.view(-1))
    # print loss every 1000 rows of dataset
    if idx % 100 == 0: 
      print("Loss:", l.item())
      loss_history.append(l.item())
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

batched_random = []

for b in range(batch_size): 
  # Pick a random token/piece
  random_token = random.choice(all_tokens)

  # Convert the token to its corresponding ID
  x = torch.tensor([tokenizer.sp.piece_to_id(random_token)])
  print(tokenizer.decode(x.tolist()))

  x = torch.cat([sos, x]) 
  batched_random.append(x)


x = torch.stack(batched_random)

top_k = 5
p_index = batch_size

while True:
  tokens.append(x[:, -1])

  # run our random start through transformer and get attention matricies out
  p, attention = m(x)
  # create probabilities from 29 token options
  p = torch.nn.functional.softmax(p, dim=batch_size)

  # choose the best prediction (most probable next token according to tranformer)
  prediction = torch.argmax(p, dim=batch_size)[:, -1]
  
  # unsqueeze to get the right dimensions for torch.cat
  x = torch.cat([x, prediction.unsqueeze(1)], dim=1)
  if x.size(1) ==20: break
for i in range(batch_size): 
  print(f"Generated text for batch {i+1}:", tokenizer.decode(x[i].tolist()))

# %%
def plot_attention_heatmap(attention, tokens):
    # Convert tensor tokens to list of integers and then decode
    decoded_tokens = []
    for tok in tokens:
        # If token tensor has more than one value, we need to handle it differently
        if tok.numel() > 1:
            decoded_tokens.append([tokenizer.decode(t.tolist()) for t in tok])
        else:
            decoded_tokens.append(tokenizer.decode([tok.item()]))
    
    num_layers = len(attention)
    
    # Adjust the subplots layout depending on the number of batches and layers
    fig, axes = plt.subplots(batch_size, num_layers, figsize=(4*num_layers, 4*batch_size))
    
    # If there's only one batch and one layer, axes won't be 2D array, so we need to wrap it in a list
    if batch_size == 1 and num_layers == 1:
        axes = [[axes]]
    elif batch_size == 1 or num_layers == 1:
        axes = [axes]
    
    for batch_idx in range(batch_size):
        for layer_idx in range(num_layers):
            ax = axes[batch_idx][layer_idx]
            sns.heatmap(attention[layer_idx][batch_idx].detach().numpy(), annot=False, cmap='viridis', ax=ax, xticklabels=decoded_tokens[batch_idx], yticklabels=decoded_tokens[batch_idx])
            ax.set_title(f'Batch {batch_idx+1}, Attention Layer {layer_idx + 1}')
            ax.set_xticks(np.arange(len(decoded_tokens[batch_idx])) + .5, minor=False)
            ax.set_yticks(np.arange(len(decoded_tokens[batch_idx])) + .5, minor=False)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.show()

# Assuming that you've tokenized 'x' and the tokens are stored in a variable called 'tokens'
plot_attention_heatmap(attention, tokens)



