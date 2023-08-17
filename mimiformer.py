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

# %% [markdown]
# Tokenize

# %%
input_file = 'sentences.txt' 
prefix = 'sentences'
vocab_size = 1000

spm.SentencePieceTrainer.train(
    input=input_file, 
    model_prefix=prefix, 
    vocab_size=vocab_size, 
    model_type='bpe' # 'unigram' or 'bpe'
)

class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{prefix}.model')
        self.vocab_size = self.sp.get_piece_size()

        self.PAD_TOKEN = self.sp.piece_to_id('<pad>')
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
batch_size = 32

class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    with open('sentences.txt', 'r') as f:
      self.stories = f.read().replace('\n', '').split("</s>")
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
        padded_sequence = torch.cat([sequence, torch.tensor([tokenizer.PAD_TOKEN] * (max_len - len(sequence)), dtype=torch.long)])
        padded_batch.append(padded_sequence)

    return torch.stack(padded_batch)


ds = Dataset()
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print("Longest item in dataset:", ds.getMaxLen())

# %% [markdown]
# Multi-headed Attention Mimi-Former

# %%
embedding_size = 64
hidden_size = 10
mask_dimensions = 1000
dropout_rate = 0.1
nheads = 2

# Encase the above functions in a modular class
class MHA_Transformer(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, tokenizer, dropout_rate, nheads):
        super(MHA_Transformer, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.dropout_rate = dropout_rate
        self.nheads = nheads
        self.head_size = embedding_size // nheads
        self.dim_k = self.head_size

        self.embedding = torch.nn.Embedding(tokenizer.vocab_size, embedding_size)
        self.query = torch.nn.Linear(embedding_size, nheads * self.dim_k, bias = False)
        self.key = torch.nn.Linear(embedding_size, nheads * self.dim_k, bias = False)
        self.value = torch.nn.Linear(embedding_size, nheads * self.dim_k, bias = False)
        self.linear = torch.nn.Linear(embedding_size, embedding_size)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, embedding_size)
        )
        self.mha_norm = torch.nn.LayerNorm(embedding_size)
        self.ffn_norm = torch.nn.LayerNorm(embedding_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.model_output = torch.nn.Linear(embedding_size, tokenizer.vocab_size)
        self.softmax = torch.nn.Softmax(dim = -1)

    def create_embedding(self, x):
        emb = self.embedding(x)
        pos = self.positional_encoding(x)
        emb = emb + pos
        return emb
        
    def forward(self, x):
        self.sequence_length = x.shape[1]
        self.mask = torch.tril(torch.ones(self.sequence_length, self.sequence_length))
        emb = x
        query = self.query(emb)
        key = self.key(emb)
        value = self.value(emb)
        
        query = self.split_heads(query, x.shape[0])
        key = self.split_heads(key, x.shape[0])
        value = self.split_heads(value, x.shape[0])

        scores = torch.matmul(query, key.transpose(-1, -2))
        scaled_scores = scores / torch.sqrt(torch.tensor(self.sequence_length))
        masked_scores = scaled_scores.masked_fill(self.mask == 0, float('-inf'))
        attention_weights = torch.nn.functional.softmax(masked_scores, dim = -1)

        output = torch.matmul(attention_weights, value)
        concat = output.permute(0, 2, 1, 3).contiguous()
        concat = concat.view(x.shape[0], -1, self.embedding_size)
        
        mha_output = self.linear(concat)
        output = self.mha_norm(mha_output + emb)
        ffn_output = self.ffn(output)
        output = self.ffn_norm(ffn_output + output)
        output = self.dropout(output)
        return output

    def last_ffw_layer(self, output):
        output = self.model_output(output)
        probs = self.softmax(output)
        return probs 
    
    def positional_encoding(self, x):
        self.sequence_length = x.shape[1]
        encoding = torch.zeros(self.sequence_length, self.embedding_size)
        for pos in range(self.sequence_length):
            for i in range(0, self.embedding_size, 2):
                encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embedding_size)))
                encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embedding_size)))
        return encoding.unsqueeze(0)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.nheads, self.head_size)
        return x.permute(0, 2, 1, 3)
    
    def generate(self, x, length):
        with torch.no_grad():
            for i in range(length):
                probs = self.forward(x)
                predicted = torch.argmax(probs, dim = -1)
                x = torch.cat((x, predicted[-1].unsqueeze(0)), dim = 0)
        return x
    
    def run_six_blocks(self, x):
        x = self.create_embedding(x)
        for i in range(6):
            x = self.forward(x)
        last_layer = self.last_ffw_layer(x)
        return last_layer
    

# Train the model
m = MHA_Transformer(embedding_size, hidden_size, tokenizer, dropout_rate, nheads)
# m = m.run_six_blocks()
opt = torch.optim.Adam(m.parameters(), lr = 0.01)

num_params = sum(p.numel() for p in m.parameters())
print(f'The model has {num_params:,} parameters')


# %% [markdown]
# Train

# %%
loss_history = []
num_epochs = 10
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"models/{timestamp}_mimiformer.pth"


# Derive sos and eos from tokenizer
sos = torch.tensor([tokenizer.SOS_TOKEN], dtype=torch.long)
eos  = torch.tensor([tokenizer.EOS_TOKEN], dtype=torch.long)

start = time.time()
it = 100

# Initialise wandb
wandb.init(
  project="Mimi's_Mimiformer",
  name= "Two_headed_Mimiformer",
  config={
  "dataset": "sentences.txt",
  "epochs": num_epochs,
  "batch_size": batch_size,
  "model_params": num_params,
  }
)


for epoch in range(num_epochs):
  for idx, batch in enumerate(dl):

    # Add sos to the beginning of each sequence in the batch
    x = [torch.cat([sos, b]) for b in batch]
    y = [torch.cat([b, eos]) for b in batch]
    # convert batch to tensor
    x = torch.stack(x)
    y = torch.stack(y)

    # run our batch through the whole transformer (attention1, ffw, attention2, ffw, linear)
    p = m.run_six_blocks(x)
    # calculate cross-entropy loss between predicted token and target token
    l = torch.nn.functional.cross_entropy(p.view(-1, p.size(-1)), y.view(-1))
    # print loss every 100 rows of dataset

    wandb.log({"loss": l.item()})
    if idx % 100 == 0: 
      print(f"Loss of {it}:", l.item())
      it += 100
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
plt.title('Loss every 100 Datapoints')
plt.grid(True)
plt.show()

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

  # p = sample(probs, 0.9)

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
  x = torch.tensor([tokenizer.sp.piece_to_id(random_token)])
  # print(tokenizer.decode(x.tolist()))

  x = torch.cat([sos, x]) 
  batched_random.append(x)


x = torch.stack(batched_random)

top_k = 5
p_index = batch_size

while True:
  # run our random start through transformer and get attention matricies out
  p, attention = m(x)
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
  if x.size(1) ==20: break
for i in range(batch_size): 
  generated_text = tokenizer.decode((x[i].tolist()))
  tokens = x[i].tolist()
  print(tokens)
  print(f"Generated text for batch {i+1}:", tokenizer.decode(x[i].tolist()))

# %%
def plot_attention_heatmap(attention, tokens):
    # Convert tensor tokens to list of integers and then decode
    decoded_tokens = []
    for tok in tokens:
        decoded_tokens.append([tokenizer.decode(t.tolist()) for t in tok])
    
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
            # ax.set_xticks(np.arange(len(decoded_tokens[batch_idx])) + .5, minor=False)
            # ax.set_yticks(np.arange(len(decoded_tokens[batch_idx])) + .5, minor=False)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.show()

# Assuming that you've tokenized 'x' and the tokens are stored in a variable called 'tokens'
plot_attention_heatmap(attention, tokens)


