import numpy as np
import torch
from torch import nn
from collections import Counter
from functools import reduce
import string
# !wget 'https://www.dropbox.com/s/obaitrix9jyu84r/quora.txt?dl=1' -O ./quora.txt

vocab = Counter()
vocab['UNK'] = 0
tokens = []
i = 0
with open("./quora.txt", encoding="utf-8") as file:
    for row in file:
        row_tokens = row.lower().translate(str.maketrans('','', string.punctuation)).split()
        vocab.update(row_tokens)
        tokens.extend(row_tokens)
        i += 1
        if i == 100:
            break
worse_words = []
for v, cnt in vocab.items():
    if cnt < 5:
        worse_words.append(v)

for word in worse_words:
    vocab['UNK'] += vocab[word]
    del vocab[word]

word2idx = {}
idx2word = {}

for (i, v) in enumerate(vocab.keys()):
    word2idx[v] = i
    idx2word[i] = v

UNK_IDX = word2idx['UNK']
WINDOW_SIZE = 5

token_seq = []
for i in range(len(tokens)):
    left = max(0, i-WINDOW_SIZE)
    right = min(len(tokens)-1, i+WINDOW_SIZE)
    for j in range(left, right+1):
        if j != i:
            token_seq.append((word2idx.get(tokens[i], UNK_IDX), word2idx.get(tokens[j], UNK_IDX)))

total_count = sum(vocab.values())
word_probs = np.array([vocab[idx2word[i]] for i in range(len(vocab))])
word_probs = (word_probs / total_count) ** 0.75
word_probs /= word_probs.sum()


class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.central = nn.Embedding(vocab_size, embedding_dim)
        self.context = nn.Embedding(vocab_size, embedding_dim)

        self.init_embeds()

    def init_embeds(self):
        nn.init.uniform_(self.central.weight, -1,1)
        nn.init.uniform_(self.context.weight, -1, 1)


    def forward(self, central_idxs, context_idxs , negative_samples_idxs):
        central_embed = self.central(central_idxs) # (batch_size, embed_dim)
        context_embed = self.context(context_idxs) # (batch_size, embed_dim)
        context_negative_embed = self.context(negative_samples_idxs) # (batch_size, k , embed_dim)

        positive_score = torch.sum(central_embed * context_embed, dim=1, keepdim=True)
        negative_score = torch.bmm(central_embed.unsqueeze(1), context_negative_embed.transpose(2, 1)).squeeze(1)
        return positive_score, negative_score

model = SkipGramNegativeSampling(len(vocab), 300)
BATCH_SIZE = 4
sample_pairs = token_seq[:BATCH_SIZE]
central = torch.tensor([pair[0] for pair in sample_pairs])
context = torch.tensor([pair[1] for pair in sample_pairs])
negative = torch.randint(0, len(vocab), (BATCH_SIZE, 5))

result = model(central, context, negative)
print(*result, sep='\n')
