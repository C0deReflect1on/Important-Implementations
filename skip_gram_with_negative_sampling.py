import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from functools import reduce
import string
# !wget 'https://www.dropbox.com/s/obaitrix9jyu84r/quora.txt?dl=1' -O ./quora.txt

vocab = Counter()
vocab['UNK'] = 0
tokens = []
i = 1
with open("./quora.txt", encoding="utf-8") as file:
    for row in file:
        row_tokens = row.lower().translate(str.maketrans('','', string.punctuation)).split()
        vocab.update(row_tokens)
        tokens.extend(row_tokens)
        i += 1

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

token_seq_pairs = []
for i in range(len(tokens)):
    left = max(0, i-WINDOW_SIZE)
    right = min(len(tokens)-1, i+WINDOW_SIZE)
    for j in range(left, right+1):
        if j != i:
            central_idx = word2idx.get(tokens[i], UNK_IDX)
            context_idx = word2idx.get(tokens[j], UNK_IDX)
            token_seq_pairs.append((central_idx, context_idx))

total_count = sum(vocab.values())
word_probs = np.array([vocab[idx2word[i]] for i in range(len(vocab))])
word_probs = (word_probs / total_count) ** 0.75
word_probs /= word_probs.sum()
word_probs_tensor = torch.tensor(word_probs, dtype=torch.float)


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


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, positive_score, negative_score):
        positive_target = torch.ones_like(positive_score)
        negative_target = torch.zeros_like(negative_score)
        return self.bce_loss(positive_score, positive_target) + self.bce_loss(negative_score, negative_target)


class Word2VecDataset(Dataset):
    def __init__(self, token_pairs):
        self.token_pairs = token_pairs

    def __len__(self):
        return len(self.token_pairs)

    def __getitem__(self, idx):
        central_word, context_word = self.token_pairs[idx]
        return central_word, context_word


EMBEDDING_DIM = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 1
K_NEGATIVE_SAMPLES = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Word2VecDataset(token_seq_pairs[:1000])
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

model = SkipGramNegativeSampling(len(word2idx), EMBEDDING_DIM).to(device)
criterion = NegativeSamplingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(EPOCHS):
    total_loss = 0
    print(f"start epoch {epoch+1}/{EPOCHS}")

    for i, (central_words, context_words) in enumerate(dataloader):
        # got list of indices, but assume it is words for simplicity
        central_words: torch.Tensor # for ide support
        context_words: torch.Tensor
        central_words = central_words.to(device)
        context_words = context_words.to(device)

        batch_size = central_words.shape[0]
        negative_samples = torch.multinomial(
            word_probs_tensor,
            K_NEGATIVE_SAMPLES*batch_size,
            replacement=True
        ).reshape(batch_size, K_NEGATIVE_SAMPLES).to(device)
        optimizer.zero_grad()

        positive_score, negative_score = model.forward(central_words, context_words, negative_samples)

        loss = criterion(positive_score, negative_score)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

print("training finished")


def find_similar_words(word, top_n=5):
    model.eval()
    with torch.no_grad():
        word_embed = model.central.weight * word2idx.get(word, word2idx['UNK'])
        all_embeds = model.central.weight.data
        similarities = torch.cosine_similarity(word_embed, all_embeds, dim=1)
        top_n_vals, top_n_indices = torch.topk(similarities, top_n + 1)

        for i in range(1, top_n+1):
            print(f"top {i}: {idx2word[top_n_indices[i].item()]}")

find_similar_words('students')
