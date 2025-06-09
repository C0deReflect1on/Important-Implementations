import re
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


corpus = """
    Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
    Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
    when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
    It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. 
    It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and 
    more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
"""

class CorpusData:
    def __init__(self, corpus: str):
        corpus = re.sub(r'[^a-z@# ]', '', corpus.lower())
        self.token_seq = corpus.split()
        vocab = set(self.token_seq)
        self.word2idx = {word: i for (i, word) in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def get_skip_gram_data(self, window_size=5):
        data = []
        for i, central_word in enumerate(self.token_seq):
            left = max(0, i - window_size)
            right = min(i + window_size + 1, len(self.token_seq))
            for j in range(left, right):  # context
                if j != i:
                    data.append((central_word, self.token_seq[j]))
        return data


class SkipGramDataset(Dataset):
    def __init__(self, context_seq: list[tuple[str, str]], word2idx: dict):
        self.idx_data = [(word2idx[center],word2idx[context]) for center, context in context_seq]

    def __len__(self):
        return self.idx_data.__len__()

    def __getitem__(self, idx): # return tuple of central and context word
        return (
            torch.tensor(self.idx_data[idx][0], dtype=torch.long),
            torch.tensor(self.idx_data[idx][1], dtype=torch.long)
        )


class Word2VecSkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        self.activation_function = nn.LogSoftmax(dim=-1)

    def forward(self, center_word_idx):
        hidden_layer = self.embeddings(center_word_idx)
        out_layer = self.out_layer(hidden_layer)
        log_probs = self.activation_function(out_layer)
        return log_probs


corpus_data = CorpusData(corpus)
dataset = SkipGramDataset(context_seq=corpus_data.get_skip_gram_data(), word2idx=corpus_data.word2idx)
model = Word2VecSkipGramModel(corpus_data.vocab_size, embedding_dim=16)
epochs = 10

dataloader = DataLoader(dataset, shuffle=True)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(epochs):
    total_loss = 0
    for center_word, context_word in dataloader:
        model.zero_grad()
        log_probs = model(center_word)
        loss = loss_fn(log_probs, context_word)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch}: Loss: {total_loss}")

embeddings = model.embeddings.weight.data.numpy()

idx2word = {idx: word for (word, idx) in corpus_data.word2idx.items()}
word2embed = {idx2word[idx]: embeddings[idx] for idx in range(len(embeddings))}
