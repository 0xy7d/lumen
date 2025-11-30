import torch
import torch.nn as nn
import torch.optim as optim

from model import LumenLLM, load_vocab, tokenizer

vocab = load_vocab()
vocab_size = len(vocab)
embedding_dim = 16
hidden_dim = 32
num_layers = 2
learning_rate = 0.001

model = LumenLLM(vocab_size, embedding_dim, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data = [
    "hello world",
    "hello world how are you",
    "how are you",
    "i am here",
    "we are here",
    "they are there",
    "i am good",
    "you are good",
    "life is good",
    "people are here",
    "this is new",
    "that is good",
    "i want more",
    "we want more",
    "you know this",
    "i think you know",
    "what do you want",
    "why are you here",
    "where are we",
    "i have life",
]

tokenized_data = [tokenizer(text, vocab) for text in data]

for epoch in range(100):
    for sentence in tokenized_data:
        for i in range(1, len(sentence)):
            input_seq = torch.tensor(sentence[:i]).unsqueeze(0)
            target = torch.tensor(sentence[i]).unsqueeze(0)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output[:, -1, :], target)
            loss.backward()
            optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
