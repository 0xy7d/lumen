import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import LumenLLM, tokenizer
from config import (
    vocab,
    vocab_size,
    embedding_dim,
    hidden_dim,
    num_layers,
    learning_rate,
    epoches,
)

model = LumenLLM(vocab_size, embedding_dim, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data = [
    "hello world",
    "hello how are you",
    "i am here with you",
    "we are in the place",
    "they are in the world",
    "this is a good day",
    "that is a new thing",
    "life is good",
    "people are here",
    "you are in my life",
    "i want more time",
    "we want a good day",
    "you know this place",
    "i think you know",
    "what do you want",
    "why are you here",
    "where are we",
    "when will you come",
    "i will go to the place",
    "you can make a new thing",
    "we can do this",
    "they can see the world",
    "i have a name",
    "your name is good",
    "our life is new",
    "i am not bad",
    "this is very good",
    "that is very bad",
    "you look good",
    "i feel good today",
    "i feel good",
    "you say the same thing",
    "i know the man",
    "you know the woman",
    "the man is here",
    "the woman is there",
    "this thing is new",
    "that thing is good",
    "i look at the world",
    "we look at life",
    "you feel the time",
    "i want to see you",
    "you want to know more",
    "we have the time",
    "they have a place",
    "i think of you",
    "we think of life",
    "you make the day good",
    "this is the world we know",
]

tokenized_data = [tokenizer(text, vocab) for text in data]

if "lumen-v0.pt" in os.listdir("./models"):
    print("Model already exists")
else:
    for epoch in range(epoches):
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

    # Save the model
    torch.save(model.state_dict(), "models/lumen-v0.pt")
