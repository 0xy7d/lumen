"""
Inference file to test out the trained model.
"""

import torch

from model import LumenLLM, tokenizer
from train import embedding_dim, hidden_dim, num_layers, vocab, vocab_size


def load_model():
    model = LumenLLM(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load("./models/lumen-v0.pt"))
    model.eval()
    return model


def predict(model, text):
    tokens = tokenizer(text, vocab)
    tokens = torch.tensor(tokens).unsqueeze(0)
    output = model(tokens)

    predicted_token = output.argmax().item()

    try:
        predicted_word = list(vocab.keys())[list(vocab.values())[predicted_token]]
    except:
        predicted_word = "<UNK>"

    return predicted_word


if __name__ == "__main__":
    model = load_model()

    while True:
        text = input("Enter text: ")
        if text == "exit":
            break
        output = predict(model, text)
        print(output)
