#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    PyTorch implementation of a simple 2-layer-deep LSTM for genre classification of musical audio.
    Feeding the LSTM stack are spectral {centroid, contrast}, chromagram & MFCC features (33 total values)

    Question: Why is there a PyTorch implementation, when we already have Keras/Tensorflow?
    Answer:   So that we can learn more PyTorch and experiment with modulations on basic
              architectures within the space of an "easy problem". For example, SRU or SincNets.
              I'm am also curious about the relative performances of both toolkits.

"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from GenreFeatureData_m import GenreFeatureData

# class definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=8, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input, state, c):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        lstm_out, (state, c) = self.lstm(input, (state, c))
        logits = self.linear(lstm_out[-1])  # equivalent to return_sequences=False from Keras
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores, state, c

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()

    def init_hidden(self, batch_size):
        return (
            nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
            nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
        )


def main():
    genre_features = GenreFeatureData()
    genre_features.augmentar=True
    # if all of the preprocessed files do not exist, regenerate them all for self-consistency
    if (
            os.path.isfile(genre_features.train_X_preprocessed_data)
            and os.path.isfile(genre_features.train_Y_preprocessed_data)
            and os.path.isfile(genre_features.dev_X_preprocessed_data)
            and os.path.isfile(genre_features.dev_Y_preprocessed_data)
            and os.path.isfile(genre_features.test_X_preprocessed_data)
            and os.path.isfile(genre_features.test_Y_preprocessed_data)
    ):
        print("Preprocessed files exist, deserializing npy files")
        genre_features.load_deserialize_data()
    else:
        print("Preprocessing raw audio files")
        genre_features.load_preprocess_data()

    train_X = torch.from_numpy(genre_features.train_X).type(torch.Tensor)
    dev_X = torch.from_numpy(genre_features.dev_X).type(torch.Tensor)
    test_X = torch.from_numpy(genre_features.test_X).type(torch.Tensor)
    train_Y = torch.from_numpy(genre_features.train_Y).type(torch.LongTensor)
    dev_Y = torch.from_numpy(genre_features.dev_Y).type(torch.LongTensor)
    test_Y = torch.from_numpy(genre_features.test_Y).type(torch.LongTensor)

    input_dim = genre_features.num_features  # should equal 33 (3 per spectral band * 11 bands)
    hidden_dim = 128
    batch_size = 30
    output_dim = 8
    num_epochs = 100

    train_data = torch.utils.data.TensorDataset(train_X, train_Y)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )

    lstm = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, batch_size=batch_size, output_dim=output_dim)

    optimizer = optim.Adam(lstm.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_fn = nn.CrossEntropyLoss()

    train_loss = []
    dev_accuracy = []

    for epoch in range(num_epochs):
        state, c = lstm.init_hidden(batch_size)
        lstm.train()
        epoch_loss = 0
        accuracy = 0
        num_batches = 0

        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            state.detach_()
            c.detach_()

            output, state, c = lstm(features, state, c)

            loss = loss_fn(output, labels)
            epoch_loss += loss.item()

            accuracy += lstm.get_accuracy(output, labels)

            loss.backward()
            optimizer.step()

            num_batches += 1

        scheduler.step()

        epoch_loss /= num_batches
        accuracy /= num_batches

        train_loss.append(epoch_loss)
        dev_accuracy.append(evaluate_model(lstm, dev_X, dev_Y, batch_size))

        print(
            f"Epoch: {epoch + 1}/{num_epochs}, "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Accuracy: {accuracy:.2f}%, "
            f"Dev Accuracy: {dev_accuracy[-1]:.2f}%"
        )

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(dev_accuracy, label="Dev Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.show()


def evaluate_model(model, X, Y, batch_size):
    model.eval()
    num_batches = X.shape[0] // batch_size
    accuracy = 0

    with torch.no_grad():
        state, c = model.init_hidden(batch_size)

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size

            features = X[start:end]
            labels = Y[start:end]

            output, state, c = model(features, state, c)

            accuracy += model.get_accuracy(output, labels)

    accuracy /= num_batches
    return accuracy


if __name__ == "__main__":
    main()
