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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

from GenreFeatureDataFMA import (
    GenreFeatureData,
)  # local python class with Audio feature extraction (librosa)
from entrenament_fma import entrenar

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
        self.dropout = nn.Dropout(p=0)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input, state, c):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        lstm_out, (state,c) = self.lstm(input, (state,c))
        x = self.dropout(self.dropout(lstm_out[-1]))
        logits = self.linear(lstm_out[-1])              # equivalent to return_sequences=False from Keras
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores, state,c

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
    
    def init_hidden(self, batch_size):
        return nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_dim)), nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        

def main():
    genre_features = GenreFeatureData()
    #genre_features.augmentar=True
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

    # Targets is a long tensor of size (N,) which tells the true class of the sample.
    train_Y = torch.from_numpy(genre_features.train_Y).type(torch.LongTensor)
    dev_Y = torch.from_numpy(genre_features.dev_Y).type(torch.LongTensor)
    test_Y = torch.from_numpy(genre_features.test_Y).type(torch.LongTensor)

    # Convert {training, test} torch.Tensors
    print("Training X shape: " + str(genre_features.train_X.shape))
    print("Training Y shape: " + str(genre_features.train_Y.shape))
    print("Validation X shape: " + str(genre_features.dev_X.shape))
    print("Validation Y shape: " + str(genre_features.dev_Y.shape))
    print("Test X shape: " + str(genre_features.test_X.shape))
    print("Test Y shape: " + str(genre_features.test_Y.shape))

    batch_size = 64  # num of training examples per minibatch
    num_epochs = 1001

    # Define model
    print("Build LSTM RNN model ...")
    model = LSTM(
        input_dim=33, hidden_dim=64, batch_size=batch_size, output_dim=8, num_layers=2
    )
    loss_function = nn.NLLLoss()  # expects ouputs from LogSoftmax
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # To keep LSTM stateful between batches, you can set stateful = True, which is not suggested for training
    stateful = False

    # To keep LSTM stateful between batches, you can set stateful = True, which is not suggested for training
    # stateful = False
    #fem axo per provar
    #train_X=torch.cat((train_X,dev_X),0)
    #train_Y=torch.cat((train_Y,dev_Y),0)

    conjunt_entrenament=(train_X,train_Y,dev_X,dev_Y,test_X,test_Y)
    entrenar(model,conjunt_entrenament,optimizer,batch_size=64,num_epochs=1001,scheduler=scheduler)
    

if __name__ == "__main__":
    main()
