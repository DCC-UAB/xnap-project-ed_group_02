import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from GenreFeatureData import GenreFeatureData

# Definició de la classe
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=8, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Configuració de la capa LSTM
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Configuració de la capa de sortida
        self.linear = nn.Linear(self.hidden_dim, output_dim)

        # Capa de normalització recurrent (Recurrent BatchNorm)
        # self.batch_norm = nn.RNNBatchNorm(self.hidden_dim)
        # https://stackoverflow.com/questions/45493384/is-it-normal-to-use-batch-normalization-in-rnn-lstm
        self.batch_norm = nn.SyncBatchNorm(self.hidden_dim)

    def forward(self, input, h, c):
        # Normalització de l'entrada amb Recurrent BatchNorm
        input = self.batch_norm(input)

        # Pas de l'LSTM
        lstm_out, (h,c) = self.lstm(input, (h,c))
        logits = self.linear(lstm_out[:, -1, :])
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores, h, c

    def get_accuracy(self, logits, target):
        """Calcula l'exactitud per a una ronda d'entrenament"""
        corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
    
    def init_hidden(self, batch_size):
        return nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_dim)), nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
     

def main():
    genre_features = GenreFeatureData()

    # Si tots els fitxers pre-processats no existeixen, es generen tots per a una coherència interna
    if (
        os.path.isfile(genre_features.train_X_preprocessed_data)
        and os.path.isfile(genre_features.train_Y_preprocessed_data)
        and os.path.isfile(genre_features.dev_X_preprocessed_data)
        and os.path.isfile(genre_features.dev_Y_preprocessed_data)
        and os.path.isfile(genre_features.test_X_preprocessed_data)
        and os.path.isfile(genre_features.test_Y_preprocessed_data)
    ):
        print("Els fitxers pre-processats existeixen, deserialitzant fitxers npy")
        genre_features.load_deserialize_data()
    else:
        print("Pre-processant fitxers d'àudio raw")
        genre_features.load_preprocess_data()

    train_X = torch.from_numpy(genre_features.train_X).type(torch.Tensor)
    dev_X = torch.from_numpy(genre_features.dev_X).type(torch.Tensor)
    test_X = torch.from_numpy(genre_features.test_X).type(torch.Tensor)

    train_Y = torch.from_numpy(genre_features.train_Y).type(torch.LongTensor)
    dev_Y = torch.from_numpy(genre_features.dev_Y).type(torch.LongTensor)
    test_Y = torch.from_numpy(genre_features.test_Y).type(torch.LongTensor)

    print("Mida de l'entrada d'entrenament:", genre_features.train_X.shape)
    print("Mida de la sortida d'entrenament:", genre_features.train_Y.shape)
    print("Mida de l'entrada de validació:", genre_features.dev_X.shape)
    print("Mida de la sortida de validació:", genre_features.dev_Y.shape)
    print("Mida de l'entrada de prova:", genre_features.test_X.shape)
    print("Mida de la sortida de prova:", genre_features.test_Y.shape)

    batch_size = 35
    num_epochs = 400

    # Definició del model
    print("Creant el model LSTM RNN...")
    model = LSTM( input_dim=33, hidden_dim=128, batch_size=batch_size, output_dim=8, num_layers=2)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Entrenant en:",device)

    train_losses = []
    train_accuracies = []
    dev_losses = []
    dev_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        state, c = model.init_hidden(batch_size) # Start with a new state in each batch 
        state = state.to(device) 
        c = c.to(device)

        train_loss = 0.0
        train_acc = 0.0

        for i in range(0, train_X.size(0), batch_size):
            if i + batch_size > train_X.size(0):
                break

            # Extraindo dades del batch
            batch_X = train_X[i : i + batch_size, :, :]
            batch_Y = train_Y[i : i + batch_size]

            # Esborra els gradients
            model.zero_grad()

            # NLLLoss does not expect a one-hot encoded vector as the target, but class indices
            batch_y_local = torch.max(batch_Y, 1)[1]
            batch_y_local = batch_y_local.to(device)
            batch_X = batch_X.to(device)

            # Pas endavant
            output, state,c = model(batch_X, state,c)
            state.detach_()
            c.detach_() 

            # Càlcul de la pèrdua
            # https://stackoverflow.com/questions/66635987/how-to-solve-this-pytorch-runtimeerror-1d-target-tensor-expected-multi-target
            # print( "Array Output:", output)
            # print( "Array Batch_y:", batch_Y)
            
            loss = loss_function(output, batch_y_local)
            train_loss += loss.item()

            # Pas endarrere
            loss.backward()

            # Actualització dels pesos
            optimizer.step()

            # Càlcul de l'exactitud
            accuracy = model.get_accuracy(output, batch_y_local)
            train_acc += accuracy

        # Decaiment de la taxa d'aprenentatge
        optimizer.param_groups[0]['lr'] *= 0.9

        # Càlcul de les pèrdues i les exactituds mitjanes
        train_loss /= len(train_X) / batch_size
        train_acc /= len(train_X) / batch_size

        # Avaluació en el conjunt de validació
        model.eval()
        with torch.no_grad():
            dev_loss = 0.0
            dev_acc = 0.0
            

            for i in range(0, dev_X.size(0), batch_size):
                if i + batch_size > dev_X.size(0):
                    break

                batch_X = dev_X[i : i + batch_size, :, :]
                batch_Y = dev_Y[i : i + batch_size]

                batch_y_local = torch.max(batch_Y, 1)[1]
                batch_y_local = batch_y_local.to(device)
                batch_X = batch_X.to(device)

                output, _, _ = model(batch_X, state, c)
                loss = loss_function(output, batch_y_local)
                dev_loss += loss.item()

                accuracy = model.get_accuracy(output, batch_y_local)
                dev_acc += accuracy

            # Càlcul de les pèrdues i les exactituds mitjanes
            dev_loss /= len(dev_X) / batch_size
            dev_acc /= len(dev_X) / batch_size

        # Imprimir el progrés
        print(
            f"Època: {epoch+1}/{num_epochs}, Pèrdua d'entrenament: {train_loss:.4f}, Exactitud d'entrenament: {train_acc:.2f}%, "
            f"Pèrdua de validació: {dev_loss:.4f}, Exactitud de validació: {dev_acc:.2f}%"
        )

        # Desar les pèrdues i les exactituds per a la representació gràfica
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        dev_losses.append(dev_loss)
        dev_accuracies.append(dev_acc)

    # Representació gràfica de les pèrdues d'entrenament i validació
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Pèrdua d\'entrenament')
    plt.plot(range(1, num_epochs + 1), dev_losses, label='Pèrdua de validació')
    plt.xlabel('Època')
    plt.ylabel('Pèrdua')
    plt.legend()
    plt.show()

    # Representació gràfica de les exactituds d'entrenament i validació
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Exactitud d\'entrenament')
    plt.plot(range(1, num_epochs + 1), dev_accuracies, label='Exactitud de validació')
    plt.xlabel('Època')
    plt.ylabel('Exactitud (%)')
    plt.legend()
    plt.show()

    # Avaluació en el conjunt de prova
    model.eval()
    with torch.no_grad():
        test_acc = 0.0
        for i in range(0, test_X.size(0), batch_size):
            if i + batch_size > test_X.size(0):
                break

            batch_X = test_X[i : i + batch_size, :, :]
            batch_Y = test_Y[i : i + batch_size]

            #X_local_minibatch = batch_X.permute(1, 0, 2)
            y_local_minibatch = torch.max(batch_Y, 1)[1]
            #X_local_minibatch = X_local_minibatch.to(device)
            batch_X = batch_X.to(device)
            y_local_minibatch = y_local_minibatch.to(device)
            
            output, _, _ = model(batch_X,state,c)
            accuracy = model.get_accuracy(output, y_local_minibatch)
            test_acc += accuracy

        # Càlcul de l'exactitud mitjana
        test_acc /= len(test_X) / batch_size

        print(f"Exactitud de prova: {test_acc:.2f}%")

if __name__ == "__main__":
    main()