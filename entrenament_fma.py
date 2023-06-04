
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns

def entrenar(model,conjunt_entrenament,optimizer,scheduler=None,loss_function=nn.NLLLoss(),num_epochs=400,batch_size=42,grafiquejar=True):
    train_X,train_Y,dev_X,dev_Y,test_X,test_Y=conjunt_entrenament
    
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print("\nTraining on GPU")
    else:
        print("\nNo GPU, training on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # all training data (epoch) / batch_size == num_batches (12)
    #batch size ha de dividir 420 de forma entera 
    #1,2,3,4,5,6,7,10,12,14,15,20,21,28,30,35,42,60,70,84,105,140,210,420
    num_batches = int(train_X.shape[0] / batch_size)
    num_dev_batches = int(dev_X.shape[0] / batch_size)

    val_loss_list, val_accuracy_list, epoch_list = [], [], []
    train_accuracy_list,train_loss_list,epoch_train_list=[],[],[]

    print("Training ...")
    for epoch in range(num_epochs):

        train_running_loss, train_acc = 0.0, 0.0

        # Init hidden state - if you don't want a stateful LSTM (between epochs)
        state, c = model.init_hidden(batch_size) # Start with a new state in each batch 
        state = state.to(device) 
        c = c.to(device)
        
        
        for i in range(num_batches):
            
            # zero out gradient, so they don't accumulate btw batches
            model.zero_grad()

            # train_X shape: (total # of training examples, sequence_length, input_dim)
            # train_Y shape: (total # of training examples, # output classes)
            #
            # Slice out local minibatches & labels => Note that we *permute* the local minibatch to
            # match the PyTorch expected input tensor format of (sequence_length, batch size, input_dim)
            X_local_minibatch, y_local_minibatch = (
                train_X[i * batch_size: (i + 1) * batch_size, ],
                train_Y[i * batch_size: (i + 1) * batch_size, ],
            )

            # Reshape input & targets to "match" what the loss_function wants
            X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

            # NLLLoss does not expect a one-hot encoded vector as the target, but class indices
            y_local_minibatch = torch.max(y_local_minibatch, 1)[1]
            
            X_local_minibatch=X_local_minibatch.to(device)
            y_local_minibatch=y_local_minibatch.to(device)
            
            y_pred, state,c = model(X_local_minibatch, state,c)  # forward pass
            
            # Stateful = False for training. Do we go Stateful = True during inference/prediction time?

            state.detach_()
            c.detach_()
            

            loss = loss_function(y_pred, y_local_minibatch)  # compute loss
            loss.backward()  # backward pass
            optimizer.step()  # parameter update

            train_running_loss += loss.detach().item()  # unpacks the tensor into a scalar value
            train_acc += model.get_accuracy(y_pred, y_local_minibatch)
        train_accuracy_list.append(train_acc / num_batches)
        train_loss_list.append(train_running_loss / num_batches)
        epoch_train_list.append(epoch)
        print(
            "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f"
            % (epoch, train_running_loss / num_batches, train_acc / num_batches)
        )

        if epoch % 10 == 0:
            print("Validation ...")  # should this be done every N=10 epochs
            val_running_loss, val_acc = 0.0, 0.0

            # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
            with torch.no_grad():
                model.eval()

                state, c = model.init_hidden(batch_size)
                state = state.to(device) 
                c = c.to(device)

                
                for i in range(num_dev_batches):
                    X_local_validation_minibatch, y_local_validation_minibatch = (
                        dev_X[i * batch_size: (i + 1) * batch_size, ],
                        dev_Y[i * batch_size: (i + 1) * batch_size, ],
                    )
                    X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
                    y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]
                    
                    X_local_minibatch=X_local_minibatch.to(device)
                    y_local_minibatch=y_local_minibatch.to(device)

                    y_pred, state,c = model(X_local_minibatch, state,c)

                    val_loss = loss_function(y_pred, y_local_minibatch)

                    val_running_loss += (
                        val_loss.detach().item()
                    )  # unpacks the tensor into a scalar value
                    val_acc += model.get_accuracy(y_pred, y_local_minibatch)

                model.train()  # reset to train mode after iterationg through validation data
                print(
                    "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f | Val Loss %.4f  | Val Accuracy: %.2f"
                    % (
                        epoch,
                        train_running_loss / num_batches,
                        train_acc / num_batches,
                        val_running_loss / num_dev_batches,
                        val_acc / num_dev_batches,
                    )
                )
                if scheduler!=None:
                    scheduler.step()

            epoch_list.append(epoch)
            val_accuracy_list.append(val_acc / num_dev_batches)
            val_loss_list.append(val_running_loss / num_dev_batches)
    
    print("Testing ...")  # should this be done every N=10 epochs
    test_running_loss, test_acc = 0.0, 0.0
    #num_test_batches = int(test_X.shape[0] / batch_size)
    # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
    conjunt_prediccions=[]
    with torch.no_grad():
        model.eval()

        state, c = model.init_hidden(test_X.shape[0])
        state = state.to(device) 
        c = c.to(device)

        
        
        X_local_test_minibatch, y_local_test_minibatch = (
            test_X[:],
            test_Y[:],
        )
        X_local_minibatch = X_local_test_minibatch.permute(1, 0, 2)
        y_local_minibatch = torch.max(y_local_test_minibatch, 1)[1]
        
        X_local_minibatch=X_local_minibatch.to(device)
        y_local_minibatch=y_local_minibatch.to(device)

        y_pred, state,c = model(X_local_minibatch, state,c)

        test_loss = loss_function(y_pred, y_local_minibatch)

        test_running_loss = test_loss.detach().item()  # unpacks the tensor into a scalar value
        test_acc = model.get_accuracy(y_pred, y_local_minibatch) *batch_size / test_X.shape[0]
        
        prediccions=torch.max(y_pred, 1)[1].view(y_local_minibatch.size()).data

    

    genre_list = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock'
                  , 'International','Electronic', 'Instrumental']
    #genres_usats=["classical","hiphop","jazz","metal","pop","reggae"]
    
    prediccio=[]
    veritat=[]
    y_local_minibatch=y_local_minibatch.cpu()
    prediccions=prediccions.cpu()
    for element in y_local_minibatch:
        veritat.append(genre_list[element])
    for element in prediccions:
        prediccio.append(genre_list[element])


    if grafiquejar:
    # visualization loss
        plt.plot(epoch_train_list, train_loss_list)
        plt.xlabel("# of epochs")
        plt.ylabel("Loss")
        plt.title("LSTM: Loss vs # epochs")
        plt.savefig('Loss.png')
        plt.show()
        plt.clf()
        # visualization accuracy
        plt.plot(epoch_train_list, train_accuracy_list, color="red")
        plt.xlabel("# of epochs")
        plt.ylabel("Accuracy")
        plt.title("LSTM: Accuracy vs # epochs")
        plt.savefig('Accuracy.png')
        plt.show()
        plt.clf()
        #visualitzar matrius
        fig = plt.figure()
        sns.heatmap(confusion_matrix(veritat,prediccio)
                    ,xticklabels=genre_list, yticklabels=genre_list,annot=True,fmt='g')
        fig.savefig('Confusion.png', dpi=400)
        fig.show()
        print("Test Accuracy:",test_acc,"| Test Loss:",test_running_loss)

