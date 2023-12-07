import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.random as nprand
import process_word_counts
from tqdm import tqdm
import math


class NNSentiment(nn.Module):
    #Defines the nn
    def __init__(self, model_name = "NN_sentiment", learning_rate=2e-5, batch_size=8, momentum = 2e-5, num_labels = 5, num_freq = 50, l1_size = 100, l2_size = 100):
        super(NNSentiment, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_freq = num_freq
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.lr = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum

        #Define the neural network
        #self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(num_freq, l1_size),
            nn.ReLU(),
            nn.Linear(l1_size, l2_size),
            nn.ReLU(),
            nn.Linear(l2_size, l2_size),
            nn.ReLU(),
            nn.Linear(l2_size, num_labels)
        )
        
    #Runs through one step of an epoch
    def forward(self, X):
       # X = self.flatten(X)
        logits = self.linear_stack(X)
        return logits


    #Processes the data from full reviews into coherent frequencies
    def preprocess_data(self, train_X, test_X):
        features = process_word_counts.choose_features(train_X, test_X, self.num_freq) #get meaningful features
        print("features: ", features)
        #reduce entire reviews down to selected features
        train_feat = []
        test_feat = []
        for i in tqdm(train_X, "Finding train data feature values"):
            train_feat.append(list(process_word_counts.count_word_freq(i, features).values()))
        for i in tqdm(test_X, "Finding test data feature values"):
            test_feat.append(list(process_word_counts.count_word_freq(i, features).values()))
        
        train_feat = torch.tensor(np.array(train_feat, dtype=np.single))
        test_feat = torch.tensor(np.array(test_feat, dtype=np.single))
        return train_feat, test_feat

    #Trains one epoch
    def train(self, train_X, train_y, optimizer, criterion):
        running_loss = 0.0
        num_correct = 0
        
        random_order = nprand.permutation(range(len(train_X)))
        #Go through each datapoint in a random order, and optimize for it
        for i in tqdm(random_order, "Training"):
            inputs = train_X[i]
            label = train_y[i] - 1

            #zero the parameter gradients
            optimizer.zero_grad()
            outputs = self(inputs)
            predicted = torch.argmax(outputs)
            loss = criterion(outputs, label.squeeze())

            loss.backward()
            optimizer.step()

            if label == predicted:
                num_correct += 1

            running_loss += loss.item()

        perc_correct = num_correct/len(train_X)

        return running_loss, perc_correct

    #Tests network
    def test(self, test_X, test_y, criterion, confusion_matrix = False):
        running_loss = 0.0
        num_correct = 0
        if confusion_matrix:
            con_matrix = np.zeros((self.num_labels, self.num_labels))
        
        random_order = nprand.permutation(range(len(test_X)))
        #Go through each datapoint in a random order, and optimize for it
        for i in tqdm(random_order, "Testing"):
            inputs = test_X[i]
            label = test_y[i] - 1

            #zero the parameter gradients
            outputs = self(inputs)
            predicted = torch.argmax(outputs)
            loss = criterion(outputs, label.squeeze())

            if label == predicted:
                num_correct += 1

            if confusion_matrix:
                con_matrix[label][predicted] += 1

            running_loss += loss.item()

        perc_correct = num_correct/len(test_X)

        if confusion_matrix:
            return running_loss, perc_correct, con_matrix

        return running_loss, perc_correct

    #Trains and tests the network for an amount of epoch and displays the results
    def train_and_test(self, train_X, train_y, test_X, test_y, epochs):
        train_X, test_X = self.preprocess_data(train_X, test_X) #preprocess data

        train_y = torch.tensor(np.array(train_y, dtype=np.int_))
        test_y = torch.tensor(np.array(test_y, dtype=np.int_))

        #Use cross entropy loss and stochastic gradient descent
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), self.lr, self.momentum)

        train_accuracies = []
        train_losses = []
        test_accuracies = []
        test_losses = []

        #Run training and testing
        for i in tqdm(range(epochs), "Training and testing"):
            train_acc, train_loss = self.train(train_X, train_y, optimizer, criterion) #train on training set
            test_acc, test_loss = self.test(test_X, test_y, criterion); #test on testing set

            #record results to charts
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            test_accuracies.append(test_acc)
            test_losses.append(test_loss)

        #Generate confusion matrix
        final_acc, final_loss, confusion_matrix = self.test(test_X, test_y, criterion, confusion_matrix=True)
        return final_acc, final_loss, confusion_matrix


    #Just tests the network 
    def test_only(self, test_X, test_y):
        pass

    #Saves the network
    def save_model(self, save_path):
        pass

    #loads the network
    def load_model(self, load_path):
        pass

    #creates a confusion matrix from 
    def confusion_matrix(self, results_array):
        pass