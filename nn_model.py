import torch
import torch.nn as nn
import torch.nn
import numpy as np
import parse_words


class NNSentiment(nn.Module):
    #Defines the nn
    def __init__(self, model_name = "NN_sentiment", num_labels = 5, num_freq = 52, l1_size = 100, l2_size = 100):
        super(NNSentiment, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_freq = num_freq
        self.l1_size = l1_size
        self.l2_size = l2_size

        #Define the neural network
        self.flatten = nn.Flatten()
        self.linear_stack == nn.Sequential(
            nn.Linear(num_freq, l1_size),
            nn.RelU(),
            nn.Linear(l1_size, l2_size),
            nn.RelU(),
            nn.Linear(l2_size, l2_size),
            nn.RelU(),
            nn.Linear(l2_size, num_freq)
            #nn.RelU()
        )
        
    #Runs through one step of an epoch
    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_stack(X)
        return logits


    #Processes the data from full reviews into coherent frequencies
    def preprocess_data(self, X, y):
        pass

    #Trains one epoch
    def train(self, learning_rate, momentum, batch_size, train_X, train_y):
        pass
        #criterion = nn.CrossEntropyLoss()
        

         #PARSE WORDS ###################################################


    #Tests network
    def test(self, test_X, test_y):
        pass

    #Trains and tests the network for an amount of epoch and displays the results
    def train_and_test(self, lr, momentum, batch_size, train_X, train_y, test_X, test_y):
        pass

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