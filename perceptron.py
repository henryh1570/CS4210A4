#-------------------------------------------------------------------------
# AUTHOR: Henry Hu
# FILENAME: perceptron.py
# SPECIFICATION: This program reads optdigits.tra, to build single and multi-layer perceptron
#               classifiers. A grid search is performed to find the best combination of the
#               hyperparameters learning rate & shuffle to get the best prediction performance
#               for each  classifier. Test accuracy with optdigits.tres
# FOR: CS 4210- Assignment #4
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

best_accuracy_perceptron = 0
best_accuracy_MLP = 0
best_hyper_parameters_perceptron = None
best_hyper_parameters_MLP = None

for w in n: # Training rates

    for b in r: # Shuffle options

        for a in range(2): # Algorithms

            # Create a Neural Network classifier
            if a == 0:
                # eta0 = learning rate, shuffle = shuffle the training data
                clf = Perceptron(eta0=w, shuffle=b)
                clf.n_iter_=1000
            else:
                # learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer, shuffle = shuffle the training data
                clf = MLPClassifier(activation='logistic', learning_rate_init=w, hidden_layer_sizes=(25,), shuffle =b, max_iter=1000)

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            # Predict against the test set and calculate accuracy.
            error = 0
            for (test_sample, truth) in zip(X_test, y_test):
                if clf.predict([test_sample])[0] != truth:
                    error += 1
            accuracy = (len(y_test) - error) / len(y_test)

            # Print the new best accuracy and hyper_parameters if found.
            if a == 0:
                if accuracy > best_accuracy_perceptron:
                    best_accuracy_perceptron = accuracy
                    best_hyper_parameters_perceptron = {w, b}
                    print(f'Perceptron: New Highest Accuracy: {accuracy} \nCurrent Best Hyper_parameters: \nLearning rate = {w}\nShuffle = {b}\n')
            else:
                if accuracy > best_accuracy_MLP:
                    best_accuracy_MLP = accuracy
                    best_hyper_parameters_MLP = {w, b}
                    print(f'MLP: New Highest Accuracy: {accuracy} \nCurrent Best Hyper_parameters: \nLearning rate = {w}\nShuffle = {b}\n')









