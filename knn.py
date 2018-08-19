# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:35:00 2018

@author: Ramya Sahithi Adari
"""
from sklearn.datasets import load_iris

data = load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['data'],data['target'], random_state = 0)

import numpy as np

import matplotlib.pyplot as plt

class KNN(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        """
        X = X_train
        y = y_train
        """
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test, k=1): 
        """
        It takes X_test as input, and return an array of integers, which are the 
        class labels of the data corresponding to each row in X_test. 
        Hence, y_project is an array of lables voted by their corresponding 
        k nearest neighbors
        """
        """ two or more class labels receive the same maximum votes, 
            then we do not classify the input"""
        y_predict = []
        for i in range(len(X_test)):            
            distance = np.sqrt(np.sum(np.square(X_test[i,:]-X_train),axis = 1))
            distance = np.argsort(distance)
            y_project = y_train[distance[0:k]]
            unique, counts = np.unique(y_project, return_counts=True)
            my_dict = dict(zip(unique, counts))
            h = [k for k,v in my_dict.items() if v == max(my_dict.values())]
            if(len(h)==1):y_predict.append(h)
            else:y_predict.append([-1]) 
        return y_predict

    
    def report(self,X_test,y_test,k=1):
        """
        return the accurancy of the test data. 
        """
        y_predict = self.predict(X_test,k)
        match = 0
        for x in range(len(y_test)):
            if y_test[x] == y_predict[x]:
                match += 1
        accuracy = (match/float(len(y_test))) * 100.0
        return accuracy


def k_validate(X_test,y_test):
    """
    plot the accuracy against k from 1 to a certain number so that one could pick the best k
    """
    accuracy = []
    acc = 0    
    k = np.arange(112)
    for i in range(1,len(k)+1):
        neighbors = KNN()
        acc = neighbors.report(X_test,y_test,i)
        accuracy.append(acc)
    print('\n')
    print('"Plot of Accuracy against K Nearest Neighbors"')
    plt.plot(k,accuracy)
    plt.xlabel('K nearest Neighbors')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs K values')
    plt.show()

k_validate(X_test,y_test)
a = KNN()
