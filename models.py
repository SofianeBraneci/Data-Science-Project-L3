# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:24:57 2020

@author: hp
"""
import numpy as np
np.random.seed(42)
from Classifiers import Classifier

class RBFKernel:
    def __init__(self, p=4):
        self.p = p
    
    def transform(self, w, x):
        return np.exp(-self.p * (np.linalg.norm(w-x)**2))


class KMeans(Classifier):
    """
        KMeans algorithm for data clustering 
        ----------------
        Note: dans le cas de la classification binaire le -1 est consideré comme 0
    """
    
    def __init__(self, input_dim, k=2, tol=10, max_iter=300):
        Classifier.__init__(self, input_dim)
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.interia = None
    
    def train(self, data):
        
        self.centroids = {}
        
        # choosing centroids
        choosen = []
        i = 0
        while i < self.k:
            index = np.random.randint(0, len(data))
            if index not in choosen:
                choosen.append(index)
                self.centroids[i] = data[index]
                i += 1
            

        for j in range(self.max_iter):
            # print(counter)
            # counter += 1
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
            for x in data:
                distances = [np.linalg.norm(x -self.centroids[centroid]) for centroid in self.centroids]
                classification = np.argmin(distances)
                self.classifications[classification].append(x)
            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

            
        wcss = []
        for classe in self.classifications:
            dist = np.sum([np.linalg.norm(self.centroids[classe] - x) for x in self.classifications[classe]])
            wcss.append(dist)
        self.interia = np.sum(wcss)
            
    
    def predict(self, x):
        
         classification = np.argmin(self.score(x))
         return classification
    def score(self, x):
        return [np.linalg.norm(x - self.centroids[centroid]) for centroid in self.centroids]
    
    def toString(self):
        return "k = {}".format(self.k)
    
    def accuracy(self, X, y):
        pass
    

class SVMClassifier(Classifier):
    def __init__(self, input_dim=784, max_inter=1000, lr=0.01, kernel=None):
        self.input_dim = input_dim
        self.w = np.random.rand(input_dim) *0.10
        self.b = np.random.randn()
        self.max_iter = max_inter
        self.lr = lr
        self.error = []
        self.kernel = kernel
        self.acc = []
    def train(self, X, y):
        error = 0
        t = 0
        for _ in range(self.max_iter):
             t += 1
             #  calcule de la fonction objective
             index = np.random.randint(0, len(y))
             # error = (0.5 *np.linalg.norm(self.w)) + np.mean([np.max([0, (1-y[i])*np.dot(self.w, X[i])]) for i in range(len(y))])
             self.error.append(error)
             self.acc.append(self.accuracy(X, y))
             # training part
             n = 1/(t * self.lr)
             if y[index]*np.dot(self.w, X[index]) < 1:
                 self.w -= n*((self.lr * self.w) -(y[index]*X[index]))
             else:
                 self.w -= n*self.lr*self.w
    def score(self, X):
        return np.dot(X, self.w) 

    def predict(self, X):
        # sign(x*w + b)
        return 1 if self.score(X) >= 0  else -1
    
        
 