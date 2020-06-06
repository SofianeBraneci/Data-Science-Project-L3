# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd

# ---------------------------
# ------------------------ A COMPLETER :
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        if input_dimension <= 0:
            raise ValueError('input_dimention doit être positife')
        self.input_dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def reset(self):
        """
            reset the classifier
        """
        pass

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # TODO !!!!
        acc = 0
        for i in range(len(label_set)):
            if (self.predict(desc_set[i]) * label_set[i]) > 0:
                acc += 1
#         print(acc)
        return (acc/ len(label_set)) * 100
    def toString(self):
        pass
        # raise NotImplementedError("Please Implement this method")
# ------------------------ A COMPLETER :
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        self.w = np.random.randn(input_dimension)
        self.w_copy = np.copy(self.w)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifier !")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)
        # raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) > 0:
            return 1
        else:
            return -1
        #raise NotImplementedError("Please Implement this method")
    def reset(self):
        self.w = np.copy(self.w_copy)
    def toString(self):
        return ""
    
# ------------------------ A COMPLETER :
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        self.learning_rate = learning_rate
        self.w = np.random.rand(input_dimension) - 0.5 
        self.w_copy = np.copy(self.w)
    
    def toString(self):
        return "w = {0}, learning rate = {}".format(self.w, self.lr)  
        
    def train(self, data_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """ 
#        // print(self.w)
        
    
#         for data, label in zip(data_desc, labels):
#             print(data, label)
#         for _ in range(1000):
        for _ in range(1000):
            index = np.random.randint(len(data_set))
            data = data_set[index,:]
#             v = np.ones(3)
#             v[:len(data)] = data.copy()
            label = label_set[index]
            if (data @ self.w)* label < 0:
                self.w = self.w + (self.learning_rate *  data *  label)
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
#         y = np.ones(3)
#         y[:len(x)] = x
        return (self.w @ x )
       #  raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
#         y = np.ones(len(x) + 1)
#         y[:len(x)] = x
        if (self.score(x)) > 0:
            return +1
        else:
            return -1
#         raise NotImplementedError("Please Implement this method")
    def rest(self):
        self.w = np.copy(self.w_copy)

    
# ---------------------------
# ------------------------ A COMPLETER :

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    #TODO: A Compléter
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        self.k = k
    
    def toString(self,):
        return "k = {}".format(self.k)
        
    def distance(self, x):
        return np.array([np.linalg.norm(x - y) for y in self.desc_set])
        
        
    def sim(self, x):
        return np.array([np.exp(-((np.linalg.norm(x - y)**2)/(2.0* (np.var(self.desc_set)**2) ))) for y in self.desc_set])
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist = self.distance(x)
#         print(dist)
        indices = np.argsort(dist)
#         print(indices[:self.k])
#         print(self.label_set[indices[:self.k]])
        nearest = self.label_set[indices[:self.k]]
#         print(nearest)
        ones = nearest[nearest == 1]
        return np.float(len(ones)/self.k)
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if self.score(x) >= 0.5:
            return +1
        return  -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
        
class Kernel():
    """ Classe pour représenter des fonctions noyau
    """
    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim
    
    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """        
        raise NotImplementedError("Please Implement this method")

class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """
    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3            
            rajoute une 3e dimension au vecteur donné
        """
        V_proj = np.asarray([V[0],V[1],1])
        return V_proj
        

class KernelPoly(Kernel):
    def transform(self,V):
        """ ndarray de dim 2 -> ndarray de dim 6            
            ...
        """
        V_proj = np.asarray([1,V[0],V[1],V[0]*V[0],V[1]*V[1],V[0]*V[1]])
        return V_proj
    

# ------------------------ A COMPLETER :
class ClassifierPerceptronKernel(Classifier):
    def __init__(self, input_dimension, learning_rate, noyau):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : 
                - noyau : Kernel à utiliser
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.lr = learning_rate
        self.noyau = noyau
        self.w = np.random.randn(6)
        self.w_copy = np.copy(self.w)
    
    def toString(self):
        return " w = {0}, lr = {1}, noyau = {2}".format(self.w, self.lr, self.noyau.__class__.__name__)
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.noyau.transform(x),self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if (self.score(x) > 0):
            return 1
        return -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
#         print(len(desc_set), len(label_set))
        for _ in range(1000):
            index = np.random.randint(0, len(desc_set))
            x = desc_set[index]
            y = label_set[index]
            if self.predict(x) * y < 0:
                self.w += self.lr * self.noyau.transform(x) * y
        #print(self.w)
        #return V_proj
    def reset(self):
        self.w = np.copy(self.w_copy)
        
        
class KNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
        Cette classe n'utilise pas la distance euclidienne mais une fonction
        de <<similarité>>
    """

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        self.k = k
    
    def toString(self,):
        return "k = {}".format(self.k)
        
    def cosine_similarity(self, x):
        return np.array([(np.dot(x, y) / len(x)*len(y)) for y in self.desc_set])
        
        
    def sim(self, x):
        return np.array([np.exp(-((np.linalg.norm(x - y)**2)/(2.0* (np.var(self.desc_set)**2) ))) for y in self.desc_set])
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist = self.co
#         print(dist)
        indices = np.argsort(dist)[::-1]
#         print(indices[:self.k])
#         print(self.label_set[indices[:self.k]])
        nearest = self.label_set[indices[:self.k]]
#         print(nearest)
        ones = nearest[nearest == 1]
        return np.float(len(ones)/self.k)
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if self.score(x) >= 0.5:
            return +1
        return  -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set





# ------------------------ A COMPLETER :
def leave_one_out(C, DS):
    """ Classifieur * tuple[array, array] -> float
    """
    X, Y = DS
    count = 0
    n = len(X)
    for i in range(len(X)):
        X_ = np.delete(X, i, axis=0)
        Y_ = np.delete(Y, i, axis=0)
        
        C.train(X_, Y_)
        if C.predict(X[i]) == Y[i] :
            count += 1
    return count / n
        
        

