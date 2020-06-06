# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de LU3IN026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
def plot2DSet(desc,labels):
    """ ndarray * ndarray -> affichage
    """
    # Ensemble des exemples de classe -1:
    negatifs = desc[labels == -1]
    # Ensemble des exemples de classe +1:
    positifs = desc[labels == +1]
    # Affichage de l'ensemble des exemples :
    plt.scatter(negatifs[:,0],negatifs[:,1],marker='o') # 'o' pour la classe -1
    plt.scatter(positifs[:,0],positifs[:,1],marker='x') # 'x' pour la classe +1
    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])    
    
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
        par défaut: binf vaut -1 et bsup vaut 1
    """  
    data = np.random.uniform(low, high, (n, p))
    labels = np.asarray([-1 for i in range(0,n//2)] + [+1 for i in range(0,n//2)])
    np.random.shuffle(labels)
    return data, labels
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    data_negative = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    data_positive = np.random.multivariate_normal(negative_center, negative_sigma,nb_points)
    data = np.vstack((data_negative, data_positive))
    labels = np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])
    return data, labels
# ------------------------ 
def create_XOR(n, var):
    #TODO: A Compléter    
    p, l1 = genere_dataset_gaussian(np.array([0,0]), np.array([[var, 0], [0, var]]), np.array([1,0]), np.array([[var, 0],[0, var]]),n)
    q, l2 = genere_dataset_gaussian(np.array([1,1]), np.array([[var, 0], [0, var]]), np.array([0,1]), np.array([[var, 0],[0, var]]),n)
    data = np.vstack((p, q))
    labels = np.concatenate((l1, l2))
    return data, labels
 # ------------------------ 
    
def cree_dataframe(DS, L_noms, Nom_label = "label"):
    """ Dataset * List[str] * Str -> DataFrame
        Hypothèse: la liste a autant de chaînes que la description a de colonnes
    """
    X, Y = DS
    dfX = pd.DataFrame(X, columns=L_noms)
    dfY = pd.DataFrame(Y, columns=[Nom_label])
    df = pd.concat((dfX, dfY), axis=1)
    return df

# ------------------------ A COMPLETER
def categories_2_numeriques(DF,nom_col_label =''):
    """ DataFrame * str -> DataFrame
        nom_col_label est le nom de la colonne Label pour ne pas la transformer
        si vide, il n'y a pas de colonne label
        rend l'équivalent numérique de DF
    """
    dfloc = DF.copy()  # pour ne pas modifier DF
    L_new_cols = []    # pour mémoriser le nom des nouvelles colonnes créées
    Noms_cols = [nom for nom in dfloc.columns if nom != nom_col_label]
     
    for c in Noms_cols:
        if dfloc[c].dtypes != 'object':  # pour détecter un attribut non catégoriel
            L_new_cols.append(c)  # on garde la colonne telle quelle dans ce cas
        else:
            for v in dfloc[c].unique():
                col = c + '_' + v
                dfloc[col] = 0
                dfloc.loc[dfloc[c] == v, col] = 1
                L_new_cols.append(col)
                
            
    return dfloc[L_new_cols]  # on rend que les valeurs numériques


# ------------------------ A COMPLETER :
class AdaptateurCategoriel:
    """ Classe pour adapter un dataframe catégoriel par l'approche one-hot encoding
    """
    def __init__(self,DF,nom_col_label=''):
        """ Constructeur
            Arguments: 
                - DataFrame représentant le dataset avec des attributs catégoriels
                - str qui donne le nom de la colonne du label (que l'on ne doit pas convertir)
                  ou '' si pas de telle colonne. On mémorise ce nom car il permettra de toujours
                  savoir quelle est la colonne des labels.
        """
        self.DF = DF  # on garde le DF original  (rem: on pourrait le copier)
        self.nom_col_label = nom_col_label 
        
        # Conversion des colonnes catégorielles en numériques:
        self.DFcateg = categories_2_numeriques(DF, nom_col_label)
        
        # Pour faciliter les traitements, on crée 2 variables utiles:
        self.data_desc = self.DFcateg.values
        self.data_label = self.DF[nom_col_label].values
        # Dimension du dataset convertit (sera utile pour définir le classifieur)
        self.dimension = self.data_desc.shape[1]
                
    def get_dimension(self):
        """ rend la dimension du dataset dé-catégorisé 
        """
        return self.dimension
        
        
    def train(self,classifieur):
        """ Permet d'entrainer un classifieur sur les données dé-catégorisées 
        """        
        # A COMPLETER
        classifieur.train(self.data_desc, self.data_label)
    
    
    def accuracy(self,classifieur):
        """ Permet de calculer l'accuracy d'un classifieur sur les données
            dé-catégorisées de l'adaptateur.
            Hypothèse: le classifieur doit avoir été entrainé avant sur des données
            similaires (mêmes colonnes/valeurs)
        """
        return classifieur.accuracy(self.data_desc,self.data_label)

    def converti_categoriel(self,x):
        """ transforme un exemple donné sous la forme d'un dataframe contenant
            des attributs catégoriels en son équivalent dé-catégorisé selon le 
            DF qui a servi à créer cet adaptateur
            rend le dataframe numérisé correspondant             
        """
        # A COMPLETER
        cols = [col for col in self.DF.columns if col != self.nom_col_label]
        for col in cols:
            if self.DF[col].dtype != 'object':
                continue
            else:
                for v in self.DF[col].unique():
                    new = col +'_'+v
                    x[new] = 0
                    x.loc[x[col] == v, new] = 1
        return x
    def predict(self,x,classifieur):
        """ rend la prédiction de x avec le classifieur donné
            Avant d'être classifié, x doit être converti
        """
        x_df = self.converti_categoriel(x)
        return classifieur.predict(x_df[self.DFcateg.columns].values)
    
    def cross_validation(self, classifier):
        """
            Leave One Out cross validation
        """
        count = 0
        for i in range(len(self.data_desc)):
            X, Y = np.delete(self.data_desc, i, axis=0), np.delete(self.data_label, i, axis=0) 
            classifier.train(X,Y)
            if classifier.predict(self.data_desc[i]) > 0:
                count += 1
        return count / self.dimension