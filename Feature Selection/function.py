import itertools as itr
import random
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from sklearn.feature_selection import (SelectKBest, VarianceThreshold,
                                       f_classif, f_regression)
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm


def get_genres(df_in):
    """
        Reçoit les méta données des textes et en fait une list de propriétées
    """
    bookshelves = {titre : str(bookshelf).split(";") for (titre,bookshelf) in zip(list(df_in["Title"]),list(df_in["Bookshelves"]))}
    
    subject = {titre : str(sujet).replace("--",".") for (titre,sujet) in zip(list(df_in["Title"]),list(df_in["Subjects"]))}
    subject = {titre : list(set(re.split('[,;:-=.]+',str(s)))) for (titre,s) in subject.items()}
    
    labels = {titre : [x[1:] if x[0] == " " else x for x in book+subject[titre] if len(x) > 0] for (titre,book) in bookshelves.items()}
    
    return bookshelves, subject, labels

def gutenberg2df(dict_in,thresh=10):
    un = np.unique([genre for key,value in dict_in.items() for genre in value])
    df_out = pd.DataFrame(0,
                         index = [key for key in dict_in.keys()],
                         columns = un)
    
    for key in tqdm(dict_in.keys()):
        df_out.loc[key,:] = [1 if genre in dict_in[key] else np.nan for genre in df_out.columns]
    
    df_out = df_out.dropna(1,thresh=thresh) 
    return df_out.fillna(0)

def data_aug_dfrantext(df_in,labels):
    list_index = [lab.split("_vers_")[0] for lab in labels]
    print(len(list_index),len(df_in))
    df_out = pd.DataFrame(0,
                         columns=df_in.columns,
                         index=[str(e) for e in range(len(list_index))])
    
    for e,ind in enumerate(list_index):
        if ind in df_in.index:
            row = df_in.loc[ind,:].to_numpy()
            try:
                df_out.loc[str(e),:] = row
            except:
                df_out.loc[str(e),:] = row[0]
        else:
            df_out.drop(index=str(e),inplace=True)
            
    df_out.index = [l for e,l in enumerate(labels) if str(e) in df_out.index]
    return df_out
def model2dists(list_in, index_in):
    
    out = [feature_normalization(X.loc[index_in,:].to_numpy()) for X in list_in]
    out = [pdist(X) for X in out]

    return out


def feature_normalization(X_in, frantext=True):
    
    X_in = RobustScaler().fit_transform(X_in)
    try: 
        X_out = VarianceThreshold(0.1).fit_transform(X_in)
        return X_out 
    except Exception:
        return X_in
    
def feature_selection_isometry(list_model_in, labels_dists,frantext=True):
    out = [select_best_isometry(X,labels_dists, frantext) for  X in list_model_in]

    list_df_out = [pd.DataFrame(X,columns=col,index=model.index) for model, (X,col) in zip(list_model_in,out)]

    return list_df_out

def select_best_isometry(X, y, frantext):
    columns = X.columns
    X = feature_normalization(X,frantext)
    if frantext:
        skb = SelectKBest(f_classif, k="all")
    else:
        skb = SelectKBest(f_regression, k="all")

    X,y = skb.fit(X,y)
    significance_mask = skb.pvalues_ < 0.01

    genre_features_selected = columns[significance_mask]
    genre_features_mask = significance_mask

    return X[:,significance_mask], genre_features_selected


"""
------------ RRelieff -------------

    1) Pick N random instances
    2) Find K nearest neighbors
    3) Pour chaque voisins 
        P_1 : Calcule la différence de valeur à prédire
    4) Pour chaque features de chaque voisins
        P_2 : Calcule la différence en terme de feature
        P_3 : Calcule la différence en terme de feature X la différence en terme de valeur à prédire
    5) Fait la somme de tout les calculs P_1, P_2 et P_3
    6) Pour chaque feature
        Les nouveaux poids = [P_3 / P_1] - [(P_2 - P_3)/(N - P_1)]


"""



def RRelief(X_in, y_in, n_iter=100, nn = 10):
    Ndc = 0
    Nda, Ndadc, W = np.zeros((X_in.shape[1])), np.zeros((X_in.shape[1])), np.zeros((X_in.shape[1]))



    dist_matrix = pdist(X_in)
    X_eval = random.sample(list(range(X_in.shape[0])),k=n_iter)

    for e,idx in enumerate(X_eval):
        R_i = X_in[idx,:]
        tau_i = y_in[idx]

        neighbours = get_neighbours(dist_matrix, idx, nn)
        neighbours_vectors = X_in[neighbours,:]
        neighbours_values = y_in[neighbours]

        distances = dist_computation(R_i, neighbours_vectors)

        for f,neigh in enumerate(neighbours):
            Ij = neighbours_vectors[f]
            d_neigh = distances[f]
            diff_pred = diff(tau_i, neighbours_values[f], y_in)
            diff_attribute = diff(R_i,Ij,X_in,True)
            # Compute the magnitude of difference in prediction value by neighbours
            
            Ndc += diff_pred * d_neigh
            
            # Compute the magnitude of difference in attributes
            Nda += diff_attribute * d_neigh
            Ndadc += diff_attribute * diff_pred * d_neigh
        
        W = (Ndadc/Ndc) - (Nda - Ndadc)/(n_iter-Ndc)

    return W

def diff(y_true, y_neigh, mat_val, attribute=False):
    if attribute:
        return np.abs(y_true-y_neigh)/(mat_val.max(0) - mat_val.min(0))
    else:
        return np.abs(y_true-y_neigh)/(mat_val.max() - mat_val.min())

def dist_computation(vec_true, vec_neigh,sigma=2):
    mat = cdist(vec_true.reshape((1,-1)),vec_neigh)
    rank = rankdata(mat.reshape((-1)))

    out = np.exp(-(rank/sigma)**2)
    Z = np.sum(out)

    return out/Z


        

def get_neighbours(dist_mat, idx, nn):
    mat = squareform(dist_mat)

    idx_out = sorted(range(mat.shape[1]),key=lambda a : mat[idx,a])
    idx_out = [ind for ind in idx_out if ind != idx]

    return idx_out[:nn]



"""
------------ Function for dataset generation ------------
"""


def gen_XY(labels, data, label_distances, n_samples, ind_model):
    nb_documents = len(labels.index)
    paires = [(a,b) for a,b in itr.product(range(nb_documents),range(nb_documents)) if a > b]

    p = random.choices(list(range(len(paires))), k=n_samples)
    p = [paires[i] for i in p]
    print(len(p))

    datas = data[ind_model].loc[labels.index,:]
    X = datas.iloc[[a for (a,b) in p], : ].to_numpy()
    X = X - datas.iloc[[b for (a,b) in p],:].to_numpy()
    X = np.abs(X)

    #y = np.array([squareform(self.distances[ind_model])[a,b] for (a,b) in p])
    y = np.array([label_distances[a,b] for (a,b) in p])

    return X, y