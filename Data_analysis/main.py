import pickle
import pandas as pd
import re 
from functions_main import get_genres, gutenberg2df, data_aug_dfrantext
from Evaluation_functions import model2dists, IsometryTesting, NeighborhoodTesting,feature_selection_isometry, ConnectednessTesting
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import TruncatedSVD


"""
---------- Here the list of the data input ----------

    0 * Base Model - Mean
    1 * Base Model - Sum

    
    2 * Function Model - Mean
    3 * Function Model - Sum
    
    (Maximum := each paragraph is represented as the function which has the highest deviation from it's mean)
    4 * Function Model - Maximum - Sum
    5 * Function Model - Maximum - Mean
    
    (BPE := the unit is the sequence, sequences are sequences of paragraphs, which occur frequently. These sequences are found using the
         Bytes Pair Encoding algorithm over the "Function Model - Maximum", over the all corpus)
    6 * Function Model - Maximum - BPE - Sum
    7 * Function Model - Maximum - BPE - Mean

    (Markov := Document are represented by their probability transition between units, here the unit are sequences obtained by BPE)
    8 * Function Model - Maximum - BPE - Markov
    9 * Function Model - Maximum - Markov

    (Register := Paragraphs go through K-Mean clustering algorithm and are labelled accordingly)
    (Discret := Paragraphs are one hot encoded depending on the cluster they are labelled with)
    10 * Register - Discret - Mean
    11 * Register - Discret - Sum

    (Continuous := Paragraphs are encoded with a vector, in which, each dimension is the distance to each cluster centroid.
                    We use a SoftMax function with a negative Beta Parameter in order to give a soft ranking of each centroid)
    12 * Register - Continuous - Euclidean - Sum
    13 * Register - Continuous - Euclidean - Mean
    14 * Register - Continuous - Cosine - Sum
    15 * Register - Continuous - Cosine - Mean

    16 * Register - Discret - BPE - Sum
    17 * Register - Discret - BPE - Mean
    19 * Register - Discret - BPE - Markov
    20 * Register - Discret - Markov

"""

path = "/Users/jean-baptistechaudron/Documents/Thèse/Coffre Fort Thèse/Données/Evaluation_embedding/"

with open(path + "Frantext/Models/all_models.pkl", "rb") as models:
    list_frantext = pickle.load(models)

with open(path + "Gutenberg/Models/all_models.pkl", "rb") as models:
    list_gutenberg = pickle.load(models)

for i in [2,3]:
    list_gutenberg[i].index.rename("titre", inplace=True)
    list_frantext[i].index.rename("titre", inplace=True)

for i in [8,9,18,19]:
    list_gutenberg[i].set_index("titre", inplace=True)
    list_frantext[i].set_index("titre", inplace=True)


"""
------------- On récupère les labels de genre de Frantext & Gutenberg ------------
"""

genre_frantext = pd.read_csv(path + "Frantext/Data_input/labels_intersection_frantext.csv",index_col=0)
genre_gutenberg = pd.read_csv(path + "Gutenberg/Data_input/labels_intersection_gutenberg.csv",index_col=0)

bookshelf, subject, labels = get_genres(genre_gutenberg)

"""
------------ Production of genre DataFrame ------------
"""

df_frantext = pd.get_dummies(genre_frantext.genre)
df_frantext.index = list(genre_frantext.title)
# get a one hot encoded version of Frantext Corpus 
df_frantext_aug = data_aug_dfrantext(df_frantext, list_frantext[0].index )

df_gutenberg_all = gutenberg2df(labels)
df_gutenberg_bookshelf = gutenberg2df(bookshelf)
df_gutenberg_subject = gutenberg2df(subject)
compressed_all = TruncatedSVD(28).fit_transform(df_gutenberg_all)
df_compressed_all = pd.DataFrame(compressed_all,
                                 index=df_gutenberg_all.index)

relevant_model = df_compressed_all
"""
------------ Isometry testing ------------
"""
models_names = ["LG_Mean",
               "LG_Sum",
               "Fct_Mean",
               "Fct_Sum",
                "Fct_Max_Sum",
                "Fct_Max_Mean",
                "Fct_BPE_Sum",
                "Fct_BPE_Mean",
                "Fct_Markov_BPE",
                "Fct_Markov_Base",
                "Reg_Discret_Sum",
                "Reg_Discret_Mean",
                "Reg_Euclidean_Sum",
                "Reg_Euclidean_Mean",
                "Reg_Cosine_Sum",
                "Reg_Cosine_Mean",
                "Reg_BPE_Sum",
                "Reg_BPE_Mean",
                "Reg_Markov_BPE",
                "Reg_Markov_Base"
               ]

# Data Transformation
ind_frantext = [ind for ind in df_frantext_aug.index]
dists_frantext = model2dists(list_frantext,ind_frantext)

# Computing the statistics of isometry
frantext_isometry = IsometryTesting(list_frantext, df_frantext_aug, dists_frantext,
                                    models_name=models_names, frantext=True)
df_frantext_isometry = frantext_isometry.get_isometry_score(return_score=True)

# Data prepration for continous labelling
ind_gutenberg = [ind for ind in relevant_model.index]
dists_gutenberg = model2dists(list_gutenberg,ind_gutenberg)
gutenberg_isometry = IsometryTesting(list_gutenberg, relevant_model, dists_gutenberg,
                                     models_name=models_names,frantext=False)

df_gutenberg_isometry = gutenberg_isometry.get_isometry_score(return_score=True)




"""
------------ Neighborhood Testing ------------
"""
dist_frtext_genre = squareform(pdist(df_frantext_aug.to_numpy(), metric="hamming"))
frantext_neighborhood = NeighborhoodTesting(dists_frantext, dist_frtext_genre,
                                            models_name=models_names)
df_frantext_neighborhood = frantext_neighborhood.compute_scores(score_to_compute="all")


#df_gutenberg_best = df_gutenberg_all.loc[:,gutenberg_isometry.genre_features_selected]
#dist_gutenberg_genre = squareform(pdist(df_gutenberg_best.to_numpy(), metric="hamming"))
dist_gutenberg_genre = squareform(pdist(relevant_model.to_numpy()))

gutenberg_neighborhood = NeighborhoodTesting(dists_gutenberg, dist_gutenberg_genre,
                                             models_name=models_names)
df_gutenberg_neighborhood = gutenberg_neighborhood.compute_scores(score_to_compute="all")


"""
------------ Connectedness ------------
"""

data_ranked = [np.argsort(X,axis=1) for X in list_frantext]
frantext_connectedness = ConnectednessTesting(data_ranked)
frantext_connectedness.n_tests=250
models_scores = frantext_connectedness.scoring_all_models()

data_ranked = [np.argsort(X,axis=1) for X in list_gutenberg]
gutenberg_connectedness = ConnectednessTesting(data_ranked)
gutenberg_connectedness.n_tests=250
gutenberg_connectivity_scores = gutenberg_connectedness.scoring_all_models()



