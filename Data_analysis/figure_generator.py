from umap import UMAP
import matplotlib.pyplot as plt
import pandas as pd
from functions_main import get_genres, gutenberg2df, data_aug_dfrantext
from Evaluation_functions import model2dists, IsometryTesting, NeighborhoodTesting,feature_selection_isometry, ConnectednessTesting
from tqdm import tqdm
from sklearn.cluster import KMeans
import pickle
import pandas as pd
import re 
from functions_main import get_genres, gutenberg2df, data_aug_dfrantext
from Evaluation_functions import model2dists, IsometryTesting, NeighborhoodTesting,feature_selection_isometry, ConnectednessTesting
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import squareform, pdist


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

"""
------------ Figure illustrant les différences entre genres ------------
"""
text_to_embed = list_gutenberg[3].loc[df_gutenberg_all.index,:]
emb_genre = UMAP(metric="hamming").fit_transform(df_gutenberg_all)
emb_texts = UMAP().fit_transform(text_to_embed)

n_clust = 20

km = KMeans(n_clusters=n_clust)
labels = km.fit_predict(emb_genre)

emb_better = UMAP().fit_transform(text_to_embed,labels)
names = ["genre_topology_in_gutenberg.png",
         "genre_topology_in_texts.png",
         "genre_topology_optimized.png"]

for e,model in enumerate([emb_genre,emb,better_emb]):
    plt.scatter(model[:,0],model[:,1],
                s=5,
                c=labels,
                cmap="tab20")
    plt.savefig(names[e],
                bbox_inches="tight",
                dpi=300)
    plt.show()

