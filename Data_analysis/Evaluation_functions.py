import numpy as np
from scipy.stats import rankdata, mannwhitneyu, spearmanr
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
import pandas as pd
from tqdm import tqdm 
import itertools as itr 
import random 
import networkx as nx 


def C_measure(X_genre,X_text):
    return np.multiply(np.triu(X_genre,1),np.triu(X_text,1)).sum()

def MDS_score(X_genre,X_text):
    return np.power((np.triu(X_genre,1)-np.triu(X_text,1)),2).sum()

def minimal_wiring(X_genre,X_text,k=3):

    genre_rank = rankdata(X_genre,axis=1,method="dense")
    X_neighbors = np.where(genre_rank<=k,X_text,0)
    return X_neighbors.mean() # On prend la moyenne car on est pas sûr de la pb d'avoir le même nombre d'éléments à chaque fois

def minimal_path_length(X_genre,X_text,k=3):
    """
        Symmétrique du Minimal Wiring
            On veut minimiser la distance dans l'input space pour des voisins dans l'output space
            À l'inverse du Minimal Wiring, le calcul est donc le même
    """
    score = minimal_wiring(X_text,X_genre,k)
    return score


def Demartines_Herault(X_genre,X_text):
    #genre_rank = rankdata(X_genre,axis=1,method="dense")
    #text_rank = rankdata(X_text,axis=1,method="dense")

    softmax = lambda a : np.exp(a)/np.sum(np.exp(a))
    return np.multiply(np.power((X_genre-X_text),2),softmax(X_text)).sum()

def Quantization_error(X_genre,X_text):
    pass

def Jones_et_al(X_genre,X_text,k=3):

    genre_rank = rankdata(X_genre,axis=1,method="min")
    text_rank = rankdata(X_text,axis=1,method="min")

    F_ij = np.where(genre_rank<k,1,0)
    G_ij = np.where(text_rank<k,1,0)

    return np.multiply(np.triu(F_ij,1),np.triu(G_ij,1)).sum()


def Topographic_Product(X_text,X_genre,k=10):
    """
        Q_1 : Pour chaque point, on
    """
    genre_rank = rankdata(X_genre,axis=1,method="dense")
    text_rank = rankdata(X_text,axis=1,method="dense")

    Q_k = lambda inp,out,rank,l : np.where(rank<l,max(out/inp,0.001),0)

    P = [(np.multiply(Q_k(X_text,X_genre,text_rank,kl),Q_k(X_genre,X_text,genre_rank,kl)),kl) for kl in range(1,X_text.shape[1])]
    P = np.array([np.power(np.log(qs),1/kl) for (qs,kl) in P])
    P = P.sum()*(1/(X_text.shape[1]*(X_text.shape[1]-1)))

    return P


def density_neigh():
    pass

def connectedness_score(X_text,labels):
    """
        # 1 - On récupère l'index ou le nom des textes qui doivent être connectés

        # 2 - On calcul la densité de bon texte par rayon pour chaque bon textes
                [density(texte) for texte in good_text ]

                density
                    A - sorted_list = sorted(liste_de_textes, key = distance, reverse = True )
                    B - [text in good_text for text in sorted_list]


        ---- Pseudo code ----

        G = nx.Graph()
        G.add_nodes_from(good_nodes)

        while len(connected_components(G)) > 1:
            edges, scores = density_score(X_text,labels, n_neigh) # On attribut un score aux différents liens possibles pour n_neigh
            good_edges = selection_edge(edges,scores)

            G.add_edges_from(good_edges)

    """

def feature_normalization(X_in, frantext=True):
    
    X_in = RobustScaler().fit_transform(X_in)
    try: 
        X_out = VarianceThreshold(0.1).fit_transform(X_in)
        return X_out 
    except Exception:
        return X_in


def model2dists(list_in, index_in):
    
    out = [feature_normalization(X.loc[index_in,:].to_numpy()) for X in list_in]
    out = [pdist(X) for X in out]

    return out


class IsometryTesting():
    def __init__(self, data, labels, distances, 
                models_name=None, frantext=False) -> None:
        """
            data : List of dataframe, encoding vector for each documents
                    shape = (nb_models, nb_documents, nb_features)
            labels : Dataframe of genre label for each document
                    shape = (nb_documents, nb_features-labels)
            distances : pairwise distance of each document, for each model
                        shape = (nb_models, nb_documents, nb_documents)

            frantext : Boolean value to know if we process frantext or Gutenberg dataset
            nb_models : number of models in the list of data

        """
        self.data = data
        self.labels = labels
        self.frantext = frantext
        self.nb_models = len(data)
        self.distances = distances
        self.models_name = models_name
    """
        Function for preprocessing

        We want to
            1 - For frantext
                A - For all model
                B - Find groups that will be compared
                C - Compute the statistics
            2 - For Gutenberg
                A - For all model
                B - Compute the correlation or R2 scores
    """

    def get_isometry_score(self, regressor = None, return_score = False):

        if self.frantext:
            self.label_comparison()
            if return_score:
                return self.df_isometrie
        else:
            if regressor == None:
                regressor = OrthogonalMatchingPursuit()
            
            self.distances_correlations(regressor)


    """
        Functions for Frantext (Discret genre labeling)
    """
    
    def get_genre_index(self, genre):
        genre_index = list(self.labels.index[self.labels[genre] == 1])

        good_index = [e for e,g in enumerate(self.labels.index) if g in genre_index]
        bad_index = [e for e,g in enumerate(self.labels.index) if not g in genre_index]

        return good_index, bad_index

    def median_distance(self, ind, good_index, bad_index):
        distance_matrix = squareform(self.distances[ind])

        g_dists = []
        for i in range(len(good_index)-1):
            g_dists += list(distance_matrix[good_index[i],good_index[i+1:]].flatten())
        
        b_dists = []
        for i in range(len(good_index)):
            b_dists += list(distance_matrix[good_index[i],bad_index].flatten())

        return g_dists, b_dists

    def label_comparison(self):

        self.frantext_genres = ["genre narratif","poésie","théâtre","traité"]

        if self.models_name == None:
            self.df_isometrie = pd.DataFrame(0,
                                        columns = self.frantext_genres,
                                        index = list(range(self.nb_models)))
        else :
            self.df_isometrie = pd.DataFrame(0,
                                        columns = self.frantext_genres,
                                        index = self.models_name)
        # Shape = (n_models, n_genre, 2, _) : the "2" dimension is for the
        #                                     within vs without
        self.distance_comparison = []

        for ind, genre in tqdm(itr.product(self.df_isometrie.index, self.df_isometrie.columns)):
            # We aim at getting the within distance and outsider distances of docs
            g, b = self.get_genre_index(genre)
            g_dist, b_dist = self.median_distance(ind,g,b)
            self.distance_comparison += [[g_dist,b_dist]]

            # We stack the data together
            U = mannwhitneyu(g_dist,b_dist, alternative="less")
            effect_size, pvalue = U.statistic/(len(g_dist)*len(b_dist)), U.pvalue
            score = effect_size * (pvalue < 0.01)

            self.df_isometrie.loc[ind, genre] = score

    """
        Functions for continuous genre labelling
    """
    def distances_correlations(self, regressor):
        if self.models_name == None:
            self.df_isometrie = pd.DataFrame(0,
                                            index=range(len(self.distances)),
                                            columns=["R2","Spearman"])
        else :
            self.df_isometrie = pd.DataFrame(0,
                                            index=self.models_name,
                                            columns=["R2","Spearman"])

        for e,model in enumerate(self.df_isometrie.index):
            self.df_isometrie.loc[model,"R2"] = self.r2(regressor, 250_000, e)
            self.df_isometrie.loc[model,"Spearman"] = self.correlation(e)
        
    def correlation(self, model_index):
        X = np.linalg.norm(self.X)
        corr = spearmanr(X,self.y)

        return corr.statistic if corr.pvalue < 0.01 else 0

    def r2(self, regressor, n_samples, ind_model):
        normalize = lambda a : (a - a.min()) / (a.max() - a.min())
        X,y = self.gen_XY(n_samples, ind_model)
        
        X = np.abs(X)
        y = normalize(y)

        X,y = self.select_best(X,y)
        self.X, self.y = X, y 
        regressor.fit(X,y)
        
        return r2_score(y, regressor.predict(X))

    def gen_XY(self, n_samples, ind_model):
        nb_documents = len(self.labels.index)
        paires = [(a,b) for a,b in itr.product(nb_documents,nb_documents) if a > b]

        p = random.choices(list(range(len(paires))), k=n_samples)
        p = [paires[i] for i in p]

        X = self.labels.loc[[a for (a,b) in p], : ].to_numpy()
        X = X - self.labels.loc[[b for (a,b) in p],:].to_numpy()

        y = np.array([self.distances[ind_model][a,b] for (a,b) in p])

        return X, y

    def select_best(self, X, y):
        skb = SelectKBest(f_regression, k="all")
        X,y = skb.fit(X,y)
        significance_mask = skb.pvalues_ < 0.01

        self.genre_features_selected = self.labels.columns[significance_mask]
        self.genre_features_mask = significance_mask

        return X[:,significance_mask], y 
        

class NeighborhoodTesting():
    """
        Return scores for different scores
        All the scores are to be maximized, when they aren't we choose to take
            the negative value of these. Thus by looking at models with the higher
            score we automatically get the best score
    """

    def __init__(self, text_distances, genre_distances, models_name=None) -> None:
        self.text_distances = text_distances
        self.genre_distance = genre_distances
        self.models_name = models_name if models_name != None else range(len(text_distances))
        self.df_scores = pd.DataFrame(0,
                                    index=["None"],
                                    columns = ["None"])

    def compute_scores(self, score_to_compute = "all", return_score = True):
        existing_functions = ["c_measure",
                            "mds_score",
                            "minimal_wiring",
                            "minimal_path_length",
                            "demartines_herault",
                            "topographic_product"]
        if score_to_compute == "all":
            if self.df_scores.columns[0] == "None":
                self.df_scores = pd.DataFrame(0,
                                             index=self.models_name,
                                             columns=existing_functions)
                score_to_compute = existing_functions
            else:
                score_to_compute = [fct for fct in existing_functions if not fct in self.df_scores.columns]
        
        if existing_functions[0] in score_to_compute:
            self.df_scores["c_measure"] = self.C_measure()
        elif existing_functions[1] in score_to_compute :
            self.df_scores["mds_score"] = self.MDS_score()
        elif existing_functions[2] in score_to_compute :
            self.df_scores["minimal_wiring"] = self.minimal_wiring()
        elif existing_functions[3] in score_to_compute :
            self.df_scores["minimal_path_length"] = self.minimal_path_length()
        elif existing_functions[4] in score_to_compute :
            self.df_scores["demartines_herault"] = self.Demartines_Herault()
        elif existing_functions[5] in score_to_compute :
            self.df_scores["topographic_product"] = self.topographic_product()
        
        if return_score:
            return self.df_scores


    def C_measure(self):
        out = []
        genre = self.genre_distance
        for model in self.text_distances:
            out += [C_measure(genre, model)]
        
        return out
        
    def MDS_score(self):
        out = []
        genre = self.genre_distance
        for model in self.text_distance:
            out += [-MDS_score(genre,model)]
        
        return out

    def minimal_wiring(self):
        out = []
        genre = self.genre_distance
        for model in self.text_distance:
            out += [minimal_wiring(genre,model)] 
        
        return out

    def minimal_path_length(self):
        out = []
        genre = self.genre_distance
        for model in self.text_distance:
            out += [minimal_path_length(genre,model)]
        
        return out

    def Demartines_Herault(self):
        out = []
        genre = self.genre_distance
        for model in self.text_distance:
            out += [-Demartines_Herault(genre,model)]
        
        return out

    def topographic_product(self):
        out = []
        genre = self.genre_distance
        for model in self.text_distance:
            out += [-Topographic_Product(genre,model)] 
        
        return out


class ConnectednessTesting():

    def __init__(self, data_ranked) -> None:
        self.data_ranked = data_ranked

    def get_connectedness_scores(self):
        self.connectivite()

    def get_nearest_neighborhood_graph(self, neighbor_rank, class_nodes):
        G = nx.Graph()
        G.add_nodes_from(class_nodes)

        for ranking_array in neighbor_rank[class_nodes]:
            for i in range(1,len(class_nodes)):
                if i in ranking_array[class_nodes]:
                    node_a = class_nodes[0]
                    node_b = np.where(class_nodes == i)[0][0]
                if node_a == node_b:
                    continue
                G.add_edge(node_a,node_b,weight=1/i)
        
        return G 
    
    def connectivite(self,neighbor_rank, class_nodes, return_graph=False):
        G = self.get_nearest_neighborhood_graph(neighbor_rank,class_nodes)

        component = [x for x in nx.connected_components(G)]
        links = []

        while len(component) > 1:
            minimal_rank = []
            
            for i in range(len(class_nodes)):
                neighbors = [x for e,x in enumerate(class_nodes) if ((not e == i) and (not (class_nodes[i],x) in G.edges))]
                rank_neighbors = neighbor_rank[class_nodes[i],neighbors]

                minimal_rank += [rank_neighbors.min()]

                links += [(class_nodes[i],) for x in np.where(neighbor_rank[class_nodes[i],:] == rank_neighbors.min())[0]]

            minimal_rank = minimal_rank.min()

            links = [(a,b, 1/c) for (a,b,c) in links if c == minimal_rank]
            G.add_weighted_edges_from(links)
            
            if len(class_nodes)/len(G.nodes) < 0.2:
                if return_graph:
                    return len(class_nodes)/len(G.nodes), len(G.edges)/((len(G.edges)*(len(G.edges)-1))/2), G
                else: 
                    return len(class_nodes)/len(G.nodes), len(G.edges)/((len(G.edges)*(len(G.edges)-1))/2)
            else:
                component = [x for x in nx.connected_components(G)]

        if return_graph:
            return len(class_nodes)/len(G.nodes), len(G.edges)/((len(G.edges)*(len(G.edges)-1))/2), G
        else: 
            return len(class_nodes)/len(G.nodes), len(G.edges)/((len(G.edges)*(len(G.edges)-1))/2)

