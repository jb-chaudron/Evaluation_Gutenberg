from function import model2dists, get_genres, gen_XY, RRelief
from function import get_genres, gutenberg2df, data_aug_dfrantext
import pickle
import pandas as pd
import re 
#from functions_main import get_genres, gutenberg2df, data_aug_dfrantext
#from Evaluation_functions import model2dists, IsometryTesting, NeighborhoodTesting,feature_selection_isometry, ConnectednessTesting
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr

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


"""
------------ Supervised Feature Selection ------------
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


"""
------------ Rrelief ------------
"""
def evaluation_systematique(df_gutenberg, list_models, names,metric="euclidean", method="RRelief"):
    scores_out = pd.DataFrame(0,
                              columns=["spearman normal", "spearman rank weight", "spearman weight"],
                              index=names)
    for i in range(len(list_models)):
        X_train, y_train = gen_XY(df_gutenberg,list_models,squareform(pdist(df_gutenberg,metric=metric)),
                      10_000,i)
        X_test, y_test = gen_XY(df_gutenberg,list_models,squareform(pdist(df_gutenberg,metric=metric)),
                      10_000,i)
        if method == "RRelief":
            W = RRelief(X_train, y_train)
            
        else :
            E_net = ElasticNet(positive=True)
            E_net.fit(X_train,y_train)
            W = E_net.coef_
            W = np.where(W>0,np.log(W), -1_000)
        
        new_dists = np.linalg.norm(X_test[:,W>0]*np.exp(W[W>0]),axis=1)
        weight_dists = np.linalg.norm(X_test*np.exp(W),axis=1)
        scores_out.loc[names[i], "spearman normal"] = spearmanr(np.linalg.norm(X_test,axis=1),y_test).correlation
        scores_out.loc[names[i], "spearman rank weight"] = spearmanr(new_dists,y_test).correlation
        scores_out.loc[names[i], "spearman weight"] = spearmanr(weight_dists,y_test).correlation
        #scores_out.loc[names[i], ["spearman normal", "spearman rank weight", "spearman weight"]] = [spearman,spearman_rank_weight,spearman_weight]

        print(scores_out.loc[names[i],:])
    return scores_out

df_scores = evaluation_systematique(df_compressed_all,list_gutenberg,models_names)


"""
--------- Linear Regression w/ penalty
"""

from sklearn.linear_model import ElasticNet

df_E_net_scores = evaluation_systematique(df_compressed_all,list_gutenberg,
                                    models_names,method="elastic_net")


"""
--------- Neural Network sans embedding ------------
"""


from torch.utils.data import DataLoader, Dataset
import itertools as itr
import random

class Embedding_Dataset(Dataset):

    def __init__(self, labels, data, label_distance, n_sample):
        self.labels = labels
        self.data = data
        self.label_distance= label_distance
        self.n_sample=n_sample
        self.gen_XY()

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
    
    def robust_normalizer(self):
        third_quant = np.quantile(self.y,0.75)

        self.y = self.y/third_quant

    def gen_XY(self):
        nb_documents = len(self.labels)

        paires = [(a,b) for a,b in itr.product(range(nb_documents),range(nb_documents)) if a > b]

        p = random.choices(list(range(len(paires))), k=self.n_sample)
        p = [paires[i] for i in p]
        print(len(p))

        datas = self.data.loc[self.labels.index,:]
        X_1 = datas.iloc[[a for (a,b) in p], :].to_numpy()
        X_2 = datas.iloc[[b for (a,b) in p], :].to_numpy()

        X_1, X_2 = X_1.reshape((X_1.shape[0],1,-1)), X_2.reshape((X_2.shape[0],1,-1))
        self.X = np.concatenate((X_1,X_2),axis=1).astype(float)

        #X = datas.iloc[[a for (a,b) in p], : ].to_numpy()
        #X = X - datas.iloc[[b for (a,b) in p],:].to_numpy()
        #self.X = np.abs(X)

        #y = np.array([squareform(self.distances[ind_model])[a,b] for (a,b) in p])
        self.y = np.array([self.label_distance[a,b] for (a,b) in p])
        self.robust_normalizer()

        self.X = torch.Tensor(self.X).to(torch.float32)
        self.y = torch.Tensor(self.y).to(torch.float32)

import torch
from torch import nn

class NN_embedding(nn.Module):
    def __init__(self,
                 dim_input : int = 1,
                 dim_output: int = 1,
                 dim_hidden: int = 1,
                 n_layers : int = 1,
                 act : nn.Module = nn.Tanh(),
                 dist_max : int = 1,
                 ) -> None:
        
        super().__init__()
        # Input layers
        self.layer_in = nn.Linear(dim_input,dim_hidden)

        # Hidden Layers
        num_middle = n_layers - 1 
        self.hidden_layers = nn.ModuleList([nn.Linear(dim_hidden,dim_hidden) for _ in range(num_middle)])
        
        # Output Layers
        self.layer_out = nn.Linear(dim_hidden,dim_output)
        
        # Activation
        self.act = act 

        # For the loss
        self.dist_max = dist_max


    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        """
            Architecture = Multi Layer Perceptron
                Input : Paires 
        """
        
        x = self.act(self.layer_in(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        
        out = self.layer_out(x)

        return out
        

    def loss(self, predictions, target):
        
        predicted_distance = predictions[:,0,:] - predictions[:,1,:]
        predicted_distance = torch.reshape(predicted_distance,(predictions.shape[0],-1))
        predicted_distance = torch.norm(predicted_distance,dim=1)

        conv = torch.exp(0.15-target)
        error = torch.pow(predicted_distance-target,2)
        total = error*conv

        return torch.sum(total,dim=0)
    

def train(dataloader,model, optimizer):

    
    size = len(dataloader.dataset)
    model.train()
    loss_out = []
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        pred = model(X)
        loss = model.loss(pred, y)
        #loss = torch.sum(loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        loss, current = loss.item(), (batch + 1) * len(X)
        loss_out += [loss]
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss_out

def test(dataloader, model):
    model.eval()
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = model.loss(pred,y)

        print(loss)

i = 3
dataset = Embedding_Dataset(df_compressed_all,list_gutenberg[i],squareform(pdist(df_compressed_all,metric="euclidean")),100_000)
data_loader = DataLoader(dataset, shuffle=True, batch_size=64)

x_mock, y_mock = dataset[0]
x_mock.shape

model = NN_embedding(x_mock.shape[1], 50, 30, 3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for t in range(5):
    train(data_loader,model,optimizer)

model.eval()

tens_test = torch.Tensor(list_gutenberg[i].loc[df_compressed_all.index,:].to_numpy().astype(float))
emb = model(tens_test.to(torch.float32))

emb_genre = UMAP().fit_transform(df_compressed_all)
#emb_texts = UMAP().fit_transform(text_to_embed)

n_clust = 20

km = KMeans(n_clusters=n_clust)
labels_clust = km.fit_predict(emb_genre)
emb_nn = UMAP().fit_transform(emb.detach().numpy(),labels_clust)
#emb_better = UMAP().fit_transform(text_to_embed,labels)
names = ["genre_topology_in_gutenberg.png",
         "genre_topology_in_texts.png",
         "genre_topology_optimized.png"]

for e,mod in enumerate([emb_genre,emb_nn]):
    plt.scatter(mod[:,0],mod[:,1],
                s=5,
                c=labels_clust,
                cmap="tab20")
    plt.show()


from scipy.stats import rankdata

og = list_gutenberg[i].loc[df_compressed_all.index,:].to_numpy().astype(float)
model.eval()
emb = model(tens_test.to(torch.float32))
y_genre = df_compressed_all.to_numpy()


minimal_wiring(squareform(pdist(og)),squareform(pdist(y_genre))), minimal_wiring(squareform(pdist(emb.detach().numpy())),squareform(pdist(y_genre)))



"""
------------ NN Embedding W/ label embedding also
"""


class Double_embedding_dataset(Dataset):

    def __init__(self, labels, data, label_distance, n_sample):
        self.labels = labels
        self.data = data
        self.label_distance= label_distance
        self.n_sample=n_sample
        self.gen_XY()

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]


    def gen_XY(self):
        nb_documents = len(self.labels)

        paires = [(a,b) for a,b in itr.product(range(nb_documents),range(nb_documents)) if a > b]

        p = random.choices(list(range(len(paires))), k=self.n_sample)
        p = [paires[i] for i in p]
        print(len(p))

        datas = self.data.loc[self.labels.index,:]
        X_1 = datas.iloc[[a for (a,b) in p], :].to_numpy()
        X_2 = datas.iloc[[b for (a,b) in p], :].to_numpy()

        X_1, X_2 = X_1.reshape((X_1.shape[0],1,-1)), X_2.reshape((X_2.shape[0],1,-1))
        self.X = np.concatenate((X_1,X_2),axis=1).astype(float)

        y_1 = self.labels.iloc[[a for (a,b) in p], :].to_numpy()
        y_2 = self.labels.iloc[[b for (a,b) in p], :].to_numpy()

        y_1, y_2 = y_1.reshape((y_1.shape[0],1,-1)), y_2.reshape((y_2.shape[0],1,-1))
        self.y = np.concatenate((y_1,y_2),axis=1).astype(float)
        

        self.X = torch.Tensor(self.X).to(torch.float32)
        self.y = torch.Tensor(self.y).to(torch.float32)

class NN_double_embedding(nn.Module):

    def __init__(self,
                 dim_input : int = 1,
                 dim_output: int = 1,
                 dim_hidden: int = 1,
                 n_layers : int = 1,
                 act : nn.Module = nn.Tanh(),
                 dist_max : int = 1,
                 ) -> None:
        
        super().__init__()
        # Input layers
        self.layer_in_textual = nn.Linear(dim_input,dim_hidden)
        self.layer_in_genre = nn.Linear(dim_output,dim_output)

        # Hidden Layers
        num_middle = n_layers - 1 
        self.hidden_layers = nn.ModuleList([nn.Linear(dim_hidden,dim_hidden) for _ in range(num_middle)])
        
        # Output Layers
        self.layer_out = nn.Linear(dim_hidden,dim_output)
        
        # Activation
        self.act = act 

        # For the loss
        self.dist_max = dist_max


    def forward(self,
                x : torch.Tensor,
                y : torch.Tensor) -> torch.Tensor:
        """
            Architecture = Pair of input

        """
        
        x = self.act(self.layer_in_textual(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        
        out_x = self.layer_out(x)
        
        out_y = self.layer_in_genre(y)

        return out_x, out_y
        

    def pair_to_dist(self, input):

        difference = input[:,0,:] - input[:,1,:]
        difference = torch.reshape(difference,(input.shape[0],-1))
        #difference = nn.functional.normalize(difference,dim=1)
        difference = torch.norm(difference,dim=1)
        difference = (difference-torch.min(difference))/(torch.max(difference)-torch.min(difference))

        return difference 
    
    def loss_desmartines(self, predictions, target):
        
        predictions_distance = self.pair_to_dist(predictions)
        target_distance = self.pair_to_dist(target)
        
        #predicted_distance = predictions[:,0,:] - predictions[:,1,:]
        #predicted_distance = torch.reshape(predicted_distance,(predictions.shape[0],-1))
        #predicted_distance = torch.norm(nn.functional.normalize(predicted_distance),dim=1)

        
        conv = torch.exp(0.1-target_distance)
        error = torch.pow(predictions_distance-target_distance,2)
        total = error*conv

        return torch.mean(total,dim=0)
    
    def loss_minimal_path_lenght(self, predictions, target):
        genre = torch.reshape(target,(-1,target.shape[1]))
        genre = genre.detach().numpy()

        text = torch.reshape(predictions,(-1,predictions.shape[1]))
        text = predictions.detach().numpy()
        print(text.shape)

        dists_genre = squareform(pdist(genre))
        dists_texts = squareform(pdist(text))

        ranks = rankdata(dists_texts,axis=1,method="dense")
        dist_neighbours = np.where(ranks<15,dists_genre,0)

        return torch.Tensor(dist_neighbours.sum()/(15*dists_genre.shape[0]))
    
def train_double_embedding(dataloader,model, optimizer):

    
    size = len(dataloader.dataset)
    model.train()
    loss_out = []
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        pred_x, pred_y = model(X,y)
        loss_d = model.loss_desmartines(pred_x, pred_y)
        #loss_m = model.loss_minimal_path_lenght(pred_x,pred_y)
        loss= loss_d
        #loss = torch.sum(loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        loss, current = loss.item(), (batch + 1) * len(X)
        loss_out += [loss]
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss_out
    
i = 3
dataset = Double_embedding_dataset(df_compressed_all,list_gutenberg[i],squareform(pdist(df_compressed_all,metric="euclidean")),100_000)
data_loader = DataLoader(dataset, shuffle=True, batch_size=64)

x_mock, y_mock = dataset[0]
x_mock.shape

model = NN_double_embedding(x_mock.shape[1], y_mock.shape[1], 30, 3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_total = []
for t in range(5):
    loss_total += train_double_embedding(data_loader,model,optimizer)
plt.plot(loss_total)
plt.show()
tensor_eval_text = torch.Tensor(list_gutenberg[i].loc[df_compressed_all.index,:].to_numpy().astype(float)).to(torch.float32)
tensor_eval_genre = torch.Tensor(df_compressed_all.to_numpy()).to(torch.float32)

X_eval, y_eval = model(tensor_eval_text,tensor_eval_genre)
X_eval, y_eval = X_eval.detach().numpy(), y_eval.detach().numpy()
from umap import UMAP 

emb_text_double, emb_genre_double = UMAP().fit_transform(X_eval), UMAP().fit_transform(y_eval)

km = KMeans(20)
labels_genre = km.fit_predict(emb_genre_double)

plt.scatter(emb_genre_double[:,0],emb_genre_double[:,1], c=labels_genre, s=5,cmap="tab20")
plt.show()

plt.scatter(emb_text_double[:,0],emb_text_double[:,1], c=labels_genre, s=5,cmap="tab20")