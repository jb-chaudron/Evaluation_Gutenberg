import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import itertools as itr
import random
import numpy as np
from scipy.stats import rankdata

class Double_embedding_dataset_neighbors(Dataset):

    def __init__(self, labels, data, label_distance, n_sample,k=15):
        self.labels = labels
        self.data = data
        self.label_distance= label_distance
        self.n_sample=n_sample
        self.gen_XY_neighbors(k=k)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx], self.X_neighbors[idx],self.y_neighbors[idx], self.X_contre[idx], self.y_contre[idx]

    def gen_XY_neighbors(self,k=15):
        np.fill_diagonal(self.label_distance,10_000)
        rank_y = rankdata(self.label_distance,axis=1,method="dense")
        rand_idx = [random.sample(list(range(rank_y.shape[1])),k=k) for _ in range(rank_y.shape[0])]
        
        y_numpy = self.labels.to_numpy()
        k_best_y = np.array([y_numpy[ranks<k+1] for ranks in rank_y])
        k_random_y = np.array([y_numpy[idx] for idx in rand_idx])

        datas = self.data.loc[self.labels.index,:]
        k_best_x = np.array([datas.iloc[ranks<k+1,:] for ranks in rank_y])
        k_random_x = np.array([datas.iloc[idx] for idx in rand_idx])

        self.X, self.y = torch.unsqueeze(torch.Tensor(datas.to_numpy()).to(torch.float32),1),torch.unsqueeze(torch.Tensor(self.labels.to_numpy()).to(torch.float32),1)
        self.y_neighbors, self.y_contre = torch.Tensor(k_best_y).to(torch.float32), torch.Tensor(k_random_y).to(torch.float32)
        self.X_neighbors, self.X_contre = torch.Tensor(k_best_x).to(torch.float32), torch.Tensor(k_random_x).to(torch.float32)


class NN_double_embedding(nn.Module):

    def __init__(self,
                 dim_input : int = 1,
                 dim_label_input : int = 1,
                 dim_output: int = 1,
                 dim_hidden: int = 1,
                 dim_latent : int = 30,
                 n_hidden : int = 1,
                 act : nn.Module = nn.Tanh(),
                 dist_max : int = 1,
                 ) -> None:
        
        super().__init__()
        # Input layers
        self.layer_in_textual = nn.Linear(dim_input,dim_hidden)
        self.layer_in_genre = nn.Linear(dim_label_input,dim_output)

        # Hidden Layers
        num_middle = int(n_hidden/2)
        size_seq = np.linspace(dim_latent,dim_hidden,num_middle)[::-1]
        
        print(size_seq)
        layers_middle = []
        for i in range(0,len(size_seq)):
            if i == 0:
                layers_middle.append(nn.Linear(dim_hidden,int(size_seq[i])))
            else:
                layers_middle.append(nn.Linear(int(size_seq[i-1]),int(size_seq[i])))
        size_seq = size_seq[::-1]
        for i in range(len(size_seq)):
            if i == len(size_seq)-1:
                layers_middle.append(nn.Linear(int(size_seq[i]),dim_hidden))
            else:
                layers_middle.append(nn.Linear(int(size_seq[i]),int(size_seq[i+1])))
        print(layers_middle)
        self.hidden_layers = nn.ModuleList(layers_middle)
        
        # Output Layers
        self.layer_out = nn.Linear(dim_hidden,dim_output)
        
        #Agreement Layer
        self.agreement = nn.Linear(dim_output,dim_output)
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
        self.x_input = x 
        self.y_input = y 

        x = self.act(self.layer_in_textual(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        x = self.layer_out(x)
        y = self.layer_in_genre(y)
        #out_x = self.act(self.layer_out(x))
        
        #out_y = self.act(self.layer_in_genre(y))
        #out_x = self.agreement(out_x)
        #out_y = self.agreement(out_y)

        return x,y
        

    def pair_to_dist(self, input):

        difference = input[:,0,:] - input[:,1,:]
        difference = torch.reshape(difference,(input.shape[0],-1))
        #difference = nn.functional.normalize(difference,dim=1)
        difference = torch.norm(difference,dim=1)
        #difference = (difference-torch.min(difference))/(torch.max(difference)-torch.min(difference))

        return difference 
    
    def loss_desmartines(self, predictions, target):
        
        predictions_distance = self.pair_to_dist(predictions)
        target_distance = self.pair_to_dist(target)
        
        #predicted_distance = predictions[:,0,:] - predictions[:,1,:]
        #predicted_distance = torch.reshape(predicted_distance,(predictions.shape[0],-1))
        #predicted_distance = torch.norm(nn.functional.normalize(predicted_distance),dim=1)

        
        conv = torch.exp(0.35-target_distance)
        error = torch.pow(predictions_distance-target_distance,2)
        total = error*conv

        return torch.sum(total,dim=0)
    def loss_neighbors2(self, ancre, target, voisins_ancre, voisins_target, contre_ancre, contre_target):
        norm = lambda a : (a-torch.min(a))/(1e-6+torch.max(a)-torch.min(a))
        
        ancre = ancre
        #print(ancre.shape, voisins_ancre.shape, contre_ancre.shape)
        l1_ancre_voisins = torch.linalg.norm(voisins_ancre-ancre,dim=-1)
        l1_ancre_contre = torch.linalg.norm(contre_ancre-ancre, dim=-1)

        target = torch.unsqueeze(target,1)
        l1_target_voisins = torch.linalg.norm(voisins_target-target, dim=-1)
        l1_target_contre = torch.linalg.norm(contre_target-target, dim=-1)

        c_voisins = norm(torch.flatten(l1_ancre_voisins))*norm(torch.flatten(l1_target_voisins))
        c_contre = norm(torch.flatten(l1_ancre_contre))*norm(torch.flatten(l1_target_contre))

        return torch.sum(c_voisins-c_contre)
        

    def loss_neighbors(self, predictions, target,k=5):
        genre = torch.reshape(target,(-1,target.shape[-1]))
        text = torch.reshape(predictions,(-1,predictions.shape[-1]))
        
        g_dist = torch.cdist(genre,genre)
        t_dist = torch.cdist(text, text)

        
        predict_top_k = torch.topk(t_dist,k=k,largest=False).indices
        #target_top_k = torch.topk(g_dist,k=k,largest=False).indices
        randoms = torch.randint(low=0,high=t_dist.shape[1],size=(predict_top_k.shape[0],predict_top_k.shape[1]))
        #print(g_dist.shape, t_dist.shape, predict_top_k.shape, target_top_k.shape)

        norm = lambda a : (a-torch.min(a))/(1e-6+torch.max(a)-torch.min(a))
        c_measure = norm(g_dist[predict_top_k])*norm(t_dist[predict_top_k])

        random_c_measure = norm(g_dist[randoms]*norm(t_dist[randoms]))
        return c_measure - random_c_measure
        #out = torch.Tensor(torch.sum(predict_top_k==target_top_k),requires_grad=True, device=torch.device("cpu"),dtype=torch.float32)
        #return out
        
    def loss_magnitude(self, predictions, target):
        loss_x = torch.sum(predictions,dim=-1)-torch.sum(self.x_input,dim=-1)
        loss_y = torch.sum(target,dim=-1)-torch.sum(self.y_input,dim=-1)

        return torch.sum(loss_x)+torch.sum(loss_y)
    
    def loss_minimal_path_lenght(self, predictions, target):
        genre = torch.reshape(target,(-1,target.shape[-1]))
        text = torch.reshape(predictions,(-1,predictions.shape[-1]))
        
        g_dist = torch.reshape(nn.functional.pdist(genre),(1,-1))
        t_dist = torch.reshape(nn.functional.pdist(text),(1,-1))

        corr_mat = torch.cat((g_dist,t_dist))
        corr = torch.corrcoef(corr_mat)
        #print(torch.mean(corr))
        #return 1-torch.mean(corr)
        
        genre = genre.detach().numpy()

        
        text = text.detach().numpy()
        print(text.shape)

        dists_genre = pdist(genre)
        dists_texts = pdist(text)

        corr = 1-spearmanr(dists_genre,dists_texts)[0]
        return torch.Tensor([corr]).to(torch.float32)

        ranks = rankdata(dists_texts,axis=1,method="dense")
        dist_neighbours = np.where(ranks<15,dists_genre,0)

        return torch.Tensor(dist_neighbours.sum()/(15*dists_genre.shape[0]))

from sklearn.preprocessing import MinMaxScaler 

def train_double_embedding(dataloader,model, optimizer):

    
    size = len(dataloader.dataset)
    model.train()
    loss_out = []
    for batch, (X, y, neigh_X, neigh_y, contre_X, contre_y) in enumerate(dataloader):

        # Compute prediction error
        pred_x, pred_y = model(X,y)
        neigh_X, neigh_y = model(neigh_X,neigh_y)
        contre_X, contre_y = model(contre_X,contre_y)
        loss_d = model.loss_neighbors2(pred_x, pred_y, neigh_X, neigh_y, contre_X, contre_y)
        #loss_mag = model.loss_magnitude(pred_x,pred_y)
        #loss_m = model.loss_minimal_path_lenght(pred_x,pred_y)
        loss=  torch.sum(loss_d)
        #loss = torch.sum(loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        loss, current = loss.item(), (batch + 1) * len(X)
        loss_out += [loss]
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss_out
"""
feed = df_train
MMS = MinMaxScaler()
df_train = big_df.groupby(by="META : titre").mean()
dataset = Double_embedding_dataset_neighbors(df_labels
,pd.DataFrame(MMS.fit_transform(feed),index=feed.index,columns=feed.columns),squareform(pdist(df_labels.to_numpy(),metric="euclidean")),100_000,k=40)
data_loader = DataLoader(dataset, shuffle=True, batch_size=64)

x_mock, y_mock, _, _, _, _ = dataset[0]
x_mock.shape
#sop+=1
ANN = NN_double_embedding(dim_input = x_mock.shape[1], 
                            dim_label_input= y_mock.shape[1],
                            dim_output = 40, 
                            dim_hidden = 20, 
                            dim_latent = 5,
                            n_hidden=6)
optimizer = torch.optim.Adam(ANN.parameters(), lr=1e-3)
loss_total = []
for t in range(100):
    loss_total += train_double_embedding(data_loader,ANN,optimizer)
plt.plot(loss_total)
plt.show()

ANN.eval()

emb_x, emb_y = ANN(torch.Tensor(MMS.transform(feed.to_numpy())).to(torch.float32),torch.Tensor(cat_emb).to(torch.float32))

from sklearn.cluster import KMeans

km = KMeans(20)


emb_umap_y = UMAP().fit_transform(emb_y.detach().numpy())


labels = km.fit_predict(emb_y.detach().numpy())
emb_umap_x = UMAP().fit_transform(emb_x.detach().numpy(),labels)
fir,ax = plt.subplots(ncols=2,figsize=(15,5))

ax[0].scatter(emb_umap_y[:,0],emb_umap_y[:,1],c=labels,cmap="tab20")
ax[1].scatter(emb_umap_x[:,0],emb_umap_x[:,1],c=labels,cmap="tab20")"""