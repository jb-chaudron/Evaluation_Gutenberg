# The project

In this project we use custom Neural Networks to perform an homeomorphism which will aim to transform representations of the same data to make them more similar.

For this we :

1) Scrap Wikipedia and :
    - Extract features from the text to have one representations of them
    - Extract the categories of the articles to have another space representation
2) We create a neural network with a custom shape and a custom loss function for it to be minimized
3) We assess the performance of our algorithm
 

# Creating the Neural Network

## The theoretical aim

We consider here two different ways to generate a topology.
1) A topology induced by a metric
    - A topology $\mathcal{T}$ is induced by a metric function $d$ applied over a set of elements $\mathcal{A}$ 
    - We have here two sets $\mathcal{A}_{text}$ and $\mathcal{A}_{categories}$ and we consider the same metric function over the two spaces.
    - Thus, in order to have the same topology, $\mathcal{A}_{text}$ and $\mathcal{A}_{categories}$ must be identical.
    - Our Neural Network will aim to find the function performing the transformation of the textual features mapping them as close as possible to the categories vectors
2) A topology by neighbouhood
    - A topology can also be understood through the neihbourhood function. If every vector in the set $\mathcal{A}_{text}$ has the same neighbours as in the set $\mathcal{A}_{categories}$
    - Thus our Neural Network will aim to make the categorical neighbours closer than the non neighbours in the textual vector space $\mathcal{A}_{text}$ 

## The basic Neural Network Architecture

- Our Neural Network will take two inputs
    - A) Vectors from $\mathcal{A}_{text}$
    - B) Vectors from $\mathcal{A}_{categories}$
- Each input goes though its path
    - A) A multi-layer perceptron 
    - B) An embedding layer
- The loss is computed as follow:
    - 1) The pairwise distance of every vectors outputed by the two pathes is computed
    - 2) The correlation between the two pairwise distances are computed

The path "A" is longer than the path "B" because we consider the categorical topology to be a noisy `*Ground Truth*', thus we do not allow much transformation of the input in the path "B".

We also Normalize the last layer, for not the Neural Network to find the trivial solution where every vectors are nuls.

![alt text](DoubleEmbedding_induced.png "Neural Network Architecture")


```python
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt 

BAD_ARTICLE = 'Wang Yun (Dynastie Yuan)'

path_global = "/Users/jean-baptistechaudron/Documents/Thèse/Coffre Fort Thèse"

df_text = pd.read_parquet(path_global+"/df_train_wiki_out.parquet.gzip")
df_categories = pd.read_parquet(path_global+"/Wikipedia_categories_labelling_smoothed.parquet.gzip")

df_text = df_text.fillna(0)
df_categories = df_categories.fillna(0)
# We remove an article which had some problems
df_text = df_text.loc[[x for x in df_text.index if not x == BAD_ARTICLE],:]
df_categories = df_categories.loc[[x for x in df_categories.index if not x == BAD_ARTICLE],:]

# We preprocess the categories
tsvd = TruncatedSVD(500).fit(df_categories)

```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /Users/jean-baptistechaudron/Documents/Thèse/Coffre Fort Thèse/Code/GitHub/Evaluation_Gutenberg/Wikipedia/Topological Embedding.ipynb Cellule 4 in <cell line: 11>()
          <a href='vscode-notebook-cell:/Users/jean-baptistechaudron/Documents/Th%C3%A8se/Coffre%20Fort%20Th%C3%A8se/Code/GitHub/Evaluation_Gutenberg/Wikipedia/Topological%20Embedding.ipynb#W3sZmlsZQ%3D%3D?line=7'>8</a> path_global = "/Users/jean-baptistechaudron/Documents/Thèse/Coffre Fort Thèse"
         <a href='vscode-notebook-cell:/Users/jean-baptistechaudron/Documents/Th%C3%A8se/Coffre%20Fort%20Th%C3%A8se/Code/GitHub/Evaluation_Gutenberg/Wikipedia/Topological%20Embedding.ipynb#W3sZmlsZQ%3D%3D?line=9'>10</a> df_text = pd.read_parquet(path_global+"/df_train_wiki_out.parquet.gzip")
    ---> <a href='vscode-notebook-cell:/Users/jean-baptistechaudron/Documents/Th%C3%A8se/Coffre%20Fort%20Th%C3%A8se/Code/GitHub/Evaluation_Gutenberg/Wikipedia/Topological%20Embedding.ipynb#W3sZmlsZQ%3D%3D?line=10'>11</a> df_categories = pd.read_parquet(path_global+"/Wikipedia_categories_labelling_smoothed.parquet.gzip")
         <a href='vscode-notebook-cell:/Users/jean-baptistechaudron/Documents/Th%C3%A8se/Coffre%20Fort%20Th%C3%A8se/Code/GitHub/Evaluation_Gutenberg/Wikipedia/Topological%20Embedding.ipynb#W3sZmlsZQ%3D%3D?line=12'>13</a> df_text = df_text.fillna(0)
         <a href='vscode-notebook-cell:/Users/jean-baptistechaudron/Documents/Th%C3%A8se/Coffre%20Fort%20Th%C3%A8se/Code/GitHub/Evaluation_Gutenberg/Wikipedia/Topological%20Embedding.ipynb#W3sZmlsZQ%3D%3D?line=13'>14</a> df_categories = df_categories.fillna(0)


    File ~/mambaforge/envs/umap-env/lib/python3.10/site-packages/pandas/io/parquet.py:503, in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, **kwargs)
        456 """
        457 Load a parquet object from the file path, returning a DataFrame.
        458 
       (...)
        499 DataFrame
        500 """
        501 impl = get_engine(engine)
    --> 503 return impl.read(
        504     path,
        505     columns=columns,
        506     storage_options=storage_options,
        507     use_nullable_dtypes=use_nullable_dtypes,
        508     **kwargs,
        509 )


    File ~/mambaforge/envs/umap-env/lib/python3.10/site-packages/pandas/io/parquet.py:251, in PyArrowImpl.read(self, path, columns, use_nullable_dtypes, storage_options, **kwargs)
        244 path_or_handle, handles, kwargs["filesystem"] = _get_path_or_handle(
        245     path,
        246     kwargs.pop("filesystem", None),
        247     storage_options=storage_options,
        248     mode="rb",
        249 )
        250 try:
    --> 251     result = self.api.parquet.read_table(
        252         path_or_handle, columns=columns, **kwargs
        253     ).to_pandas(**to_pandas_kwargs)
        254     if manager == "array":
        255         result = result._as_manager("array", copy=False)


    File ~/mambaforge/envs/umap-env/lib/python3.10/site-packages/pyarrow/parquet/core.py:2986, in read_table(source, columns, use_threads, metadata, schema, use_pandas_metadata, read_dictionary, memory_map, buffer_size, partitioning, filesystem, filters, use_legacy_dataset, ignore_prefixes, pre_buffer, coerce_int96_timestamp_unit, decryption_properties, thrift_string_size_limit, thrift_container_size_limit)
       2975         # TODO test that source is not a directory or a list
       2976         dataset = ParquetFile(
       2977             source, metadata=metadata, read_dictionary=read_dictionary,
       2978             memory_map=memory_map, buffer_size=buffer_size,
       (...)
       2983             thrift_container_size_limit=thrift_container_size_limit,
       2984         )
    -> 2986     return dataset.read(columns=columns, use_threads=use_threads,
       2987                         use_pandas_metadata=use_pandas_metadata)
       2989 warnings.warn(
       2990     "Passing 'use_legacy_dataset=True' to get the legacy behaviour is "
       2991     "deprecated as of pyarrow 8.0.0, and the legacy implementation will "
       2992     "be removed in a future version.",
       2993     FutureWarning, stacklevel=2)
       2995 if ignore_prefixes is not None:


    File ~/mambaforge/envs/umap-env/lib/python3.10/site-packages/pyarrow/parquet/core.py:2614, in _ParquetDatasetV2.read(self, columns, use_threads, use_pandas_metadata)
       2606         index_columns = [
       2607             col for col in _get_pandas_index_columns(metadata)
       2608             if not isinstance(col, dict)
       2609         ]
       2610         columns = (
       2611             list(columns) + list(set(index_columns) - set(columns))
       2612         )
    -> 2614 table = self._dataset.to_table(
       2615     columns=columns, filter=self._filter_expression,
       2616     use_threads=use_threads
       2617 )
       2619 # if use_pandas_metadata, restore the pandas metadata (which gets
       2620 # lost if doing a specific `columns` selection in to_table)
       2621 if use_pandas_metadata:


    KeyboardInterrupt: 



```python
import numpy as np 

n_comp = sum(np.cumsum(tsvd.explained_variance_ratio_)<0.95)
print(n_comp)
val = np.cumsum(tsvd.explained_variance_ratio_)[n_comp]
plt.plot(np.cumsum(tsvd.explained_variance_ratio_))
plt.plot(list(range(0,500)),[val for _ in range(500)])

```

    450





    [<matplotlib.lines.Line2D at 0x1567f7490>]




    
![png](Topological%20Embedding_files/Topological%20Embedding_4_2.png)
    



```python
df_categories = pd.DataFrame(TruncatedSVD(n_comp).fit_transform(df_categories),
                             index=df_categories.index)
```


```python
index_train, index_test = train_test_split(np.unique([x for x in df_text.index if x in df_categories.index]),
                                           train_size=0.6,
                                           shuffle=True,
                                           random_state=2060954)
#index_test, index_val = train_test_split(index_test,
#                                           train_size=0.4,
#                                           shuffle=True,
#                                           random_state=2060954)
"""
X_train, X_test, y_train, y_test = train_test_split(df_text, df_categories,
                                                    train_size=0.7, shuffle=True,
                                                    random_state=2060954)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                train_size=0.7, shuffle=True,
                                                random_state=2060954)
"""

X_train, X_test, X_val = df_text.loc[index_train,:], df_text.loc[index_test,:], df_text.loc[index_test,:]
y_train, y_test, y_val = df_categories.loc[index_train,:], df_categories.loc[index_test,:], df_categories.loc[index_test,:]

```

## Data Preparation

We need now to create our DataLoader for the training of the Neural Network.
In Pytorch a dataset class must implement two functions
* __len__ : Which gives the length of the dataset we will use
* __getitem__ : Which, given an index value, return the input, output and every parameter needed for one iteration of the training of our Neural Network

In the __init__ function we generate our dataset, then the __getitem__ go fetch the data in the dataset


```python
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import random 
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
        return self.X[idx], self.y[idx], self.X_neighbors[idx],self.y_neighbors[idx], self.X_contrefactual[idx], self.y_contrefactual[idx]

    def gen_XY_neighbors(self,k=15):
        """
            The generator of the dataset work as follow
            1 - We rank the data in terms of Y values
                We select random values of y to be paired with a given y
            2 - We select the K-best y value for each y values
                We create the vector of best value and random ones
            3 - We select, given the same indices, the random and good values of X
        """
        np.fill_diagonal(self.label_distance,10_000)
        rank_y = rankdata(self.label_distance,axis=1,method="ordinal")
        rand_idx = [random.sample(list(range(rank_y.shape[1])),k=k) for _ in range(rank_y.shape[0])]
        
        y_numpy = self.labels.to_numpy()
        k_best_y = np.array([y_numpy[ranks<k+1].astype(float) for ranks in rank_y])
        k_random_y = np.array([y_numpy[idx].astype(float) for idx in rand_idx])

        datas = self.data.loc[self.labels.index,:]
        k_best_x = np.array([datas.iloc[ranks<k+1,:].to_numpy().astype(float) for ranks in rank_y])
        k_random_x = np.array([datas.iloc[idx].to_numpy().astype(float) for idx in rand_idx])

        print(k_best_x.shape, k_best_y.shape)
        self.X, self.y = torch.unsqueeze(torch.Tensor(datas.to_numpy()).to(torch.float32),1),torch.unsqueeze(torch.Tensor(self.labels.to_numpy()).to(torch.float32),1)
        self.y_neighbors, self.y_contrefactual = torch.Tensor(k_best_y).to(torch.float32), torch.Tensor(k_random_y).to(torch.float32)
        self.X_neighbors, self.X_contrefactual = torch.Tensor(k_best_x).to(torch.float32), torch.Tensor(k_random_x).to(torch.float32)

```


```python
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

MMS = MinMaxScaler()

dataset = Double_embedding_dataset_neighbors(y_train.fillna(0)
                                             ,pd.DataFrame(MMS.fit_transform(X_train),index=X_train.index,columns=X_train.columns),
                                             squareform(pdist(y_train.to_numpy(),metric="jaccard")),
                                             100_000,
                                             k=15)

validation_dataset = Double_embedding_dataset_neighbors(y_val.fillna(0),
                                                        pd.DataFrame(MMS.transform(X_val), index= X_val.index, columns=X_val.columns),
                                                        squareform(pdist(y_val.to_numpy(),metric="jaccard")),
                                                        50_000,
                                                        k=15)

test_dataset = Double_embedding_dataset_neighbors(y_test.fillna(0),
                                                  pd.DataFrame(MMS.transform(X_test),index=X_test.index,columns=X_test.columns),
                                                  squareform(pdist(y_test.to_numpy(),metric="jaccard")),
                                                  50_000,
                                                  k=15)

```

    (11378, 15, 79) (11378, 15, 450)
    (7583, 15, 79) (7583, 15, 450)
    (7583, 15, 79) (7583, 15, 450)


## The Neural Network implementation


```python
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
    
    def loss_desmartines(self, predictions, target, alpha=-0.01):
        
        #predictions_distance = self.pair_to_dist(predictions)
        #target_distance = self.pair_to_dist(target)
        predictions_distance = torch.cdist(predictions,predictions)
        target_distance = torch.cdist(target,target)

        predictions_distance = nn.functional.normalize(predictions_distance)
        target_distance = nn.functional.normalize(target_distance)
        #target_distance /= torch.sum(target_distance)
        #predicted_distance = predictions[:,0,:] - predictions[:,1,:]
        #predicted_distance = torch.reshape(predicted_distance,(predictions.shape[0],-1))
        #predicted_distance = torch.norm(nn.functional.normalize(predicted_distance),dim=1)

        
        conv = torch.exp(alpha*target_distance)
        error = torch.pow(predictions_distance-target_distance,2)
        total = error*conv

        return torch.sum(total,dim=0)
    
    def loss_neighbors2(self, ancre, target, voisins_ancre, voisins_target, contre_ancre, contre_target):
        norm = lambda a : (a-torch.min(a))/(1e-6+torch.max(a)-torch.min(a))
        
        #ancre = ancre
        #print(ancre.shape, voisins_ancre.shape, contre_ancre.shape)
        l1_ancre_voisins = torch.linalg.norm(voisins_ancre-ancre,dim=-1)
        l1_ancre_contre = torch.linalg.norm(contre_ancre-ancre, dim=-1)

        #target = torch.unsqueeze(target,1)
        l1_target_voisins = torch.linalg.norm(voisins_target-target, dim=-1)
        l1_target_contre = torch.linalg.norm(contre_target-target, dim=-1)

        #print(l1_ancre_voisins.shape, l1_target_voisins.shape)
        c_voisins = norm(torch.flatten(l1_ancre_voisins))*norm(torch.flatten(l1_target_voisins))
        c_contre = norm(torch.flatten(l1_ancre_contre))*norm(torch.flatten(l1_target_contre))

        return torch.sum(c_voisins-c_contre)
     
```

## The training procedure


```python

def train_double_embedding(model, optimizer, training_dataset, validation_dataset):

    size = len(training_dataset.dataset)
    model.train()
    training_loss, validation_loss = [], []
    for batch, (X, y, neigh_X, neigh_y, contre_X, contre_y) in enumerate(training_dataset):

        # Compute prediction error
        pred_x, pred_y = model(X,y)
        #neigh_X, neigh_y = model(neigh_X,neigh_y)
        #contre_X, contre_y = model(contre_X,contre_y)
        
        #print(torch.squeeze(pred_x,1).shape, torch.squeeze(pred_y,1).shape)
        loss_d = model.loss_desmartines(torch.squeeze(pred_x,1), torch.squeeze(pred_y,1))
        
        loss=  torch.sum(loss_d)
        #loss = torch.sum(loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        loss, current = loss.item(), (batch + 1) * len(X)
        training_loss += [loss]
        #print(f"training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    model.eval()

    for batch, (X, y, neigh_X, neigh_y, contre_X, contre_y) in enumerate(validation_dataset):

        # Compute prediction error
        pred_x, pred_y = model(X,y)
        neigh_X, neigh_y = model(neigh_X,neigh_y)
        contre_X, contre_y = model(contre_X,contre_y)
        
        loss_d = model.loss_desmartines(torch.squeeze(pred_x,1), torch.squeeze(pred_y,1))
        
        loss=  torch.sum(loss_d)
        
        loss, current = loss.item(), (batch + 1) * len(X)
        validation_loss += [loss]
        #print(f"validation loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Training loss = ",np.mean(training_loss)," | Validation Loss = ",np.mean(validation_loss))
    return  training_loss, validation_loss
```

## Putting it all together


```python
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

data_loader = DataLoader(dataset, shuffle=True, batch_size=32)
data_loader_validation = DataLoader(validation_dataset, shuffle=True, batch_size=32)

x_mock, y_mock, _, _, _, _ = dataset[0]
ANN = NN_double_embedding(dim_input = x_mock.shape[1],                             
                          dim_label_input= y_mock.shape[1],
                            dim_output = 100, 
                            dim_hidden = 50, 
                            dim_latent = 20,
                            n_hidden=3)

```

    [20.]
    [Linear(in_features=50, out_features=20, bias=True), Linear(in_features=20, out_features=50, bias=True)]



```python
import matplotlib.pyplot as plt

optimizer = torch.optim.Adam(ANN.parameters(), lr=1e-3)
training_loss_total, validation_loss_total = [],[]
for t in range(500):
    loss_train, loss_val = train_double_embedding(ANN,optimizer,data_loader, data_loader_validation)
    training_loss_total.append(np.mean(loss_train))
    validation_loss_total.append(np.mean(loss_val))

fig, ax = plt.subplots(ncols=2, figsize=(15,5))
ax[0].plot(training_loss_total)
ax[1].plot(validation_loss_total)

plt.show()
```

    Training loss =  4.586685508489609  | Validation Loss =  3.695779786331241
    Training loss =  3.192562997341156  | Validation Loss =  3.590349599278929
    Training loss =  2.777000430259812  | Validation Loss =  3.338201331186898
    Training loss =  2.5402457154868694  | Validation Loss =  3.170481997200205
    Training loss =  2.334294359335739  | Validation Loss =  2.883866782932845
    Training loss =  2.1520117695411938  | Validation Loss =  2.688811637681245
    Training loss =  1.8261459004343226  | Validation Loss =  2.3917059968795455
    Training loss =  1.665838910622543  | Validation Loss =  2.175831596559613
    Training loss =  1.5476354482803452  | Validation Loss =  1.9436850537730672
    Training loss =  1.4339933864186318  | Validation Loss =  1.866753722041971
    Training loss =  1.3469293552838015  | Validation Loss =  1.7581385489757555
    Training loss =  1.2496319373002214  | Validation Loss =  1.6742430887644804
    Training loss =  1.2050633448898123  | Validation Loss =  1.5433830691792292
    Training loss =  1.1602863965744383  | Validation Loss =  1.595863198177724
    Training loss =  1.1049982024712508  | Validation Loss =  1.4783657658452223
    Training loss =  1.0802181630991818  | Validation Loss =  1.4464157653760306
    Training loss =  1.0463238239623187  | Validation Loss =  1.4017624093007437
    Training loss =  1.023128118575289  | Validation Loss =  1.3823354078244559
    Training loss =  1.0026865109299006  | Validation Loss =  1.3472673191299922
    Training loss =  0.9871130513676097  | Validation Loss =  1.337433684978807
    Training loss =  0.9556936661848862  | Validation Loss =  1.2961013935789276
    Training loss =  0.9337565651100673  | Validation Loss =  1.297445335217166
    Training loss =  0.9215353723992122  | Validation Loss =  1.2784854261684016
    Training loss =  0.917828671383054  | Validation Loss =  1.2632308657662275
    Training loss =  0.8876705885435758  | Validation Loss =  1.2686786251732065
    Training loss =  0.8818207846933537  | Validation Loss =  1.247811559886369
    Training loss =  0.8676655901114593  | Validation Loss =  1.242381269670237
    Training loss =  0.8565406172295634  | Validation Loss =  1.2110940283360863
    Training loss =  0.8458574310447393  | Validation Loss =  1.2153948646054489
    Training loss =  0.8431761166018047  | Validation Loss =  1.2239247726488718
    Training loss =  0.8295679516002034  | Validation Loss =  1.214719158687672
    Training loss =  0.8202364603790004  | Validation Loss =  1.2092012873681788
    Training loss =  0.8173436156651946  | Validation Loss =  1.2011817482453357
    Training loss =  0.800417479671789  | Validation Loss =  1.179302054366985
    Training loss =  0.7872227029351706  | Validation Loss =  1.165176632283609
    Training loss =  0.7879830728588479  | Validation Loss =  1.1497153715242314
    Training loss =  0.7794163248009895  | Validation Loss =  1.153891053632342
    Training loss =  0.7729149879364485  | Validation Loss =  1.1754910825174065
    Training loss =  0.7668970097484213  | Validation Loss =  1.1684282935118373
    Training loss =  0.7628547045287122  | Validation Loss =  1.1537875349511577
    Training loss =  0.7564557284284174  | Validation Loss =  1.157105722507847
    Training loss =  0.7484295865458049  | Validation Loss =  1.1360322392942532
    Training loss =  0.7466916030042627  | Validation Loss =  1.1175876702437422
    Training loss =  0.7446811499722888  | Validation Loss =  1.1241158832980611
    Training loss =  0.7362772311722294  | Validation Loss =  1.119258902243924
    Training loss =  0.7350953668355942  | Validation Loss =  1.1061094627098695
    Training loss =  0.7324587893619966  | Validation Loss =  1.1035183746100479
    Training loss =  0.7269276322776013  | Validation Loss =  1.1336364388968874
    Training loss =  0.7258950768226988  | Validation Loss =  1.1115954794964207
    Training loss =  0.7164885297082784  | Validation Loss =  1.114281886740576
    Training loss =  0.715450798695007  | Validation Loss =  1.0863229556928706
    Training loss =  0.7110897584745054  | Validation Loss =  1.1057663668057083
    Training loss =  0.7031438303797433  | Validation Loss =  1.107758469722442
    Training loss =  0.7000103778718563  | Validation Loss =  1.0919299319323608
    Training loss =  0.6993524515059557  | Validation Loss =  1.0892332666533908
    Training loss =  0.6914841925327697  | Validation Loss =  1.0861647154208478
    Training loss =  0.6930463442976555  | Validation Loss =  1.0979216616867966
    Training loss =  0.6841828749420937  | Validation Loss =  1.071357917685046
    Training loss =  0.6888546591226974  | Validation Loss =  1.0781378952259755
    Training loss =  0.6855473464794373  | Validation Loss =  1.0719692199039057
    Training loss =  0.6790957260835037  | Validation Loss =  1.0726166191483348
    Training loss =  0.6815511908256606  | Validation Loss =  1.076369963366271
    Training loss =  0.6743004573195168  | Validation Loss =  1.0648098266074426
    Training loss =  0.6698146437326175  | Validation Loss =  1.0509644533008462
    Training loss =  0.6701940701583798  | Validation Loss =  1.0532708047311516
    Training loss =  0.6686345887150658  | Validation Loss =  1.0648395524749272
    Training loss =  0.6651020850358385  | Validation Loss =  1.054322461538677
    Training loss =  0.6647735289141034  | Validation Loss =  1.045908872336778
    Training loss =  0.6583784469560291  | Validation Loss =  1.0404499017236606
    Training loss =  0.6603054356541527  | Validation Loss =  1.042546997593425
    Training loss =  0.6544370185793116  | Validation Loss =  1.033134458688744
    Training loss =  0.6513868232121628  | Validation Loss =  1.0389008511973836
    Training loss =  0.642805008238621  | Validation Loss =  1.020488388930695
    Training loss =  0.6446305453107598  | Validation Loss =  1.0190261954496682
    Training loss =  0.6396193468336309  | Validation Loss =  1.0126888960725648
    Training loss =  0.6343492785196626  | Validation Loss =  1.0257384042699629
    Training loss =  0.6328382333008091  | Validation Loss =  1.0005250002261457
    Training loss =  0.6282383445441053  | Validation Loss =  1.0091668244152632
    Training loss =  0.623982809484005  | Validation Loss =  1.012428979330425
    Training loss =  0.6223089862405584  | Validation Loss =  0.9907319515566283
    Training loss =  0.6183220133017958  | Validation Loss =  1.0033791751801213
    Training loss =  0.6181514686747883  | Validation Loss =  0.9954072810426543
    Training loss =  0.6153494871399375  | Validation Loss =  0.9964490195869896
    Training loss =  0.6122496903277515  | Validation Loss =  0.9748674070281822
    Training loss =  0.6095584191465646  | Validation Loss =  0.9807269012374717
    Training loss =  0.610011192520013  | Validation Loss =  0.9929703524344078
    Training loss =  0.6072132216578119  | Validation Loss =  0.975366856729934
    Training loss =  0.6047009863377957  | Validation Loss =  0.9760023803147586
    Training loss =  0.6025704456011901  | Validation Loss =  0.9901298885607015
    Training loss =  0.6051856220773096  | Validation Loss =  0.9661440655651978
    Training loss =  0.5993250969253229  | Validation Loss =  0.9657268056386634
    Training loss =  0.6010546575436432  | Validation Loss =  0.972197527120888
    Training loss =  0.5912930066331049  | Validation Loss =  0.9547432052435251
    Training loss =  0.593700937675626  | Validation Loss =  0.9618596530161829
    Training loss =  0.5879311884722013  | Validation Loss =  0.9583770833940949
    Training loss =  0.5917836631951707  | Validation Loss =  0.9618394822510989
    Training loss =  0.5864391152778369  | Validation Loss =  0.9404209993056607
    Training loss =  0.5891493415732062  | Validation Loss =  0.9483674138407164
    Training loss =  0.580439127814234  | Validation Loss =  0.9319973462241612
    Training loss =  0.5787074569236027  | Validation Loss =  0.9433836972160179
    Training loss =  0.5703287695565921  | Validation Loss =  0.9234012274802486
    Training loss =  0.5650271583306655  | Validation Loss =  0.912099917981192
    Training loss =  0.5616245096486606  | Validation Loss =  0.9135509108189289
    Training loss =  0.5548872305603509  | Validation Loss =  0.9027505094491983
    Training loss =  0.5524138988236363  | Validation Loss =  0.8909081140147985
    Training loss =  0.546369565588035  | Validation Loss =  0.8758755441959397
    Training loss =  0.5385714436180136  | Validation Loss =  0.8696222468770506
    Training loss =  0.5329403251911817  | Validation Loss =  0.8679873465485713
    Training loss =  0.5283548351419106  | Validation Loss =  0.8763216110221445
    Training loss =  0.5271402092629605  | Validation Loss =  0.8612191170328277
    Training loss =  0.5239212060912272  | Validation Loss =  0.839044718928478
    Training loss =  0.5153618784767858  | Validation Loss =  0.8458728845612409
    Training loss =  0.5147000706932517  | Validation Loss =  0.8423023349625148
    Training loss =  0.5052245105250498  | Validation Loss =  0.8284879941729051
    Training loss =  0.50101185899772  | Validation Loss =  0.8139341108909639
    Training loss =  0.4986071839426341  | Validation Loss =  0.8245560699122868
    Training loss =  0.4991853062189027  | Validation Loss =  0.8204659252478603
    Training loss =  0.4886570967147859  | Validation Loss =  0.8022619116155407
    Training loss =  0.4861442773194795  | Validation Loss =  0.7949289616653185
    Training loss =  0.4790638331114576  | Validation Loss =  0.8010724067436492
    Training loss =  0.47898669423681967  | Validation Loss =  0.7924284140268961
    Training loss =  0.4730017753631881  | Validation Loss =  0.7879701235374821
    Training loss =  0.4693791131290157  | Validation Loss =  0.7864108400002814
    Training loss =  0.46414238365178695  | Validation Loss =  0.7740342424640173
    Training loss =  0.4605165755146005  | Validation Loss =  0.7669187580482869
    Training loss =  0.46349314457914803  | Validation Loss =  0.7581254452592713
    Training loss =  0.457651266639822  | Validation Loss =  0.7489624953974149
    Training loss =  0.4456442818045616  | Validation Loss =  0.7476451677360615
    Training loss =  0.4474352841799179  | Validation Loss =  0.7354455897073705
    Training loss =  0.437092134456956  | Validation Loss =  0.7347336525906993
    Training loss =  0.43701905832531746  | Validation Loss =  0.7375246120404594
    Training loss =  0.43157403514291465  | Validation Loss =  0.7346833362106533
    Training loss =  0.4299707392628273  | Validation Loss =  0.7278867014349765
    Training loss =  0.4272190329901288  | Validation Loss =  0.7193465035424453
    Training loss =  0.42189095232091595  | Validation Loss =  0.7163327055892864
    Training loss =  0.41489494156636547  | Validation Loss =  0.7166293639422469
    Training loss =  0.41772332833556647  | Validation Loss =  0.711100379500208
    Training loss =  0.4152455882409985  | Validation Loss =  0.7102138229060274
    Training loss =  0.41066606334421074  | Validation Loss =  0.7073455300763689
    Training loss =  0.4069339574387904  | Validation Loss =  0.7034357664202839
    Training loss =  0.40019564460335155  | Validation Loss =  0.6968621811534785
    Training loss =  0.3968330386063356  | Validation Loss =  0.7002946310656987
    Training loss =  0.3943879282876347  | Validation Loss =  0.6882661797326325
    Training loss =  0.39274268279249747  | Validation Loss =  0.6916675699662559
    Training loss =  0.3918144539249747  | Validation Loss =  0.684804605532296
    Training loss =  0.3832942024710473  | Validation Loss =  0.6772848898348426
    Training loss =  0.38346626040305987  | Validation Loss =  0.6814902311638941
    Training loss =  0.3829631648036871  | Validation Loss =  0.668482568445085
    Training loss =  0.3825070425616891  | Validation Loss =  0.6691023027092092
    Training loss =  0.37947977970490293  | Validation Loss =  0.6612625723146688
    Training loss =  0.37479292237189377  | Validation Loss =  0.6614590972536224
    Training loss =  0.37181763861621364  | Validation Loss =  0.6611295609031549
    Training loss =  0.3701426772756523  | Validation Loss =  0.6607737482097078
    Training loss =  0.3675827706714025  | Validation Loss =  0.6572428287836066
    Training loss =  0.3639883551369892  | Validation Loss =  0.6535038165905305
    Training loss =  0.3644265149918835  | Validation Loss =  0.6491437428611241
    Training loss =  0.36045628493086673  | Validation Loss =  0.6439411714358672
    Training loss =  0.3583857695875543  | Validation Loss =  0.6439377118515063
    Training loss =  0.35746426746416626  | Validation Loss =  0.6309328363414555
    Training loss =  0.35442744480090194  | Validation Loss =  0.6362878379690999
    Training loss =  0.3527800724161475  | Validation Loss =  0.6422702214888882
    Training loss =  0.3497496079025644  | Validation Loss =  0.6395918837579494
    Training loss =  0.35018860465020274  | Validation Loss =  0.6385865847772687
    Training loss =  0.34720324176583395  | Validation Loss =  0.6391496091200832
    Training loss =  0.34439543253752625  | Validation Loss =  0.635665424262421
    Training loss =  0.34126041126385165  | Validation Loss =  0.6289923333166018
    Training loss =  0.3355074708213967  | Validation Loss =  0.6212246219317118
    Training loss =  0.3345857099786903  | Validation Loss =  0.625726833122189
    Training loss =  0.3317460649217782  | Validation Loss =  0.6205443246958124
    Training loss =  0.3305575477608134  | Validation Loss =  0.6194986215623622
    Training loss =  0.32743270448252054  | Validation Loss =  0.6183000561557238
    Training loss =  0.3239155349353056  | Validation Loss =  0.6130639605129822
    Training loss =  0.3226922824141685  | Validation Loss =  0.6094618041052597
    Training loss =  0.31973866013328683  | Validation Loss =  0.6148781402956082
    Training loss =  0.32126347435994096  | Validation Loss =  0.6157841106507346
    Training loss =  0.31923883165536304  | Validation Loss =  0.6138689863782392
    Training loss =  0.31558600448992813  | Validation Loss =  0.6123589717386141
    Training loss =  0.31606934014498517  | Validation Loss =  0.6105596547639822
    Training loss =  0.3134965087088306  | Validation Loss =  0.5999351558806021
    Training loss =  0.31386559565415545  | Validation Loss =  0.6047702097188571
    Training loss =  0.31255204544475906  | Validation Loss =  0.5975134746183323
    Training loss =  0.3091686069128219  | Validation Loss =  0.6077004863491541
    Training loss =  0.3078788610154323  | Validation Loss =  0.6009064268965258
    Training loss =  0.3081340382105849  | Validation Loss =  0.5979394135595877
    Training loss =  0.30864081690820416  | Validation Loss =  0.594734530911667
    Training loss =  0.3062888587709893  | Validation Loss =  0.608513632152654
    Training loss =  0.30380678813109235  | Validation Loss =  0.597952268168896
    Training loss =  0.299624563393633  | Validation Loss =  0.599561310518643
    Training loss =  0.30162213650647174  | Validation Loss =  0.5943176396788424
    Training loss =  0.3024690310271938  | Validation Loss =  0.593388888906326
    Training loss =  0.30151359284861706  | Validation Loss =  0.5906560432307327
    Training loss =  0.3005543683938096  | Validation Loss =  0.5935132822909939
    Training loss =  0.29890452551372937  | Validation Loss =  0.5928857273693326
    Training loss =  0.2988808716364791  | Validation Loss =  0.6016360779100329
    Training loss =  0.2949463048049908  | Validation Loss =  0.5909019813004426
    Training loss =  0.29646018457211804  | Validation Loss =  0.5880502384423204
    Training loss =  0.2974721302979448  | Validation Loss =  0.5860493311399146
    Training loss =  0.294295399250944  | Validation Loss =  0.5859626198368233
    Training loss =  0.2943990487563476  | Validation Loss =  0.5794699952823703
    Training loss =  0.2936099506077472  | Validation Loss =  0.5832700323203445
    Training loss =  0.2968667233258151  | Validation Loss =  0.5886218544048599
    Training loss =  0.29164078413100725  | Validation Loss =  0.5867240269224352
    Training loss =  0.2907612534721246  | Validation Loss =  0.5835269157645069
    Training loss =  0.2910550528028038  | Validation Loss =  0.5821546355128792
    Training loss =  0.2923442337117838  | Validation Loss =  0.5847634581322408
    Training loss =  0.2898673237709517  | Validation Loss =  0.5859826990572209
    Training loss =  0.28742159396577416  | Validation Loss =  0.5862999938208343
    Training loss =  0.2857662371328373  | Validation Loss =  0.583266704645841
    Training loss =  0.2896167424329546  | Validation Loss =  0.5885615816599206
    Training loss =  0.2882686635751403  | Validation Loss =  0.5873441010839325
    Training loss =  0.28447638690639077  | Validation Loss =  0.5791013155305436
    Training loss =  0.2894568687325783  | Validation Loss =  0.5802418017437689
    Training loss =  0.28606292961186236  | Validation Loss =  0.5860427745032412
    Training loss =  0.2844867546822918  | Validation Loss =  0.5748240044851344
    Training loss =  0.2848788170834606  | Validation Loss =  0.5730288085303729
    Training loss =  0.2868268492558364  | Validation Loss =  0.5841070188500207
    Training loss =  0.2841841135634465  | Validation Loss =  0.5858853104748304
    Training loss =  0.28728161346209186  | Validation Loss =  0.5797295438337929
    Training loss =  0.2835149687578839  | Validation Loss =  0.5798319851044361
    Training loss =  0.283326665965024  | Validation Loss =  0.58238174756871
    Training loss =  0.28505972969565496  | Validation Loss =  0.5844620412160576
    Training loss =  0.28492911473921173  | Validation Loss =  0.5864805840741733
    Training loss =  0.2845209393692151  | Validation Loss =  0.5856626622284515
    Training loss =  0.2821608819904622  | Validation Loss =  0.5912102242059345
    Training loss =  0.28386996125572184  | Validation Loss =  0.5794685027770352
    Training loss =  0.2821368547888954  | Validation Loss =  0.5782943198198005
    Training loss =  0.28448185927412484  | Validation Loss =  0.5750967975910203
    Training loss =  0.281086912525169  | Validation Loss =  0.5879896618897402
    Training loss =  0.28215963426935536  | Validation Loss =  0.5808910697321349
    Training loss =  0.28053123596009244  | Validation Loss =  0.5859034368257482
    Training loss =  0.281797575481822  | Validation Loss =  0.5764364153775485
    Training loss =  0.2779738454964389  | Validation Loss =  0.5839286671660621
    Training loss =  0.28099827003780375  | Validation Loss =  0.5839026317566256
    Training loss =  0.2797356684891026  | Validation Loss =  0.5835669049482305
    Training loss =  0.28255170834868143  | Validation Loss =  0.5864574669031152
    Training loss =  0.2798647248091992  | Validation Loss =  0.5862257500741049
    Training loss =  0.2781058116240448  | Validation Loss =  0.5811533147021185
    Training loss =  0.2805229657319155  | Validation Loss =  0.5835336084607281
    Training loss =  0.2781382028976183  | Validation Loss =  0.5881799136284535
    Training loss =  0.27838668034652647  | Validation Loss =  0.5861358094315992
    Training loss =  0.2790774369842551  | Validation Loss =  0.5823453940168212
    Training loss =  0.27916151913029424  | Validation Loss =  0.5955632525154307
    Training loss =  0.28155544283015005  | Validation Loss =  0.5929111269958914
    Training loss =  0.27749727854735395  | Validation Loss =  0.5832821743146277
    Training loss =  0.2783049809212765  | Validation Loss =  0.5778648675997046
    Training loss =  0.27909351973218866  | Validation Loss =  0.5771183386633668
    Training loss =  0.2743102155123534  | Validation Loss =  0.5913630022278314
    Training loss =  0.27679780117246544  | Validation Loss =  0.5920951873189789
    Training loss =  0.27605791996871487  | Validation Loss =  0.5931130997239286
    Training loss =  0.2780244421172008  | Validation Loss =  0.5892751479702156
    Training loss =  0.27760067490044604  | Validation Loss =  0.5827490453478656
    Training loss =  0.2774220396126254  | Validation Loss =  0.5928775088445044
    Training loss =  0.2765219215345517  | Validation Loss =  0.5842980799041216
    Training loss =  0.27627424159076774  | Validation Loss =  0.5848214035295736
    Training loss =  0.27463102901584646  | Validation Loss =  0.5848227976998196
    Training loss =  0.27735956477817525  | Validation Loss =  0.5957440380557177
    Training loss =  0.27476550965161806  | Validation Loss =  0.5907533620228748
    Training loss =  0.27639076754115943  | Validation Loss =  0.5931017374942071
    Training loss =  0.27444795087984436  | Validation Loss =  0.5935373765255328
    Training loss =  0.2752196841527907  | Validation Loss =  0.5800381712772675
    Training loss =  0.27511069856667786  | Validation Loss =  0.584505813659998
    Training loss =  0.2750832690030671  | Validation Loss =  0.5896632511655993
    Training loss =  0.2735255897212564  | Validation Loss =  0.5925348862565519
    Training loss =  0.27425161399616954  | Validation Loss =  0.5883231058653899
    Training loss =  0.2743086518531435  | Validation Loss =  0.5856664808239112
    Training loss =  0.2716902745573708  | Validation Loss =  0.5884289435947998
    Training loss =  0.27382926128051255  | Validation Loss =  0.5928865801935961
    Training loss =  0.2740526877008797  | Validation Loss =  0.5926414311938145
    Training loss =  0.273565647600407  | Validation Loss =  0.5851325640447029
    Training loss =  0.2720996562935663  | Validation Loss =  0.583851168930279
    Training loss =  0.27116306879547203  | Validation Loss =  0.5829310847234123
    Training loss =  0.2717655723014574  | Validation Loss =  0.594856546020709
    Training loss =  0.2716390195020129  | Validation Loss =  0.5926087873394479
    Training loss =  0.2701042608849788  | Validation Loss =  0.5903112908455893
    Training loss =  0.2724012290075254  | Validation Loss =  0.5898275037103564
    Training loss =  0.27061948665742125  | Validation Loss =  0.5914248776335254
    Training loss =  0.2731398315577025  | Validation Loss =  0.5859858993488022
    Training loss =  0.2722253653356868  | Validation Loss =  0.5882117510596409
    Training loss =  0.27155954323792725  | Validation Loss =  0.5884095822708516
    Training loss =  0.271584002866169  | Validation Loss =  0.5904542357358249
    Training loss =  0.27319966072446844  | Validation Loss =  0.587675993955588
    Training loss =  0.27062512570050323  | Validation Loss =  0.5912886768705231
    Training loss =  0.27327860138389504  | Validation Loss =  0.590419891137111
    Training loss =  0.2716757913821199  | Validation Loss =  0.5895195355646721
    Training loss =  0.27141211303264906  | Validation Loss =  0.582685184252413
    Training loss =  0.2713056575465068  | Validation Loss =  0.5900134139926122
    Training loss =  0.27317383496111697  | Validation Loss =  0.5923789623920425
    Training loss =  0.27196361547273196  | Validation Loss =  0.5913643926256317
    Training loss =  0.27333639578872854  | Validation Loss =  0.5878077178313259
    Training loss =  0.2688597480693225  | Validation Loss =  0.5872210807438139
    Training loss =  0.2686324544836966  | Validation Loss =  0.5892554921439931
    Training loss =  0.2702655356502935  | Validation Loss =  0.5921993190226172
    Training loss =  0.2703063531538074  | Validation Loss =  0.5928381529286944
    Training loss =  0.2701606791508332  | Validation Loss =  0.585827688501857
    Training loss =  0.27085734995898236  | Validation Loss =  0.5906127884921142
    Training loss =  0.2703448727392079  | Validation Loss =  0.5885489051100574
    Training loss =  0.2672055385420831  | Validation Loss =  0.590758235268452
    Training loss =  0.2680742675752452  | Validation Loss =  0.5907769322646821
    Training loss =  0.2672391659004635  | Validation Loss =  0.5863565031737717
    Training loss =  0.2711713607475329  | Validation Loss =  0.5895771950106078
    Training loss =  0.2686547970587618  | Validation Loss =  0.5903512640592921
    Training loss =  0.26762406381495885  | Validation Loss =  0.5951117477085017
    Training loss =  0.27018647816743746  | Validation Loss =  0.5925787396823303
    Training loss =  0.2696176507416066  | Validation Loss =  0.600589727424871
    Training loss =  0.27206024171763593  | Validation Loss =  0.5970340628915698
    Training loss =  0.26920411158143803  | Validation Loss =  0.5917200556787258
    Training loss =  0.2677909886527262  | Validation Loss =  0.5886125514275917
    Training loss =  0.26844470495923184  | Validation Loss =  0.5901587562721993
    Training loss =  0.2679178681266442  | Validation Loss =  0.5814703328197013
    Training loss =  0.2668730401004968  | Validation Loss =  0.5893138825641906
    Training loss =  0.2697604615032003  | Validation Loss =  0.5911531781596977
    Training loss =  0.2690850046327275  | Validation Loss =  0.5897443603865707
    Training loss =  0.2684175796974241  | Validation Loss =  0.5895049671332041
    Training loss =  0.264486731103297  | Validation Loss =  0.5854994342799931
    Training loss =  0.26797777878936757  | Validation Loss =  0.6120158083076719
    Training loss =  0.2679339852811915  | Validation Loss =  0.596162972822471
    Training loss =  0.26780383707432265  | Validation Loss =  0.5901125384030966
    Training loss =  0.26654797269219765  | Validation Loss =  0.5879113786079713
    Training loss =  0.26927928517708616  | Validation Loss =  0.5897225809248188
    Training loss =  0.2665483208184832  | Validation Loss =  0.5913257820193778
    Training loss =  0.2660745262178812  | Validation Loss =  0.5968224931869829
    Training loss =  0.26455583456861836  | Validation Loss =  0.5889586812584712
    Training loss =  0.26543176940150476  | Validation Loss =  0.5883688252686449
    Training loss =  0.2645936238296916  | Validation Loss =  0.5923589635247419
    Training loss =  0.26342807350198877  | Validation Loss =  0.5881988003787109
    Training loss =  0.26474118031812516  | Validation Loss =  0.5886142125612572
    Training loss =  0.26675804679313403  | Validation Loss =  0.5830400387446085
    Training loss =  0.2660082060466991  | Validation Loss =  0.5855354373716604
    Training loss =  0.2652434868256697  | Validation Loss =  0.5936077434553879
    Training loss =  0.2638953010017952  | Validation Loss =  0.5881052122840399
    Training loss =  0.26626495506321446  | Validation Loss =  0.587646411818291
    Training loss =  0.26914278656411705  | Validation Loss =  0.5940016017684454
    Training loss =  0.26746250617872463  | Validation Loss =  0.5961375454055609
    Training loss =  0.2667840766521652  | Validation Loss =  0.5880503825497526
    Training loss =  0.26578171145212787  | Validation Loss =  0.5964185330183697
    Training loss =  0.2648142047142715  | Validation Loss =  0.5923741456074051
    Training loss =  0.26251163771062086  | Validation Loss =  0.5923265377177468
    Training loss =  0.26441843083567834  | Validation Loss =  0.590084698255555
    Training loss =  0.2659400113010674  | Validation Loss =  0.5886158963295981
    Training loss =  0.2634476144285349  | Validation Loss =  0.5964884263805196
    Training loss =  0.26479242603932873  | Validation Loss =  0.5902351627621469
    Training loss =  0.2649312061205339  | Validation Loss =  0.5950964523267143
    Training loss =  0.2644079710576641  | Validation Loss =  0.6004885417499622
    Training loss =  0.2637120432565721  | Validation Loss =  0.5842183203385349
    Training loss =  0.26461833426624204  | Validation Loss =  0.5979752678911394
    Training loss =  0.26514141424820664  | Validation Loss =  0.593499241373207
    Training loss =  0.2645664393399539  | Validation Loss =  0.5973460875483002
    Training loss =  0.26297661343987067  | Validation Loss =  0.594905153357027
    Training loss =  0.2640597188238348  | Validation Loss =  0.595375332651259
    Training loss =  0.2641257808407706  | Validation Loss =  0.59591920164567
    Training loss =  0.2621613069950195  | Validation Loss =  0.5941337065354682
    Training loss =  0.2632402533225799  | Validation Loss =  0.5977622177530442
    Training loss =  0.26423620675387005  | Validation Loss =  0.5945194148564641
    Training loss =  0.2649413981380757  | Validation Loss =  0.5954485812267674
    Training loss =  0.26328195497561035  | Validation Loss =  0.5960468657409088
    Training loss =  0.26403331321276974  | Validation Loss =  0.6004576587475805
    Training loss =  0.26422378491987003  | Validation Loss =  0.5843128604727958
    Training loss =  0.2637040274364225  | Validation Loss =  0.5926025649414787
    Training loss =  0.26495769724584695  | Validation Loss =  0.5926376030414919
    Training loss =  0.2636846720837475  | Validation Loss =  0.5826732689821268
    Training loss =  0.26099139707309477  | Validation Loss =  0.5912960521279508
    Training loss =  0.2614619953746206  | Validation Loss =  0.5871130702113301
    Training loss =  0.2621287391678001  | Validation Loss =  0.5833719601611045
    Training loss =  0.26229041994790014  | Validation Loss =  0.5920748513207656
    Training loss =  0.2620399808197209  | Validation Loss =  0.5934814056263694
    Training loss =  0.263886004560784  | Validation Loss =  0.582186105130594
    Training loss =  0.2657798949126782  | Validation Loss =  0.5992852943607524
    Training loss =  0.2636385910380422  | Validation Loss =  0.5885900062850759
    Training loss =  0.26092734275741525  | Validation Loss =  0.5950622703204175
    Training loss =  0.26324042757408  | Validation Loss =  0.5901103301390314
    Training loss =  0.26417066466607403  | Validation Loss =  0.5876685650036808
    Training loss =  0.2628844967007302  | Validation Loss =  0.5932079730909082
    Training loss =  0.26175200746635374  | Validation Loss =  0.5997284347749461
    Training loss =  0.2616796940565109  | Validation Loss =  0.5863975387082321
    Training loss =  0.2629165934042984  | Validation Loss =  0.5991384117915157
    Training loss =  0.261828050477786  | Validation Loss =  0.5979552170898341
    Training loss =  0.25891188481885397  | Validation Loss =  0.59687854607397
    Training loss =  0.25969862812355665  | Validation Loss =  0.5900782379419994
    Training loss =  0.2601826171275605  | Validation Loss =  0.5892978006274389
    Training loss =  0.2610636840710479  | Validation Loss =  0.5957418061509917
    Training loss =  0.26214363095298243  | Validation Loss =  0.6025438062249356
    Training loss =  0.2634578022394287  | Validation Loss =  0.5906555124475986
    Training loss =  0.26302192885470527  | Validation Loss =  0.5867236064204687
    Training loss =  0.2608078329750661  | Validation Loss =  0.6015911811514746
    Training loss =  0.26014231907182866  | Validation Loss =  0.5967961534669127
    Training loss =  0.26139638778031543  | Validation Loss =  0.5917590586193503
    Training loss =  0.2614646056311184  | Validation Loss =  0.5940432421517271
    Training loss =  0.26070474374913766  | Validation Loss =  0.5950889202612865
    Training loss =  0.26192895578366987  | Validation Loss =  0.5907561094952032
    Training loss =  0.26205133617426574  | Validation Loss =  0.5927074159247966
    Training loss =  0.2612218043526237  | Validation Loss =  0.6028292826710874
    Training loss =  0.26174842195815584  | Validation Loss =  0.5867715552134856
    Training loss =  0.26030265548256004  | Validation Loss =  0.5947935544740298
    Training loss =  0.25667287913684766  | Validation Loss =  0.6001742489730255
    Training loss =  0.2589652601085352  | Validation Loss =  0.6041655467532355
    Training loss =  0.2601936116940185  | Validation Loss =  0.5873224801906554
    Training loss =  0.26167954546347094  | Validation Loss =  0.59660621811066
    Training loss =  0.2624661397565617  | Validation Loss =  0.5938317921594225
    Training loss =  0.26165371400754106  | Validation Loss =  0.5972535903192271
    Training loss =  0.26165434546517524  | Validation Loss =  0.6036980416201339
    Training loss =  0.2600508210866639  | Validation Loss =  0.6036816748637187
    Training loss =  0.26257033608435243  | Validation Loss =  0.607255361125439
    Training loss =  0.2593093404567309  | Validation Loss =  0.5991833524361945
    Training loss =  0.2586679517218236  | Validation Loss =  0.5913903051790809
    Training loss =  0.2599662199114146  | Validation Loss =  0.5945322926034404
    Training loss =  0.2593628515939364  | Validation Loss =  0.5979608506090027
    Training loss =  0.2586225287967853  | Validation Loss =  0.5906342078110337
    Training loss =  0.2599457180483288  | Validation Loss =  0.5921680845540284
    Training loss =  0.26149428820007303  | Validation Loss =  0.5918915096969041
    Training loss =  0.25869517543175247  | Validation Loss =  0.5955584906827548
    Training loss =  0.26024700165464637  | Validation Loss =  0.593134124570758
    Training loss =  0.2628031663512916  | Validation Loss =  0.6071760069720352
    Training loss =  0.2599806884785047  | Validation Loss =  0.6010551006230624
    Training loss =  0.2576486822008417  | Validation Loss =  0.599182254151453
    Training loss =  0.2606015364859211  | Validation Loss =  0.5937180441140123
    Training loss =  0.2588494492883093  | Validation Loss =  0.5980977945438417
    Training loss =  0.26093669273377806  | Validation Loss =  0.6026408357962274
    Training loss =  0.2594451390290528  | Validation Loss =  0.5959384096825676
    Training loss =  0.2598812075938736  | Validation Loss =  0.5969879178306724
    Training loss =  0.25997848235321847  | Validation Loss =  0.6001453341814033
    Training loss =  0.26052161042442484  | Validation Loss =  0.598538566616517
    Training loss =  0.2589036962540632  | Validation Loss =  0.5981501021465672
    Training loss =  0.25910341417354144  | Validation Loss =  0.5976874557980002
    Training loss =  0.25863412118862183  | Validation Loss =  0.6008109711896518
    Training loss =  0.2585811476024349  | Validation Loss =  0.593566665795282
    Training loss =  0.25838446989655495  | Validation Loss =  0.5932458288307432
    Training loss =  0.2610285037270423  | Validation Loss =  0.6021785470755291
    Training loss =  0.2606699239839329  | Validation Loss =  0.5943870109344837
    Training loss =  0.2574440252161428  | Validation Loss =  0.6015513168357093
    Training loss =  0.25868167941657344  | Validation Loss =  0.6039895680634785
    Training loss =  0.25778050745805997  | Validation Loss =  0.5891919739638702
    Training loss =  0.25974630518408304  | Validation Loss =  0.5953685563324876
    Training loss =  0.25929719820786057  | Validation Loss =  0.603069609982052
    Training loss =  0.25877762849578695  | Validation Loss =  0.6018264058521529
    Training loss =  0.2576292793611797  | Validation Loss =  0.5904170267189606
    Training loss =  0.25713252041781887  | Validation Loss =  0.5950258723794156
    Training loss =  0.2610086151555683  | Validation Loss =  0.5946173655332895
    Training loss =  0.2607385154520528  | Validation Loss =  0.5986397554351308
    Training loss =  0.2603811720448933  | Validation Loss =  0.5972663757670278
    Training loss =  0.25727807646721934  | Validation Loss =  0.6052328008639661
    Training loss =  0.2574125479362654  | Validation Loss =  0.5967323392755371
    Training loss =  0.25695914840011785  | Validation Loss =  0.5973362351142908
    Training loss =  0.2606452953781975  | Validation Loss =  0.6071703064290783
    Training loss =  0.2569692235733016  | Validation Loss =  0.6022840618835723
    Training loss =  0.2564857386136323  | Validation Loss =  0.6124807737044644
    Training loss =  0.25788058972593103  | Validation Loss =  0.6035852231053863
    Training loss =  0.25740518349777447  | Validation Loss =  0.5917301633186984
    Training loss =  0.2592227802182851  | Validation Loss =  0.6050858378158843
    Training loss =  0.2555748971827914  | Validation Loss =  0.6035013563522307
    Training loss =  0.25967430850763  | Validation Loss =  0.5975464214001024
    Training loss =  0.25860396312193923  | Validation Loss =  0.5983680363445846
    Training loss =  0.25868949938691066  | Validation Loss =  0.5950495105252487
    Training loss =  0.2570258281455281  | Validation Loss =  0.5993003491862414
    Training loss =  0.2557672347999021  | Validation Loss =  0.603439739359079
    Training loss =  0.2567561139049155  | Validation Loss =  0.5950333284929332
    Training loss =  0.25762317314995137  | Validation Loss =  0.5907337269451045
    Training loss =  0.258414024319709  | Validation Loss =  0.5940217438629408
    Training loss =  0.25808402268069514  | Validation Loss =  0.5914520765910168
    Training loss =  0.2589202966917767  | Validation Loss =  0.5974266991836612
    Training loss =  0.2571470129690813  | Validation Loss =  0.6079113845583759
    Training loss =  0.2550678444879778  | Validation Loss =  0.6051363123871606
    Training loss =  0.2577525931881385  | Validation Loss =  0.6025973269205053
    Training loss =  0.2537921220064163  | Validation Loss =  0.602537620796936
    Training loss =  0.2571620720993267  | Validation Loss =  0.598186393390225
    Training loss =  0.25775560405984355  | Validation Loss =  0.6005797410061591
    Training loss =  0.25764362476347535  | Validation Loss =  0.5964558461547401
    Training loss =  0.25475126869055664  | Validation Loss =  0.5999260564896628
    Training loss =  0.2552823100573896  | Validation Loss =  0.602106270659322
    Training loss =  0.25624087867274714  | Validation Loss =  0.5966799782298285
    Training loss =  0.2535960048102261  | Validation Loss =  0.6074780924913752
    Training loss =  0.2534913760976175  | Validation Loss =  0.6073391848727118
    Training loss =  0.2569921219700508  | Validation Loss =  0.6006263376288273
    Training loss =  0.25783779857198846  | Validation Loss =  0.5984917415093772
    Training loss =  0.256262982787376  | Validation Loss =  0.6022953780894541
    Training loss =  0.25737081752734237  | Validation Loss =  0.6130104235958952
    Training loss =  0.2539951705447074  | Validation Loss =  0.5946851787687857
    Training loss =  0.25498006421695935  | Validation Loss =  0.602816360534998
    Training loss =  0.2567864942751574  | Validation Loss =  0.6013571425077784
    Training loss =  0.2545287284288513  | Validation Loss =  0.6033330293144354
    Training loss =  0.2577422000467777  | Validation Loss =  0.5997310537326185
    Training loss =  0.25439980448213184  | Validation Loss =  0.6082525873234503
    Training loss =  0.25427795410825965  | Validation Loss =  0.5999536347037127
    Training loss =  0.2570792891587434  | Validation Loss =  0.5978052569592552
    Training loss =  0.25511172087339873  | Validation Loss =  0.6037391655555757
    Training loss =  0.25516359051794146  | Validation Loss =  0.5939874822580362
    Training loss =  0.25597116476699205  | Validation Loss =  0.6088510581964179
    Training loss =  0.25822237213508464  | Validation Loss =  0.5987721612433341
    Training loss =  0.2552847456647439  | Validation Loss =  0.6020324782228671
    Training loss =  0.25747390593705555  | Validation Loss =  0.5977985718079257
    Training loss =  0.25292874186226494  | Validation Loss =  0.5980096222982125
    Training loss =  0.25589244214169093  | Validation Loss =  0.6106352105673858
    Training loss =  0.25592226003495494  | Validation Loss =  0.6029686654921825
    Training loss =  0.2536236708251278  | Validation Loss =  0.601555869800632
    Training loss =  0.25634879137525396  | Validation Loss =  0.6004878789312226
    Training loss =  0.2563164195234186  | Validation Loss =  0.5946534967623682
    Training loss =  0.25685763798570366  | Validation Loss =  0.593817834989934
    Training loss =  0.2578468101329348  | Validation Loss =  0.6086051421326424
    Training loss =  0.25655069156141763  | Validation Loss =  0.6028520414597878
    Training loss =  0.2539419336139821  | Validation Loss =  0.6054969050210236
    Training loss =  0.25543724749697727  | Validation Loss =  0.5993760621748896



    
![png](Topological%20Embedding_files/Topological%20Embedding_16_1.png)
    



```python
from umap import UMAP

ANN.eval()

y_test = y_test.drop_duplicates()
X_test = X_test.drop_duplicates()
common_index = [x for x in X_test.index if x in y_test.index]
X_test = X_test.loc[common_index,:]
y_test = y_test.loc[common_index,:]

emb_x, emb_y = ANN(torch.Tensor(X_test.to_numpy()).to(torch.float32),torch.Tensor(y_test.to_numpy()).to(torch.float32))

from sklearn.cluster import KMeans

km = KMeans(20)


emb_umap_y = UMAP().fit_transform(emb_y.detach().numpy())


labels = km.fit_predict(emb_y.detach().numpy())
emb_umap_x = UMAP().fit_transform(emb_x.detach().numpy(),labels)
fir,ax = plt.subplots(ncols=2,figsize=(15,5))

ax[0].scatter(emb_umap_y[:,0],emb_umap_y[:,1],c=labels,cmap="tab20",s=5)
ax[1].scatter(emb_umap_x[:,0],emb_umap_x[:,1],c=labels,cmap="tab20",s=5)
```




    <matplotlib.collections.PathCollection at 0x2a99cf3a0>




    
![png](Topological%20Embedding_files/Topological%20Embedding_17_1.png)
    



```python
torch.save(ANN.state_dict(),"double_embedding_desmartines.pth")
```
