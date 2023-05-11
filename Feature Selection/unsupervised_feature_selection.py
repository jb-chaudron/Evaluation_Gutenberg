from scipy.stats import entropy
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler


def get_columns_entropy(col_in,n_bins=10):
    un, count =  np.unique(pd.cut(col_in,n_bins, duplicates="drop",labels=list(range(n_bins))),return_counts=True)
    return entropy(count/sum(count))
#entropy(pd.cut(X[:,1],10,duplicates="drop",labels=list(range(10))))

def entropy_reduction(data_in,entropy_cut,n_dim_min=100, n_dim_max=1_000,n_bins=10):
    pipe = Pipeline([("scaler",RobustScaler()),
                    ("feature_selection",VarianceThreshold(0.3))])
    X = pipe.fit_transform(data_in)

    out = {name : get_columns_entropy(X[:,i],n_bins) for i,name in enumerate(pipe["feature_selection"].get_feature_names_out(list(data_in.columns)))}
    name_out = sorted(list(out.keys()), key=lambda a : out[a], reverse=True)

    out = {name : ent for name, ent in out.items() if ent > entropy_cut}
    if len(out) < n_dim_min:
        return name_out[:n_dim_min]
    elif len(out) > n_dim_max:
        return name_out[:n_dim_max]
    else:
        return list(out.keys())
    
    import networkx as nx

def graph_correlation(data_in,corr_thresh):
    G = nx.Graph()
    G.add_nodes_from(list(data_in.columns))
    corr_mat = data_in.corr().to_numpy()
    for dim, name_1 in zip(corr_mat,data_in.columns):
        for d, name_2 in zip(dim, data_in.columns):
            if ((name_1 != name_2) and (d > corr_thresh)):
                G.add_edge(name_1,name_2)
    return G

from sklearn.decomposition import PCA


def pca_reduction(data_in, graph_in):
    out = []
    for component in nx.connected_components(graph_in):
        if len(component) == 1:
            out += [data_in.loc[:,list(component)[0]].to_numpy().reshape((-1))]
        else:
            pca = PCA(1)
            dim = pca.fit_transform(data_in.loc[:,list(component)])
            out += [dim.reshape((-1))]
    return np.array(out)


from tqdm import tqdm 

def reduction_all(data_in, entropy_thresh, corr_thresh, y):
    entropy_range = np.linspace(0,5,20)
    corr_range = np.linspace(0,1,20)

    df_corr = pd.DataFrame(0,
                            index = entropy_range,
                            columns = corr_range)
    df_pval = pd.DataFrame(0,
                            index=entropy_range,
                            columns = corr_range)
    for ent, cor in tqdm(itr.product(entropy_range,corr_range)):
        selected = entropy_reduction(data_in, ent)
        graph = graph_correlation(data_in.loc[:,selected],cor)
        new_val = pca_reduction(data_in.loc[:,selected],graph)
        new_val = new_val.T

        X = pdist(new_val)
        
        score = spearmanr(X,y)
        correlation = score.correlation
        pvalue = score.pvalue

        df_corr.loc[ent,cor] = correlation
        df_pval.loc[ent,cor] = pvalue 

    return df_corr, df_pval