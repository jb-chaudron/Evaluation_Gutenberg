import re
import numpy as np
import pandas as pd
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