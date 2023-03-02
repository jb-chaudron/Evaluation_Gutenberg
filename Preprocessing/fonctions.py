import time
from multiprocessing import Pool
from tqdm import tqdm
import spacy
from collections import Counter
import pandas as pd
import numpy as np

import pandas as pd
from os import listdir
from os.path import isfile, join
from spacy.tokens import DocBin
import textacy
import os


class Gut_text(object):

    def __init__(self, path):

        self.path = path

    def open_clean(self):
        with open(self.path, encoding="latin1", errors="ignore") as f:
            self.texte = f.read()

        try:
            self.titre = str(self.texte).split("Title: ")[1].split("\n")[0]
            self.texte = str(self.texte).split("*** START OF ")[1].split("End of the Project")[0]
            self.texte = [paragraphe for paragraphe in self.texte.split("\n\n")]
            self.bad = 0
            return True

        except Exception :

            self.titre = ""
            self.bad = 1
            return False

    def para2nlp(self):
        nlp = spacy.load("fr_core_news_sm")
        self.doc = list(nlp.pipe(self.texte, n_process=15))

        doc_bin = DocBin()

        for doc in self.doc:
            doc_bin.add(doc)
        path = "/projects/LaboratoireICAR/MACDIT/data/Evaluation/gutenberg_spacy/"+self.titre

        if not os.path.exists(path):
            os.makedirs(path)

        doc_bin.to_disk(path+"/"+self.titre+".spacy")

    def nlp2vec(self):
        spans = [span for span in self.doc]
        self.vect = pd.DataFrame([self.Span2Vec(span) for span in self.doc]).fillna(0)

    def Span2Vec(self,span):
        #l_span  = [token for token in span]
        """
            - Extraction des propriétées du span

                1) Nombre de tokens
                2) Nombre de tokens sans stopwords
                3) Longueur des tokens non stopword

                4) Nombre de phrases
                5) Longueur moyenne des phrases

                6) Diversité token
                7) Diversité Lemmes

                8) POS - DEP
                9) VerbMorph

                10) Suffixes / Préfixes
                11) Morphologie



        """
        vec_out = {"nb tokens" : 0,
                   "nb token no stpword" : 0,
                   "lg token no stpword" : 0,
                   "nb clauses" : 0,
                   "lg clauses" : 0,
                   "diversité token" : 0,
                   "diversité lemmes" : 0}

        non_stop = [token for token in span if not (token.is_space or token.is_punct or token.is_stop)]
        tokens = [token for token in span if not (token.is_space or token.is_punct)]

        phrases = [[tok for tok in sent if not (tok.is_space or tok.is_punct)] for sent in span.sents]
        phrases = [sent for sent in phrases if len(sent) > 0]

        # 1-2-3
        vec_out["nb tokens"] = max(len(tokens),0.01)
        vec_out["nb token no stpword"] = max(len(non_stop),0.01)
        vec_out["lg token no stpword"] = np.mean([len(x) for x in non_stop]) if len(non_stop) != 0 else 0

        # 4-5
        vec_out["nb clauses"] = len(phrases)
        vec_out["lg clauses"] = np.mean([len(sent) for sent in phrases]) if len(phrases) != 0 else 0

        # 6-7
        vec_out["diversité token"] = len(np.unique([tok.text for tok in tokens]))/max(len(tokens),0.01)
        vec_out["diversité lemmes"] = len(np.unique([tok.lemma_ for tok in tokens]))/max(len(tokens),0.01)

        # Fonctions pour récupérer la morphologie
        get_verb = lambda a : ["{} : {}".format(x,a.morph.get(x)) for x in ["VerbForm","Voice","Mood","Person","Tense"]]
        get_morph = lambda a : ["{} : {}".format(x,a.morph.get(x)) for x in ["Gender","Number"] if len(a.morph.get(x)) != 0]

        # 8-9
        prop_tok = [x for token in span for x in ["POS : "+token.pos_,"DEP : "+token.dep_]]
        verb_morph = [x for token in tokens for x in get_verb(token) if token.pos_ == "VERB"]

        tout = prop_tok+verb_morph
        tout = Counter(tout)
        tout["META : titre"] = self.titre

        vec_out.update(tout)

        return vec_out



path = "/projects/LaboratoireICAR/MACDIT/data/Evaluation/gutenberg_fr/"
gutenfiles = [f for f in listdir(path) if isfile(join(path, f))]

if not os.path.exists("/projects/LaboratoireICAR/MACDIT/data/Evaluation/raw_dataframes"):
    os.makedirs("/projects/LaboratoireICAR/MACDIT/data/Evaluation/raw_dataframes")


for file in tqdm(gutenfiles):


    test = Gut_text(path+file)
    good = test.open_clean()

    if good:

        if os.path.exists("/projects/LaboratoireICAR/MACDIT/data/Evaluation/raw_dataframes/"+test.titre+".csv"):
            continue
        else:
            try:
                test.para2nlp()
                test.nlp2vec()
                test.vect.to_csv("/projects/LaboratoireICAR/MACDIT/data/Evaluation/raw_dataframes/"+test.titre+".csv")
            except Exception:
                continue
    else:
        continue
