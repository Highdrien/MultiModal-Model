import csv
import pandas as pd


PATH="data/paco-cheese/transcr/AAOR_merge.csv"

def load_one_csv(path):
    """
    Load one csv file from the given path and convert it to dataframe
    
    """
    data = pd.read_csv(path, na_values=['']) # one speaker name is 'NA'
    
    return data

##
#turn after: est ce que ya changement de speaker apres cet IPU
#turn at start: est ce que ya changement de speaker au debut de cet IPU
#yield: propose a l'autre personne de reprendre la parol

if __name__ == "__main__":
    data = load_one_csv(PATH)
    ##print dataframe
    print(data)
    ##afficher les colonnes
    print(data.columns)
    ## afficher seulement la colonne text et turn after
    print(data[['text', 'turn_after']])

