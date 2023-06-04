import pandas as pd
import os
#llegir el tracks df
# aquest csv conte informacio sobre arxius que no tenim en la versio small a mes de 
# molta informacio inecesaria
print("carregant dades")
tracks= pd.read_csv("fma/tracks.csv",low_memory=False)
#nomes volem les dos columnes
df=tracks[[	"Unnamed: 0","track.7"]]
#canvis per tal de arreglar els noms de les columnes
df=df.iloc[2:,:]
df=df.set_axis(["track_id","genre"],axis=1)
#ens carreguem els nans per evitar problemes(cap)
df.dropna(axis=0,inplace=True)
#agafem el checksum perque justament conte carpeta/arxiu per a tots els arxius i de manera ordenada
with open("fma/fma_small/checksums") as sumes:
    data=sumes.read()
    partida=data.split()

print("generant dataframe")
llista_ids=[]
llista_arxius=[]
for i,arxiu in enumerate(partida):
    if i%2==1:#en el chacksum el carpeta/arxiu esta a la segona columna
        id = arxiu.split("/")[1].split(".")[0]
        llista_ids.append(int(id))
        llista_arxius.append(arxiu)


dict_df={}
#ara fem un dataframe que nomes contingui id i genere dels arxius al small
for index,row in df.iterrows():
    if int(row[0]) in llista_ids:
        dict_df[index]=row
#estem basicament reduint el tamany del dataframe
df_small=pd.DataFrame(data=dict_df).T

df_small.set_index("track_id",inplace=True)#canvi per poder accedir a traves del id posteriorment
df_small.to_csv("fma/tracks_small.csv")

dict_genres={'Hip-Hop':[], 'Pop':[], 'Folk':[]
             , 'Experimental':[], 'Rock':[], 'International':[]
             ,'Electronic':[], 'Instrumental':[]}
#generem un diccionari amb clau genere i llista de ids
for index,row in df_small.iterrows():
    dict_genres[row[0]].append(index)
print("separant train i test")
ids_train=[]
ids_test=[]
ids_validation=[]
for genere in dict_genres:
    ids_train.extend(dict_genres[genere][:800]) 
    ids_test.extend(dict_genres[genere][800:900])
    ids_validation.extend(dict_genres[genere][900:])

arxius_train=[]
arxius_test=[]
arxius_validation=[]
for arxiu in llista_arxius:
    id = arxiu.split("/")[1].split(".")[0]
    int(id)
    if id in ids_train:
        arxius_train.append(arxiu)
    elif id in ids_test:
        arxius_test.append(arxiu)
    elif id in ids_validation:
        arxius_validation.append(arxiu)
print("guardant arxius")
import json
with open("fma/arxius_validation.txt", "w") as fp:
    json.dump(arxius_validation, fp)
with open("fma/arxius_test.txt", "w") as fp:
    json.dump(arxius_test, fp)
with open("fma/arxius_train.txt", "w") as fp:
    json.dump(arxius_train, fp)