import os

import numpy as np
import pandas as pd

listaDiSostiturioni =dict()
last=dict()


## trasformo il vecchio dataset in una matrice
dataframe=pd.read_csv("./dataset/san-francisco-payroll_2011-2019.csv")
dataframe=dataframe.drop(columns=["Employee Name"])
#Sostituiamo i valori Nan con NotProvided per evitare di avere problemi con il dizionario
print("Number of istance Nan :" + str(dataframe.isna().sum().sum()))

print("Number of istance Nan in tutto il dataset:" + str(dataframe.isna().sum().sum()))
print("Number of istance Nan nella colonna 'Status' :" + str(dataframe["Status"].isna().sum()))
dataframe=dataframe.replace(np.nan,"Not Present",regex=True)
print("Number of istance Nan in tutto il dataset:" + str(dataframe.isna().sum().sum()))
print("Number of istance Nan nella colonna 'Status' :" + str(dataframe["Status"].isna().sum()))

print ("Number of row: "+str(len(dataframe.index)))
print ("Number on 'Not Provided': "+str(dataframe[dataframe == 'Not Provided'].count()))

print ("Number of row before cleaning: "+str(len(dataframe.index)))
dataframe=dataframe.replace("Not Provided",np.nan,regex=True)
print("Number of istance Nan in tutto il dataset before cleaning:" + str(dataframe.isna().sum().sum()))
dataframe=dataframe.dropna()
print("Number of istance Nan in tutto il dataset after cleaning:" + str(dataframe.isna().sum().sum()))
print ("Number of row after cleaning: "+str(len(dataframe.index)))

print(dataframe.info())

data=[]
data.append([header for header in dataframe.columns])
for liste in dataframe.values.tolist():
    data.append(liste)


## inizializzo un dizionario che mi indicarà per ogni categoria l'ultimo l indice inserito
## inizializzo il dizinario che conterra un dizionario per ogni catergoria dove c'è una stringa
for i in range(len(data[0])):
    if type(data[1][i]) is str:
        if not data[1][i].replace(",","",1).replace(".","",1).isnumeric() or not data[1][i].replace(",","",1).replace(".","",1).isdecimal() or not data[1][i].replace(".","",1).replace(",","",1).isdigit() :
            last[data[0][i]] = 0
            listaDiSostiturioni[data[0][i]] = dict()

print(last)
print(listaDiSostiturioni)


for i in range(1,len(data)):
    for j in range(0,len(data[i])):
        if data[0][j] in listaDiSostiturioni.keys():
            if str(data[i][j]).lower() not in listaDiSostiturioni[data[0][j]].keys():
                listaDiSostiturioni[data[0][j]][data[i][j].lower()]=last[data[0][j]]+1
                last[data[0][j]]+=1
            data[i][j]=listaDiSostiturioni[data[0][j]][data[i][j].lower()]
        else:
            if data[i][j]=="Not Provided":
                data[i][j]=0
            else:
                if "," in str(data[i][j]) or "." in str(data[i][j]):
                    data[i][j] = float(data[i][j])
                else:
                    data[i][j] = int(data[i][j])

print(listaDiSostiturioni)

try:
    os.mkdir("../dataset")
except OSError as e:
    pass

newFile=open("../dataset/newDataset.csv","w")

for i in range(0,len(data)):
    newFile.write(str(data[i]).removesuffix("]").removeprefix("[").replace(" ","").replace("'","")+"\n")
newFile.close()


indexSostitution = open("../dataset/indexSostitution.txt","w")

for key in listaDiSostiturioni.keys():
    indexSostitution.write(key+str(listaDiSostiturioni[key])+"\n")
indexSostitution.close()

