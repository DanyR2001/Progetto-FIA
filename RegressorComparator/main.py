import os

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from AgentFarm import AgentFarm

path="./dataset"
file_name="newDataset.csv"


#prepariamo il data frame da pandas
dataframe=pd.read_csv(path+"/"+file_name)
#print(dataframe.info(memory_usage='deep'))
print("Numero di istanze prima dell'partizionamento :"+str(len(dataframe.index)))
dataframe = dataframe.sample(frac=0.02, random_state=42)
print("Numero di istanze dopo dell'partizionamento :"+str(len(dataframe.index)))



listaRimossi=list({"TotalPay","TotalPay&Benefits"})
listaCleaning=["BasePay","Benefits"]
n_job=8

farm = AgentFarm(dataframe,n_job,"Benefits",mode="Manual",outlier=True)
farm.start(listaRimossi,listaCleaning,0.33)


os.system("python3 gui.py")



