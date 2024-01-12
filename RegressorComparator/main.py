import os

import pandas as pd
from sklearn.utils import parallel_backend


from AgentFarm import AgentFarm

path="./dataset"
file_name="newDataset.csv"


#prepariamo il data frame da pandas
dataframe=pd.read_csv(path+"/"+file_name)
#print(dataframe.info(memory_usage='deep'))
print(len(dataframe.index))
dataframe = dataframe.sample(frac=0.02, random_state=42)
print(len(dataframe.index))

agentComparison = ["LinearRegression","Ridge","SGDRegressor","LARS","LassoLars",
                   "BayesianRidge","ARDRegression","TweedieRegressor",
                   "DecisionTreeRegressor","RandomForestRegressor","KNeighborsRegressor",
                   "RadiusNeighborsRegressor","GaussianProcessRegressor","SVR","NuSVR","LinearSVR"]
typeNormalization = ["StandardScaler","MinMaxScaler","None"]
listaRimossi=list({"TotalPay","TotalPay&Benefits"})
listaCleaning=["BasePay","Benefits"]
n_job=8


for norm in typeNormalization:
    print("Normalizin: "+norm)
    farm=AgentFarm(dataframe,n_job)
    farm.dataCleaning(listaCleaning)
    farm.correlazioneVariabili(norm+"_before")
    farm.featureScaling(norm)
    farm.featureSelection(listaRimossi,0.50)
    farm.correlazioneVariabili(norm+"_after")
    with parallel_backend('threading', n_jobs=n_job):
        farm.startComparison(agentComparison)

os.system("python3 gui.py")



