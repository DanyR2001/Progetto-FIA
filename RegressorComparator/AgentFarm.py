import os
import random

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.utils import parallel_backend
from mlxtend.feature_selection import ExhaustiveFeatureSelector


from Agent import Agent
import seaborn as sns
from scipy.stats import shapiro, kstest, anderson
import statsmodels.api as sm
import matplotlib.pyplot as plt

class AgentFarm:

    listaAgent = None
    var_dip, var_ind = None, None
    indi_training_set, indi_testing_set, dip_training_set, dip_test_set = None, None, None,None
    __agentComparison = ["LinearRegression", "Ridge", "SGDRegressor", "LARS", "LassoLars",
                       "BayesianRidge", "ARDRegression", "TweedieRegressor",
                       "DecisionTreeRegressor", "RandomForestRegressor", "KNeighborsRegressor",
                       "RadiusNeighborsRegressor", "GaussianProcessRegressor", "SVR", "NuSVR", "LinearSVR"]
    __typeNormalization = ["StandardScaler", "MinMaxScaler", "None"]


    def __init__(self, dataframe:pd.DataFrame,n_job:int,target:str,mode:str="Manual",outlier:bool=False,rangeOutlier:float=2.00):
        self.mode=mode
        self.dataframeBackUp=dataframe
        self.dataframe=dataframe
        self.n_job=n_job
        self.target=target

        if outlier:
            self.distriubuzioneFeateure("Before")
            # Calcolare l'IQR per la colonna
            Q1 = self.dataframeBackUp.quantile(0.25)
            Q3 = self.dataframeBackUp.quantile(0.75)
            IQR = Q3 - Q1

            # Applicare il filtro separatamente per ciascuna colonna
            mask = self.dataframeBackUp.apply(
                lambda x: (x >= Q1[x.name] - rangeOutlier * IQR[x.name]) & (x <= Q3[x.name] + rangeOutlier * IQR[x.name]))

            # Applicare la maschera al DataFrame usando il metodo loc
            self.dataframeBackUp = self.dataframeBackUp.loc[mask.all(axis=1)]


            pd.options.mode.copy_on_write = True
            self.dataframe=pd.DataFrame(self.dataframeBackUp)
            self.distriubuzioneFeateure("After")

    def dataCleaning(self,listaRimossi:list,valueTarget=None):
        #eliminiamo le tuple con il peso non presente
        print("Number of istance before cleaning: "+str(self.dataframe.size))
        for lable in listaRimossi:
            self.dataframe.drop(self.dataframe[self.dataframe[lable] == valueTarget].index, inplace=True)
        print("Number of istance after cleaning: "+str(self.dataframe.size))
        print("Number of istance Nan :"+str(self.dataframe.isna().sum().sum()))

        self.dataframe = self.dataframe.reset_index()

    def distriubuzioneFeateure(self,lable:str):

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.dataframe)
        plt.title("Box Plot delle Feature")

        try:
            os.mkdir("./analysis/")
        except OSError as e:
            pass
        try:
            os.mkdir("./analysis/DistribuzioneFeature/")
        except OSError as e:
            pass
        plt.savefig("./analysis/DistribuzioneFeature/"+lable)
        plt.close('all')

    def correlazioneVariabili(self,lable:str):
        # controlliamo la dipendenza tra le variabili
        arr = lable.split("_")
        correlazione = self.dataframe.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlazione, annot=True, cmap='coolwarm', fmt=".2f")
        try:
            os.mkdir("./analysis/")
        except OSError as e:
            pass
        try:
            os.mkdir("./analysis/TabelleDiCorrelazione/")
        except OSError as e:
            pass
        try:
            os.mkdir("./analysis/TabelleDiCorrelazione/" + arr[0])
        except OSError as e:
            pass
        plt.savefig("./analysis/TabelleDiCorrelazione/" + arr[0] + "/correlazione_" + arr[1])
        plt.close('all')

    def featureScaling(self,normalizazione:str,includeTarget:bool=False):
        self.normalizazione=normalizazione

        if not includeTarget:
            var_dip=self.dataframe["Benefits"]
            var_ind=self.dataframe.drop("Benefits",axis=1)

            if self.normalizazione == "MinMaxScaler":
                type = MinMaxScaler().fit_transform(var_ind.values)
                var_ind=pd.DataFrame(type,columns=var_ind.columns,index=var_ind.index,dtype="float32")
            elif self.normalizazione == "StandardScaler":
                type = Normalizer().fit_transform(var_ind.values)
                var_ind = pd.DataFrame(type, columns=var_ind.columns, index=var_ind.index)

            var_ind['Benefits']=var_dip
            self.dataframe=var_ind
        else:
            if self.normalizazione == "MinMaxScaler":
                type = MinMaxScaler().fit_transform(self.dataframe.values)
                self.dataframe=pd.DataFrame(type,columns=self.dataframe.columns,index=self.dataframe.index,dtype="float32")
            elif self.normalizazione == "StandardScaler":
                type = Normalizer().fit_transform(self.dataframe.values)
                self.dataframe = pd.DataFrame(type, columns=self.dataframe.columns, index=self.dataframe.index)

    def featureSelection(self,listaRimossi:list):
        # togliamo le colonne che non ci hanno una bassa varianza
        print("Feature selection, feaure before removing:" + str(list(self.dataframe.columns)))
        self.dataframe = self.dataframe.drop(columns=listaRimossi)
        print("Feature selection, feaure after removing:" + str(list(self.dataframe.columns)))

    def initComparison(self,percentageTest:float):
        # identifichiamo variabile dipendente e indipendeti
        self.var_dip = self.dataframe[self.target]
        self.var_ind = self.dataframe.drop(columns=[self.target])
        # dividiamo tra train e test
        self.indi_training_set, self.indi_testing_set, self.dip_training_set, self.dip_test_set = train_test_split(
            self.var_ind, self.var_dip,
            test_size=percentageTest,
            random_state=random.randint(
                0, 256), shuffle=True)

    def startComparison(self):
        for NameAgent in self.__agentComparison:
            self.singleComparison(NameAgent)

    def singleComparison(self,NameAgent:str):
        agente = Agent(NameAgent, self.n_job)
        agente.fit(self.indi_training_set, self.dip_training_set)
        prediction = agente.predict(self.indi_testing_set)

        try:
            os.mkdir("./analysis/" + agente.name)
            os.mkdir("./analysis/" + agente.name + "/" + self.normalizazione)
        except OSError as error:
            pass

        try:
            os.mkdir("./analysis/" + agente.name + "/" + self.normalizazione)
        except OSError as error:
            pass

        report = open("./analysis/" + agente.name + "/" + self.normalizazione + "/reportMetrics.txt", "w")

        if np.any(np.isnan(prediction)) or np.all(np.isnan(prediction)):
            report.write(
                agente.name + ": One or more samples have no neighbors within specified radius; predicting NaN.\n")
            report.close()
        else:
            ## effetueremo delle analisi
            errori_residui = self.dip_test_set - prediction
            # Calcolare l'errori residuo e mostrare un istogramma per vedere se hanno una distribuzione normale
            plt.xlabel('Errori residui')
            plt.ylabel('Densità degli errori')
            plt.title('Grafico distribuzione degli errori')
            sns.histplot(errori_residui, kde=True)
            plt.savefig("./analysis/" + agente.name + "/" + self.normalizazione + "/distribuzioneErroreResiduo")
            plt.close('all')

            # capire se c'è un Normalità dei residui con - TEST Shapiro-Wilk o il test di Kolmogorov-Smirnov
            # Test di Shapiro-Wilk
            report.write("Normalità del errore residuo: - TEST Shapiro-Wilk e Kolmogorov-Smirnov -\n")
            shapiro_test_stat, shapiro_p_value = shapiro(errori_residui)
            # print(f"Test di Shapiro-Wilk: Statistica={shapiro_test_stat}, p-value={shapiro_p_value}")
            report.write(f" Test di Shapiro-Wilk: Statistica={shapiro_test_stat}, p-value={shapiro_p_value}.\n")
            if shapiro_test_stat > 0.05:
                report.write(
                    "      Non ci sono prove a sufficenza per rifiutare l'ipotesi che seguano una distribuzione normale.\n")
            else:
                report.write(
                    "      Ci sono prove a sufficenza per rifiutare l'ipotesi che seguano una distribuzione normale.\n")
            # Test di Kolmogorov-Smirnov
            ks_test_stat, ks_p_value = kstest(errori_residui, 'norm')
            # print(f"Test di Kolmogorov-Smirnov: Statistica={ks_test_stat}, p-value={ks_p_value}")
            report.write(f" Test di Kolmogorov-Smirnov: Statistica={ks_test_stat}, p-value={ks_p_value}.\n")
            if ks_test_stat > 0.05:
                report.write(
                    "      Non ci sono prove a sufficenza per rifiutare l'ipotesi che seguano una distribuzione normale.\n")
            else:
                report.write(
                    "      Ci sono prove a sufficenza per rifiutare l'ipotesi che seguano una distribuzione normale.\n")

            statistica, valori_critici, livelli_di_significativita = anderson(errori_residui, dist='norm')

            report.write(f" Test di Anderson-Darling: \n")
            # Confronta la statistica del test con i valori critici
            for i, livello_di_significativita in enumerate(livelli_di_significativita):
                if statistica > valori_critici[i]:
                    report.write("      Livello di significatività:" + str(
                        livello_di_significativita) + "- Non ci sono evidenze sufficienti per dire che la distribuzione sia non normale.\n")
                else:
                    report.write("      Livello di significatività:" + str(
                        livello_di_significativita) + "- La distribuzione non è normale.\n")

            # Indipendenza degli errori - test DURBIN-WATSON
            report.write("Indipendeza degli errori residui - test DURBIN-WATSON -\n")
            durbin_watson_statistic = sm.stats.stattools.durbin_watson(errori_residui)
            # print(f"Test di Durbin-Watson: Statistica={durbin_watson_statistic},")
            report.write(f" Test di Durbin-Watson: Statistica={durbin_watson_statistic}\n")
            if 1.8 < durbin_watson_statistic < 2.3:
                report.write("      Non c'è evidenza di autocorrelazione significativa\n")
            elif durbin_watson_statistic >= 2.3:
                report.write("      Potrebbe esseci una autocorrelazione negartiva\n")
            elif durbin_watson_statistic <= 1.8:
                report.write("      Potrebbe esseci una autocorrelazione positiva\n")

            # Normalizzare gli errori residui calcolado lo z-score di ciascun errore residuo. Lo z-score misura quante deviazioni standard un dato residuo è al di sopra o al di sotto della media degli errori residui.
            media_residui = np.mean(errori_residui)
            deviazione_standard_residui = np.std(errori_residui)
            z_scores = (errori_residui - media_residui) / deviazione_standard_residui

            # per verificare se i dati hanno una varianza costante (omoschedasticità)
            plt.scatter(prediction, z_scores)
            plt.axhline(0, color='red', linestyle='dashed', linewidth=2)
            plt.xlabel('Valori Previsti')
            plt.ylabel('Errori Residui standardizzati')
            plt.title('Grafico dei Residui standardizzati vs. Valori Previsti')
            plt.savefig("./analysis/" + agente.name + "/" + self.normalizazione + "/varianzaErroreResiduo")
            plt.close('all')

            #print(prediction)

            variance_score, mean_absolute, mean_squared, root_mean_absolute, r2 = agente.valuation(self.dip_test_set,
                                                                                                   prediction)
            report.write("Metrics:\n")
            report.write(
                f"    Variance_score:{'%.2f' % (variance_score)}\n" + f"    Mean absolute error:{'%.2f' % (mean_absolute)}\n" + f"    Mean squared error:{'%.2f' % (mean_squared)}\n" + f"    Root mean absolute error:{'%.2f' % (root_mean_absolute)}\n" + f"    R^2:{'%.2f' % (r2)}\n")
            # print(agente.name+f" Variance_score:{'%.2f' % (variance_score)}\n"+agente.name+f" mean squared error:{'%.2f' % (mean_squared)}\n"+agente.name+f" mean absolute error:{'%.2f' % (mean_absolute)}\n"+agente.name+f"R^2:{'%.2f' %(r2)}\n")
            fit_time_mean, score_time_mean, absolute_error_mean, squared_error_mean, root_mean_squared_error_mean, r2_mean = agente.cross_validation(
                X_train=self.indi_training_set, y_train=self.dip_training_set)
            # print("Cross validatio means:\n")
            # print(agente.name+f"Fit time mean:{fit_time_mean}\n"+agente.name+f"Score time mean:{score_time_mean}\n"+agente.name+f" MAE test mean: {absolute_error_mean}\n"+agente.name+f" MSE test mean: {squared_error_mean}\n"+agente.name+f" RMSE test mean:{root_mean_squared_error_mean}\n"+agente.name+f" R^2 test mean:{r2_mean}")

            # scrittura del Reaport
            report.write("Metrics - Kcross validation - :\n")
            report.write(
                f"    Fit time mean:{fit_time_mean}\n" + f"    Score time mean:{score_time_mean}\n" + f"    MAE test mean: {absolute_error_mean}\n" + f"    MSE test mean: {squared_error_mean}\n" + f"    RMSE test mean:{root_mean_squared_error_mean}\n" + f"    R^2 test mean:{r2_mean}")

    def autoDataPreparation(self,nameAgent:str):
        regressor = Agent(nameAgent,self.n_job)

        #dividiamo in dip e ind
        var_dip = self.dataframe[self.target]
        var_ind = self.dataframe.drop(self.target, axis=1)


        # Crea un selettore di feature esaustivo
        #print(sklearn.metrics.get_scorer_names())
        efs = ExhaustiveFeatureSelector(regressor.model, min_features=1, max_features=var_ind.shape[1],
                                        scoring='r2', cv=5,n_jobs=self.n_job)

        # Esegui la ricerca esaustiva delle feature
        efs = efs.fit(var_ind, var_dip)

        # Visualizza le feature selezionate
        selected_features = var_ind.columns[list(efs.best_idx_)]
        tutte_le_feature = set(var_ind.columns)
        feature_selezionate = set(selected_features)
        return list(tutte_le_feature - feature_selezionate), list(feature_selezionate)

    def autoCompariosn(self,typeNorm:str,percentageTest:float):
        for NameAgent in self.__agentComparison:
            self.dataframe = self.dataframeBackUp
            self.featureScaling(typeNorm,True)
            self.featureSelection(self.listaRimossi)
            self.correlazioneVariabili(typeNorm + "_before")
            listaRimossi,listaSelezionati = self.autoDataPreparation(NameAgent)
            self.dataCleaning(listaSelezionati, 0)
            print("Feature selezionate automaticamente: "+str(listaSelezionati))
            print("Feature eliminate automaticamente: "+str(listaRimossi))
            self.featureSelection(listaRimossi)
            self.correlazioneVariabili(typeNorm + "_after")
            self.initComparison(percentageTest)
            with parallel_backend('threading', n_jobs=self.n_job):
                self.singleComparison(NameAgent)

    def start(self,listaRimossi:list,listaCleaning:list,percentageTest:float):
        self.listaRimossi = listaRimossi
        self.listaCleaning = listaCleaning
        for norm in self.__typeNormalization:
            print("Normalizin: " + norm)
            if self.mode == "Manual":
                self.dataframe = self.dataframeBackUp
                self.dataCleaning(listaCleaning,0)
                self.correlazioneVariabili(norm + "_before")
                self.featureScaling(norm,False)
                self.featureSelection(self.listaRimossi)
                self.correlazioneVariabili(norm + "_after")
                self.initComparison(percentageTest)
                print(self.dataframe)
                with parallel_backend('threading', n_jobs=self.n_job):
                    self.startComparison()
            elif self.mode == "Auto":
                self.autoCompariosn(norm,percentageTest)


