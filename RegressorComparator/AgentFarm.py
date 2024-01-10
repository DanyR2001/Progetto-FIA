import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer

from Agent import Agent
import seaborn as sns
from scipy.stats import shapiro, kstest, anderson
import statsmodels.api as sm
import matplotlib.pyplot as plt

class AgentFarm:

    listaAgent = None
    var_dip, var_ind = None, None
    indi_training_set, indi_testing_set, dip_training_set, dip_test_set = None, None, None,None


    def __init__(self, dataframe:pd.DataFrame,n_job:int):
        self.dataframe=dataframe
        self.n_job=n_job


    def dataCleaning(self,listaRimossi:list):
        #eliminiamo le tuple con il peso non presente
        print("Number of istance before cleaning: "+str(self.dataframe.size))
        for lable in listaRimossi:
            self.dataframe.drop(self.dataframe[self.dataframe[lable] == 0].index, inplace=True)
        print("Number of istance after cleaning: "+str(self.dataframe.size))
        print("Number of istance Nan :"+str(self.dataframe.isna().sum().sum()))
        self.dataframe = self.dataframe.reset_index()


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

    def featureScaling(self,normalizazione:str):
        self.normalizazione=normalizazione
        if self.normalizazione == "MinMaxScaler":
            normal = MinMaxScaler().fit_transform(self.dataframe.values)
            self.dataframe=pd.DataFrame(normal,columns=self.dataframe.columns,index=self.dataframe.index,dtype="float32")
        elif self.normalizazione == "StandardScaler":
            normal = Normalizer().fit_transform(self.dataframe.values)
            self.dataframe = pd.DataFrame(normal, columns=self.dataframe.columns, index=self.dataframe.index)

    def featureSelection(self,listaRimossi:list,percentageTest:float):
        # togliamo le colonne che non ci hanno una bassa varianza
        self.dataframe = self.dataframe.drop(columns=listaRimossi)
        # identifichiamo variabile dipendente e indipendeti
        self.var_dip = self.dataframe["Benefits"]
        self.var_ind = self.dataframe.drop(columns=["Benefits"])
        # dividiamo tra train e test
        self.indi_training_set, self.indi_testing_set, self.dip_training_set, self.dip_test_set = train_test_split(self.var_ind, self.var_dip,
                                                                                               test_size=percentageTest,
                                                                                               random_state=random.randint(
                                                                                                   0, 256),shuffle=True)


    def startComparison(self,listaAgent:list):
        for NameAgent in listaAgent:
            agente = Agent(NameAgent,self.n_job)
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
                report.write(agente.name+": One or more samples have no neighbors within specified radius; predicting NaN.\n")
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

                print(prediction)

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
