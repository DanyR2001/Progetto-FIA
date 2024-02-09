# main.py
Prima di eseguire il comparatore di regressori, bisogna verificare di aver risolto tutte le dipendenze, ovvero:
- Pandas: Puoi risolvere con: pip install pandas
- Joblib: Puoi risolvere con: pip install joblib
- Scikit-learn: Puoi risolvere con: pip install scikit-learn
- Matplotlib: Puoi risolvere con: pip install matplotlib
- Seaborn: Puoi risolvere con: pip install seaborn
- Statsmodels: Puoi risolvere con: pip install statsmodels
- appJar: Puoi risolvere con: pip install appjar
- PIL: Puoi risolvere con: pip install PIL
- Scipy: Puoi risolvere con: pip install scipy
- Mlxtend: Puoi risolvere con: pip install mlxtend


Per verificare il corretto funzionamento del progetto, è necessario solo avviare main.py; le altre classi sono state utilizzare per strutturate il progetto e consentire una corretta comprensione e lettura del codice.
Qual'ora si volesse utilizzare con un dataset diverso consiglio la lettura della documentazione; e per una maggiore comprensione del programma può essere esplicativo capire come far prendere il dataset al progetto. Quando abbiamo un dataset possiamo trovarci davanti a 2 casi:
- Dataset numerico: In questo caso, non dovremmo avere problemi, e possiamo inserire il dataset nella cartella ./dataset; ovviamente andranno modificate tutti i nomi che si riferiscono alle colonne del dataset utilizzati attualmente, questo sia nel main.py, che nel file AgentFarm.py.
- Dataset alfanumerico: In questo caso il il dataset andrà inserito nella cartella ./util/dataset e andrà avvito il file normalizer.py. Il programma in automatico creerà nella cartella ./dataset un dataset pulito con il relativo indice delle sostituzioni. Andranno fatte tutte le modifiche ai nomi delle colonne sia nel main.py, che nel file AgentFarm.py.
Inoltre l'AgentFarm permentte di essere utilizzata in 2 modalità distrinte:
-Automatica: Il modulo prevederà ad effettuare in automatico le fasi di:
  * Pulizia Outlier: Questo avverrà rimuovendo le tuple presenti nel primo e quarto quartile,
  in quanto non costituiscono informazioni rilevanti, ma bensì confondono il modello.
  * Feature Scaling: Per ogni feature andiamo a effettuare l’analisi dei curti oltre a valutare la sua distribuzione, per capire se applicare o meno una normalizzazione.
  * Feature selection: Attraverlo l’utilizzo del metodo ExhaustiveFeatureSelector andiamo a prendere le feature che per un determinato modello aumentano la metrica r2.
  * Data cleaning: Sulle feature restituite da ExhaustiveFeatureSelector andiamo a prendere solo le tuple con valori non uguali a zero.
- Manuale: Qui verranno fatte tutte le fasi di datacleaning, featurescaling, featureSelection seguendo le indicazioni del utente.

Adesso possiamo eseguire il main.py. Alla fine dell'esecuzione il programma avrà creato la cartella ./analysis nella quale possiamo trovare i vari grafici e score per ogni algoritmo, anche con diverse normalizzazioni applicate sui dati.
