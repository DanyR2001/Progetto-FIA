# main.py
Prima di eseguire il comparatore di regressori, bisogna verificare di aver risolto tutte le dipendenze, ovvero:
- Pandas: Puoi risolvere con: pip install pandas
- Joblib: Puoi risolvere con: pip install joblib
- Scikit-learn: Puoi risolvere con: pip install scikit-learn
- Matplotlib: Puoi risolvere con: pip install matplotlib
- Numpy: Puoi risolvere con: pip install numpy
- Seaborn: Puoi risolvere con: pip install seaborn
- Statsmodels: Puoi risolvere con: pip install statsmodels
- appJar: Puoi risolvere con: pip install appjar
- PIL: Puoi risolvere con: pip install PIL
- Scipy: Puoi risolvere con: pip install scipy
- Mlxtend: Puoi risolvere con: pip install mlxtend

Per verificare il corretto funzionamento del progetto, è necessario solo avviare main.py; le altre classi sono state utilizzare per strutturate il progetto e consentire una corretta comprensione e lettura del codice.


Qualora si volesse utilizzare con un dataset diverso, consiglio la lettura della documentazione; per una maggiore comprensione del programma può essere esplicativo capire come far prendere il dataset al progetto. Quando abbiamo un dataset possiamo trovarci davanti a 2 casi:
- Dataset numerico: In questo caso, non dovremmo avere problemi, e possiamo inserire il dataset nella cartella ./dataset; ovviamente andranno modificate tutti i nomi che si riferiscono alle colonne del dataset utilizzati attualmente, questo sia nel main.py, che nel file AgentFarm.py.
- Dataset alfanumerico: In questo caso il dataset andrà inserito nella cartella ./util/dataset e andrà avviato il file normalizer.py. Il programma in automatico creerà nella cartella ./dataset un dataset pulito con il relativo indice delle sostituzioni. Andranno fatte tutte le modifiche ai nomi delle colonne sia nel main.py, che nel file AgentFarm.py.

Adesso possiamo eseguire il main.py. Alla fine dell'esecuzione il programma avrà creato la cartella ./analysis nella quale possiamo trovare i vari grafici e score per ogni algoritmo, anche con diverse normalizzazioni applicate sui dati.
Di seguito avvierà la GUI per consentire una facile comparazione dei vari algortimi.

Qualora si volesse comprendere il funzionamento anche degli altri moduli, per quest'ultimi la loro spiegazione è stata approfondita nella documentazione.
