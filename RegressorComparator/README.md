# Progetto-FIA
Prima di eseguire il comparatore di regressori, bisogna verificare di aver risolto tutte le dipendenze, ovvero:
• Pandas:
  Puoi risolvere con: pip install pandas
• Joblib:
  Puoi risolvere con: pip install joblib
• Scikit-learn:
  Puoi risolvere con: pip install scikit-learn
• Matplotlib:
  Puoi risolvere con: pip install matplotlib
• Seaborn:
  Puoi risolvere con: pip install seaborn
• Statsmodels:
  Puoi risolvere con: pip install statsmodels
Per verificare il corretto funzionamento del progetto, è necessario solo avviare main.py; le altre classi sono state utilizzare per strutturate il progetto e consentire una corretta comprensione e lettura del codice.


Qual'ora si volesse utilizzare con un dataset diverso consiglio la lettura della documentazione; e per una maggiore comprensione del programma può essere esplicativo capire come far prendere il dataset al progetto.
Quando abbiamo un dataset possiamo trovarci davanti a 2 casi:
• Dataset numerico:
  In questo caso, non dovremmo avere problemi, e possiamo inserire il dataset nella cartella ./dataset; ovviamente andranno modificate tutti i nomi che si riferiscono alle colonne del dataset utilizzata attualmente, questo sia nel main, che nel file AgentFarm.py.
• Dataset alfanumerico:
  In questo caso il il dataset andrà inserito nella cartella ./util/dataset e andrà avvito il file normalizer.py. Il programma in automatico creerà nella cartella un dataset pulito in ./dataset con il relativo indice delle sostituzioni. Andranno fatte tutte le modifiche ai nomi delle colonne sia nel main, che nel file AgentFarm.py.

Infine possiamo eseguire il main.py
