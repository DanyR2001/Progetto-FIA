# RegressorComparator — Progetto FIA

> Comparatore universale di regressori per Machine Learning  
> Esame di Fondamenti di Intelligenza Artificiale  
> Università degli Studi di Salerno · A.A. 2022/2023

---

## Descrizione

Il progetto nasce dall'obiettivo di **rendere accessibile il mondo dei regressori** a chi si avvicina al Machine Learning. Invece di dover testare manualmente ogni algoritmo, `RegressorComparator` automatizza il confronto su un dataset fornito dall'utente, indicando quale regressore ottiene le migliori performance.

---

## Funzionalità

- Caricamento di dataset personalizzati
- Addestramento automatico di multipli regressori in parallelo
- Confronto delle metriche (R², MSE, MAE) su train/test split
- Output tabellare con ranking dei modelli
- Notebook Jupyter con documentazione e visualizzazioni

---

## Regressori supportati

- Linear Regression
- Ridge / Lasso
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR (Support Vector Regression)
- KNN Regressor

---

## Stack

| Layer | Tecnologia |
|---|---|
| Linguaggio | Python 3 |
| ML | scikit-learn |
| Data | pandas, numpy |
| Visualizzazione | matplotlib, seaborn |
| Notebook | Jupyter |

---

## Struttura

```
├── RegressorComparator/   # Codice sorgente del comparatore
├── Documentazione/        # Relazione e documentazione tecnica
├── Presentazione/         # Slide del progetto
└── README.md
```

---

## Come eseguire

> Leggi prima il README nella cartella `./RegressorComparator` per installare le dipendenze.

```bash
cd RegressorComparator
pip install -r requirements.txt
jupyter notebook
```

---

## Autore

**Daniele Russo**  
[LinkedIn](https://www.linkedin.com/in/danielerusso01) · [GitHub](https://github.com/DanyR2001)
