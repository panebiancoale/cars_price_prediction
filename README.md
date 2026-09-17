# Car Price Prediction
Sistema predittivo del prezzo di un veicolo utilizzando un modello di machine learning, in questo caso della famiglia XGB.
Il progetto affronta l'intero ciclo di vita del dato: dalla pulizia, al feature engineering seppur basico, fino allo sviluppo di interfaccia grafica con l'uso di Streamlit per la visualizzazione e interazione con il sistema.

---
## Il Dataset
Viene utilizzato un dataset presente su Kaggle, qui viene presentata solo una versione leggera del file in data/sample_data.csv, la versione completa è recuperabile al seguente link: https://www.kaggle.com/datasets/alemazz11/europe-car-prices/data

---

## Pipeline del Progetto
1. **Data Cleaning:**
   * Gestione corretta dei valori non compatibili o mancanti attraverso la fase di preprocessing.
2. **Feature Engineering:**
   * Creazione di 'Old_Year' (età del veicolo) partendo dall'anno di immatricolazione
   * Encoding delle variabili categoriche utilizzando Target Encoding
3. **Tuning:**
   * Ottimizzazione degli iperparametri per massimizzare la precisione del modello
  
--- 

## Risultati e Performance
Il modello finale basato su **XGBoost** ha ottenuto i seguenti risultati sul test set:

| Metrica | Valore | Spiegazione
| :------ | :----- | :----------
| **RMSLE** | 0.2306 | Errore logaritmico medio. Penalizza gli errori in percentuale anzichè in valore assoluto, in questo caso il range è 23-24%, che risulta abbastanza buono per il tipo di mercato dell'usato.

---

## Come Eseguire il Progetto In Locale

### 1. Clona la repository

### 2. Configura l'ambiente virtuale ed installa le dipendenze necessarie

### 3. Scarica il dataset completo da kaggle e assicurati che sia presente in data/

### 4. Avvia il server FastApi e poi l'interfaccia Streamlit

---

## Struttura della Repository 
```text
|--- data/                       #Contiene il file sample_data.csv di esempio
|--- models/                     # Contiene i modello di training e il preprocessor
|--- notebooks/                  # Jupyter Notebooks utilizzati per fase di analisi esplorativa
|--- src/
     |--- api/                   # Modello per richiesta http
     |--- gui/                     # Streamlit per interfaccia grafica
     |--- config_parma_model.yaml  # File di configurazione per gli iperparametri del modello
     |--- data_preprocessing.py    # Gestione encoding e feature engeeniring
     |--- model.py                 # Contiene il file necessario al Modello di ML XGBoost
     |--- test.http                # File di test per chiamata POST
|--- main.py                       # Punto di avvio dell'applicazione FastAPI
|--- requirements.txt              # Dipendenze necessarie per l'ambiente virtuale
|--- train_model.py                # Addestramento e salvataggio modello
```
---
