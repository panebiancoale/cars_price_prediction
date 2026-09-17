import os
import streamlit as st
import requests
import pandas as pd


@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.normpath(os.path.join(current_dir, "../../data/car_prices.csv"))
    print(f"Cerco file in : {path_csv}")
    df = pd.read_csv(path_csv)
    return df

df_cars = load_data()

st.set_page_config(
    page_title="Stima il prezzo",
    layout="centered",
)
st.title("Predizione Prezzo Auto 🚗")

elenco_marche = sorted(df_cars['Brand'].unique().tolist())
brand = st.selectbox("Seleziona la Marca", elenco_marche)

modelli_filtrati = df_cars[df_cars['Brand'] == brand]
elenco_modelli = sorted([str(m) for m in modelli_filtrati['Model'].unique() if pd.notna(m)])
model = st.selectbox("Seleziona il Modello", elenco_modelli)

elenco_paesi = df_cars['Country'].unique().tolist()
country = st.selectbox("Seleziona il Paese", elenco_paesi)

opzioni_cambio = {
    "Manual":"Manuale",
    "Automatico":"Automatico",
    "Semi-Automatic":"Semi-Automatico",
}

opzioni_carburante = {
    "Gasoline":"Benzina",
    "Diesel":"Diesel",
    "LGP": "GPL",
    "Eletric": "Elettrica",
    "Hybrid":"Ibrida"
}

opzioni_trazione= {
    "Front": "Anteriore",
    "Rear": "Posteriore",
    "4WD": "Integrale"
}

opzioni_venditore = {
    "Private seller": "Privato",
    "Dealer": "Autorizzato"
}


km = st.number_input("Chilometri", min_value=0)
gearbox = st.selectbox("Cambio", options=list(opzioni_cambio.keys()),format_func=lambda x: opzioni_cambio[x])
year = st.number_input("Anno", min_value=1900, max_value=2026, value=2020)
fuel = st.selectbox("Alimentazione", options=list(opzioni_carburante),format_func=lambda x: opzioni_carburante[x])
power = st.number_input("Potenza (CV)", min_value=1)
seller = st.selectbox("Venditore", options=list(opzioni_venditore),format_func=lambda x: opzioni_venditore[x])
drivetrain = st.selectbox("Trazione", options=list(opzioni_trazione),format_func=lambda x: opzioni_trazione[x])
color = st.text_input("Colore",value="Colore")

if st.button("Calcola Prezzo"):
    payload = {
        "data": [
            {
                "Brand": brand,
                "Model": model,
                "Country": country,
                "Kilometers": float(km),
                "Gearbox": gearbox,
                "Year": int(year),
                "Fuel": fuel,
                "Power": int(power),
                "Seller": seller,
                "Drivetrain": drivetrain,
                "Color": color
            }
        ]
    }

    response = requests.post("http://localhost:8000/predict", json=payload)

    if response.status_code == 200:
        prezzo = response.json()["predictions"][0]
        st.success(f"Il prezzo stimato è: € {prezzo:,.2f}")
    else:
        st.error("Errore nella predizione")
