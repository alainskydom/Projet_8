import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Crédit", layout="centered")
st.title("📊 Dashboard - Décision de crédit")

# Charger les données
#def load_data():
    #df_ = pd.read_csv(r"Streamlit/df_api_1000.csv")
    #df_=df_.loc[:, ~df_.columns.str.match ('Unnamed')]
#df_=load_data()

# 🔁 Récupérer la liste des IDs depuis l'API

st.sidebar.header("Merci de selectionner la demande de crédit:")
try:
    id_response = requests.get("https://projet8-production-31ea.up.railway.app/api/ids") 
    id_response.raise_for_status()
    ids = id_response.json().get("ids", [])
    client_id = st.sidebar.selectbox("Sélectionnez un identifiant client :", ids)
except Exception as e:
    st.error(f"Erreur lors de la récupération des IDs : {e}")
    st.stop()
st.write("Vous avez selectionné la demande n°",  client_id)





if st.button("Obtenir la prédiction via API"):
    url = "https://projet8-production-31ea.up.railway.app/api/predict"

    try:
        response = requests.post(url, json={"id_client": int(client_id)})
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            proba = result["probability"]

            #if prediction == 1:
            if proba > 0.07:
                st.error("❌ Prêt NON accordé")
            else:
                st.success("✅ Prêt accordé")

            st.metric(label="Probabilité de défaut", value=f"{proba*100:.2f} %")


            #st.sidebar.write("*Caractéritiques du client :**", result["features"])
            
            st.sidebar.subheader("🧾 Comparaison client vs moyenne (5 variables clés)")
            df_compare = pd.DataFrame({
                "Valeur client": result["features"],
                "Moyenne globale": result["global_means"]
            })
            st.sidebar.dataframe(df_compare)

            st.sidebar.subheader("📉 Visualisation comparative")
            fig, ax = plt.subplots(figsize=(8, 4))
            df_compare.plot(kind="bar", ax=ax)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.sidebar.pyplot(fig)

            st.subheader("🔍 Interprétation SHAP des variables clés")
            shap_df = pd.DataFrame.from_dict(result["shap_values"], orient="index", columns=["SHAP value"])
            shap_df = shap_df.sort_values("SHAP value", key=abs, ascending=True)

            fig2, ax2 = plt.subplots()
            shap_df.plot(kind="barh", legend=False, ax=ax2)
            ax2.set_title("Impact des variables sur la prédiction")
            plt.tight_layout()
            st.pyplot(fig2)

        else:
            st.warning(f"Erreur API : {response.status_code}")
            st.write(response.json())
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")

# Afficher les graphiques des variables:
 
st.sidebar.header("Plus d'informations")
st.sidebar.subheader("Visualisations univariées")
variables=['CREDIT_TERM','DAYS_BIRTH', "DAYS_EMPLOYED", "AMT_ANNUITY", "CREDIT_INCOME_PERCENT","ANNUITY_INCOME_PERCENT"]
features=st.sidebar.multiselect("les variables clés:", variables)
df_ = pd.read_csv(r"Streamlit/df_api_1000.csv")
 
for feature in features:
         # Set the style of plots
         plt.style.use('fivethirtyeight')
         fig=plt.figure(figsize=(6, 6))
         #if feature=='DAYS_BIRTH':
         # Plot the distribution of feature
         st.write(feature)
         h1=plt.hist(df_[feature], edgecolor = 'k', bins = 25)
         plt.axvline(int(df_[feature][df_.index==client_id]), color="red", linestyle=":")
         plt.title(feature + " distribution", size=5)
         plt.xlabel(feature, size=5)
         plt.ylabel("Nombre d'observations", size=5)
         plt.xticks(size=5)
         plt.yticks(size=5)
         st.pyplot(fig)

st.sidebar.subheader("Analyse bivariées, choisissez deux variables")
df_ = pd.read_csv(r"Streamlit/df_api_1000.csv")
variables_1=['CREDIT_TERM','DAYS_BIRTH', "DAYS_EMPLOYED", "AMT_ANNUITY", "CREDIT_INCOME_PERCENT","ANNUITY_INCOME_PERCENT"]
variables_2=['CREDIT_TERM','DAYS_BIRTH', "DAYS_EMPLOYED", "AMT_ANNUITY", "CREDIT_INCOME_PERCENT","ANNUITY_INCOME_PERCENT"]
features_1=st.sidebar.selectbox("Sélectionnez une première caractéristique :", variables_1)
st.sidebar.write("Vous avez selectionné ", features_1)

features_2=st.sidebar.selectbox("Sélectionnez une deuxième caractéristique :", variables_2)
st.sidebar.write("Vous avez selectionné ", features_2)

# Set the style of plots
plt.style.use('fivethirtyeight')
fig=plt.figure(figsize=(6, 6))

h2=plt.scatter(df_[features_1], df_[features_2], color='blue')
plt.xlabel(features_1)
plt.ylabel(features_2)
plt.title(feature_1, "vs", feature_2)
st.pyplot(fig)

        

