import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="Dashboard Crédit", layout="centered")
st.title("📊 Dashboard - Décision de crédit")
URL_API = "https://projet8-production-31ea.up.railway.app/api/"

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

@st.cache()
def load_features():
    # Requête permettant de récupérer la liste des features
    data_json = requests.get("https://projet8-production-31ea.up.railway.app/api/load_features")
    data = data_json.json()
      # Récupération des valeurs sans les [] de la réponse
    lst_id = []
    for i in data:
        lst_id.append(i)
    return lst_id

@st.cache()
def load_feature_importance():
    # Requête permettant de récupérer la liste des features importance
    data_json = requests.get("https://projet8-production-31ea.up.railway.app/api/load_feature_importance")
    data = data_json.json()
      # Récupération des valeurs sans les [] de la réponse
    lst_id = []
    for i in data:
        lst_id.append(i)
    return lst_id

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
plt.title("Analyse bivariée")
st.pyplot(fig)

# Création d'un pie-chart interactif avec Plotly


# Création des classes
df_["Classe"] = pd.cut(df_["ANNUITY_INCOME_PERCENT"], bins=[0, 0.25, 0.5, 0.75, 1], labels=["Classe 1 (<0.25)", "Classe 2 (<0.50)","Classe 3 (<0.75)", "Classe 4 (≤1)"])

# Calcul des pourcentages
class_counts = df_["Classe"].value_counts(normalize=True) * 100
df_pie = class_counts.reset_index()
df_pie.columns = ["Classe", "Pourcentage"]

# Création du pie-chart interactif
fig = px.pie(df_pie, names="Classe", values="Pourcentage", title="Répartition des valeurs de ANNUITY_INCOME_PERCENT  en 4 classes",
             color_discrete_sequence=["#ff9999", "#66b3ff", "#99ff99"])

# Affichage dans Streamlit
st.title("Pie-Chart interactif de la colonne de  ANNUITY_INCOME_PERCENT")
st.plotly_chart(fig)



features=load_features()

st.markdown("<u>Interprétation du modèle - Importance des variables globale :</u>", unsafe_allow_html=True) 
feature_importance=load_feature_importance()
st.write("feature:", len(feature_importance))
st.write("feature_importance: ", len(features))
df = pd.DataFrame({'feature': features,'importance': feature_importance}).sort_values('importance', ascending = False)
df = df.sort_values('importance', ascending = False).reset_index()
    
# Normalize the feature importances to add up to one
df['importance_normalized'] = df['importance'] / df['importance'].sum()
# Make a horizontal bar chart of feature importances
fig=plt.figure(figsize = (15, 10))
ax = plt.subplot()
# Need to reverse the index to plot most important on top
ax.barh(list(reversed(list(df.index[:30]))), 
df['importance_normalized'].head(30), 
align = 'center', edgecolor = 'k')
    
# Set the yticks and labels
ax.set_yticks(list(reversed(list(df.index[:30]))))
ax.set_yticklabels(df['feature'].head(30))
    
# Plot labeling
plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
st.pyplot(fig)
        

