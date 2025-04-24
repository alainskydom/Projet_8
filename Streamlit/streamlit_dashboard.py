import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Crédit", layout="centered")
st.title("📊 Dashboard - Décision de crédit")

# 🔁 Récupérer la liste des IDs depuis l'API
try:
    id_response = requests.get("https://shap-credit-api-mamdou-0a39fd6254f1.herokuapp.com/api/ids")
    id_response.raise_for_status()
    ids = id_response.json().get("ids", [])
    client_id = st.selectbox("Sélectionnez un identifiant client :", ids)
except Exception as e:
    st.error(f"Erreur lors de la récupération des IDs : {e}")
    st.stop()

if st.button("Obtenir la prédiction via API"):
    url = "https://shap-credit-api-mamdou-0a39fd6254f1.herokuapp.com/api/predict"

    try:
        response = requests.post(url, json={"id_client": int(client_id)})
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            proba = result["probability"]

            if prediction == 1:
                st.error("❌ Prêt NON accordé")
            else:
                st.success("✅ Prêt accordé")

            st.metric(label="Probabilité de défaut", value=f"{proba*100:.2f} %")

            st.subheader("🧾 Comparaison client vs moyenne (5 variables clés)")
            df_compare = pd.DataFrame({
                "Valeur client": result["features"],
                "Moyenne globale": result["global_means"]
            })
            st.dataframe(df_compare)

            st.subheader("📉 Visualisation comparative")
            fig, ax = plt.subplots(figsize=(8, 4))
            df_compare.plot(kind="bar", ax=ax)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

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
