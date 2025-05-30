
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import shap
from lightgbm import LGBMClassifier

os.environ["JOBLIB_MULTIPROCESSING"] = "0"

#model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "best_model.pickle"))
#model = pickle.load(open(r"model.pkl", 'rb'))

#data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "sample_full.csv"))
#df = pd.read_csv(r"api/df_api_1000.csv")

#def load_model():
    #try:
        #return pickle.load(open(r"api/model.pkl", 'rb'))
    #except FileNotFoundError:
        #print("Error: Model file not found. Make sure 'model.pkl' is in the correct directory.")
        #return None

#model = load_model()

# Charger le modèle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# Charger les données
df = pd.read_csv(r"df_api_1000.csv")
df=df.loc[:, ~df.columns.str.match ('Unnamed')]
#df_calc= df_.drop(['TARGET', 'SK_ID_CURR'], axis=1)
# df.drop(columns='index', inplace=True)

# Define the threshold of for application.
threshold = 0.07

df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

top_features = [
    "CREDIT_TERM", "DAYS_BIRTH", "AMT_ANNUITY", "CREDIT_INCOME_PERCENT","ANNUITY_INCOME_PERCENT"
]

explainer = shap.Explainer(model, df[top_features])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 60  # Timeout en secondes

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        print("🔍 Données reçues :", data)

        if not data or "id_client" not in data:
            return jsonify({'error': "ID client manquant"}), 400

        ID = int(data["id_client"])

        if ID not in df['SK_ID_CURR'].values:
            return jsonify({'error': "Client non trouvé"}), 404

        client_row = df[df['SK_ID_CURR'] == ID]
        X = client_row.drop(['SK_ID_CURR'], axis=1)
        X_top = client_row[top_features]

        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[:, 1][0]
        #if proba < threshold:
            #prediction=0
        #else: prediction=1
        client_data = X_top.iloc[0].to_dict()
        global_means = df[top_features].mean().round(4).to_dict()

        shap_values = explainer(X_top, check_additivity=False)
        shap_dict = dict(zip(top_features, shap_values.values[0].tolist()))
        print(shap_values[0])

        return jsonify({
            'id_client': ID,
            'prediction': int(prediction),
            'probability': round(proba, 4),
            'features': client_data,
            #'global_means': global_means,
            'shap_values': shap_dict
        })

    except Exception as e:
        print("🚨 Erreur dans l'API :", str(e))
        return jsonify({'error': str(e)}), 500


# charger les features du modèle
@app.route("/api//load_features", methods=["GET"])
def load_features():
    df=df.drop(['SK_ID_CURR'], axis=1)
    features=df.columns.values.tolist() 
    return jsonify(features)


# calcul de l'importance des features
@app.route("/api/load_feature_importance", methods=["GET"])
def load_feature_importance():
    features=df.columns.values.tolist()
    for f in features:
        features_importance = model.feature_importances_
        features_importance=features_importance.tolist()
    return jsonify(features_importance)


@app.route('/api/ids', methods=['GET'])
def get_ids():
    ids = sorted(df['SK_ID_CURR'].tolist())
    return jsonify({'ids': ids})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080,_timeout=60))
    app.run(host='0.0.0.0', port=port)
