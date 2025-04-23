
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import shap
from lightgbm import LGBMClassifier

os.environ["JOBLIB_MULTIPROCESSING"] = "0"

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "best_model.pickle"))
model = pickle.load(open(model_path, "rb"))

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "sample_full.csv"))
df = pd.read_csv(data_path)
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

top_features = [
    "EXT_SOURCE_3", "EXT_SOURCE_2", "CREDIT_TO_ANNUITY_RATIO",
    "DAYS_BIRTH", "DAYS_LAST_PHONE_CHANGE"
]

explainer = shap.Explainer(model, df[top_features])

app = Flask(__name__)

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
            'global_means': global_means,
            'shap_values': shap_dict
        })

    except Exception as e:
        print("🚨 Erreur dans l'API :", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/ids', methods=['GET'])
def get_ids():
    ids = sorted(df['SK_ID_CURR'].tolist())
    return jsonify({'ids': ids})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
