import os
import requests
import mlflow
import pandas as pd
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId

EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:8085')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")


app = Flask('trip_duration')
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("taxi_data")

def set_features(trip):
    """Transforma os dados de entrada no formato esperado pelo modelo."""
    try:
        features_dict = {
            'trip_distance': trip['trip_distance'],
            'PU_DO_LocationID': f"{trip['PULocationID']}_{trip['DOLocationID']}"
        }
        features_df = pd.DataFrame([features_dict])
        return features_dict, features_df
    except KeyError as e:
        raise KeyError(f"❌ Chave ausente no dicionário de entrada: {e}")

def load_models():
    """Carrega os modelos do MLflow."""
    TRACKING_SERVER_HOST = "http://191.235.81.94:5000"
    mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
    mlflow.set_registry_uri(TRACKING_SERVER_HOST)

    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_SERVER_HOST)

    preprocessor_registered = "nyc-taxi-preprocessor-prod"
    preprocessor_alias = "latest-preprocessor"
    model_registered = "nyc-taxi-model-prod"
    model_alias = "champion"

    mlflow_preprocessor = mlflow.pyfunc.load_model(f"models:/{preprocessor_registered}@{preprocessor_alias}")
    mlflow_model = mlflow.sklearn.load_model(f"models:/{model_registered}@{model_alias}")

    model_version = client.get_model_version_by_alias(model_registered, model_alias)
    model_id = model_version.run_id
    # Obtém os detalhes da execução pelo run_id
    run_info = client.get_run(model_id)
    # Obtém o nome do modelo a partir das tags do experimento (caso tenha sido registrado como artefato)
    model_name = run_info.data.tags.get("mlflow.runName", "Nome do modelo não encontrado")
    RUN_ID = f"{model_name} (MLflow ID: {model_id})"

    return mlflow_preprocessor, mlflow_model, RUN_ID

# Carregar modelos ao iniciar a aplicação
mlflow_preprocessor, mlflow_model, RUN_ID = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    trip = request.get_json()
    print(f"📥 Entrada recebida: {trip}")

    if not trip:
        return jsonify({"error": "Nenhum dado recebido"}), 400

    trip['PU_DO'] = '%s_%s' % (trip['PULocationID'], trip['DOLocationID'])

    features_dict, features_df = set_features(trip)
    X = mlflow_preprocessor.predict(features_df)
    pred = mlflow_model.predict(X)

    results = {
        'trip_duration': float(pred[0]),
        'model_version': str(RUN_ID)
    }

    features_dict.update(results)
    collection.insert_one(features_dict)

    # Converte ObjectId para string antes de retornar a resposta
    for key, value in features_dict.items():
        if isinstance(value, ObjectId):
            features_dict[key] = str(value)

    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/taxi", json=[features_dict])

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

# Alternative to flask:
# In bash type: gunicorn --bind=0.0.0.0:9696 app:app    