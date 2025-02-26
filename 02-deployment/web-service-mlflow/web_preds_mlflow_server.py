import mlflow
from pandas import DataFrame
from flask import Flask, request, jsonify

# Esse servidor Flask tem a vantagem de não depender do MLflow Tracking Server durante a inferência.
# Vale ressaltar que o MLflow é crucial nesse código para gerenciar e carregar os artefatos do modelo,
# garantindo reprodutibilidade e organização. Utilizando apenas o RUN_ID, o MLflow permite carregar 
# em tempo de execução os modelos e pipelines de pré-processamento armazenados no Azure Blob Storage.
# Isso também permite que a inferência seja realizada de forma mais rápida e eficiente, 
# já que evita chamadas à API do Tracking Server durante a execução.


def set_features(trip):
    """
    Recebe um dicionário contendo informações de uma corrida de táxi
    e gera as features experadas por um modelo de Machine Learning.

    Args:
        trip (dict): Dicionário contendo os dados do trajeto, incluindo:
            - 'trip_distance': Distância percorrida.
            - 'PULocationID': ID do local de embarque.
            - 'DOLocationID': ID do local de desembarque.

    Returns:
        dict: Dicionário com as features processadas.

    Raises:
        KeyError: Se qualquer chave necessária ('trip_distance', 'PULocationID', 'DOLocationID') estiver ausente.
    """
    try:
        # Inicializa o dicionário
        features = dict()

        # Gera um UUID único para o rideID
        #features['rideID'] = str(uuid.uuid4())

        # Passa o valor de trip_distance
        features['trip_distance'] = trip['trip_distance']

        # Cria PU_DO_LocationID combinando os IDs de origem e destino
        features['PU_DO_LocationID'] = f"{trip['PULocationID']}_{trip['DOLocationID']}"

        features_df = DataFrame([features])

        return features_df
    
    except KeyError as e:
        raise KeyError(f"❌ Chave ausente no dicionário de entrada: {e}")


def load_models():
    """
    Carrega e recria os ojetos de pré-processamento e de treinamento 
    em tempo de execução a partir do MLflow Model Registry. 
    
    Obs: É Necessário que um tracking server esteja ativo para logar
    os artefatos que foram registrados com aliases. 

    Returns:
    --------
        mlflow_preprocessor : mlflow.pyfunc - Modelo de pré-processamento registrado como PyFunc puro.
        mlflow_model : mlflow.sklearn. - Modelo regressor otimizado para inferência registrado via Sklearn Python API (Pyfunc Flavor).
        run_id : Identificador único do experimento no MLflow
    """

    TRACKING_SERVER_HOST = "http://191.235.81.94:5000"
    mlflow.set_tracking_uri(TRACKING_SERVER_HOST)


    mlflow.set_registry_uri(TRACKING_SERVER_HOST)  # 🔹 Adiciona o registry_uri

    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_SERVER_HOST)

    # Definir nomes e aliases dos artefatos registrados
    preprocessor_registered = "nyc-taxi-preprocessor-prod"
    preprocessor_alias = "latest-preprocessor"
    model_registered = "nyc-taxi-model-prod"
    model_alias = "champion"

    # Carregar o preprocessador e o modelo do MLflow
    mlflow_preprocessor = mlflow.pyfunc.load_model(f"models:/{preprocessor_registered}@{preprocessor_alias}")
    mlflow_model = mlflow.sklearn.load_model(f"models:/{model_registered}@{model_alias}")

    # Obter o run_id associado aos artefatos
    model_version = client.get_model_version_by_alias(model_registered, model_alias)
    RUN_ID = model_version.run_id

    return mlflow_preprocessor, mlflow_model, RUN_ID



# Criando a API Flask 
app = Flask('preds-trip-duration')

# Definindo o endpoint /predict, que aceita requisições POST, permitindo postar os dados para o servidor.
@app.route('/predict', methods=['POST'])
def predict():

    trip = request.get_json()
    print(f"📥 Entrada recebida: {trip}")

    if not trip:
        return jsonify({"error": "Nenhum dado recebido"}), 400

    # Prepara os dados
    features = set_features(trip)

    # Carrega os artefatos
    mlflow_preprocessor, mlflow_model, RUN_ID = load_models()


    X_processed = mlflow_preprocessor.predict(features)

    # Realiza a previsão
    pred = mlflow_model.predict(X_processed)

    # Convertendo para tipos Python nativos
    preds_trip_duration = float(pred[0])  # Garante que seja serializável
    model_version = str(RUN_ID)  # Garante que não seja um objeto inesperado

    results = {
        'preds_trip_duration': preds_trip_duration,
        'model_version': model_version
    }

    return jsonify(results)


# Inicia o servidor Flask na porta 9696.
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

# Alternative to flask:
# In bash type: gunicorn --bind=0.0.0.0:9696 web_preds_mlflow_server:app

