import uuid
import mlflow
import pandas as pd
from flask import Flask, request, jsonify

# Esse servidor Flask tem a vantagem de não depender da API Tracking Server do MLflow. 
# Utilizando apenas o RUN_ID, é possível logar aos artefatos armazenados no Azure Blob Storage.
# Permitindo uma inferência mais rápida. Por outro lado a independência do Tracking Server
# perde-se no aspecto reprodutibilidade e organização.



#TRACKING_SERVER_HOST = "http://191.235.81.94:5000"
#mlflow.set_tracking_uri(TRACKING_SERVER_HOST)


RUN_ID_MODEL = '03dd38b76561449bb9c03bb097d3ccb8'
RUN_ID_PREPROCESSOR = 'd08af4a91b814be9b04efff1d4c6ecf4'

# Caminho do artefato referente ao pipeline de preprocessamento
path_preprocessor = f'wasbs://mlflowcontainer@mlflowartifacts01.blob.core.windows.net/nyc-mlflow-artifacts/{RUN_ID_PREPROCESSOR}/artifacts/preprocessor'
# Carregar o artefato de preprocessamento como PyFuncModel
preprocessor = mlflow.pyfunc.load_model(path_preprocessor)

# Caminho do artefato referente ao pipeline de treinamento do modelo de inferência
path_model = f'wasbs://mlflowcontainer@mlflowartifacts01.blob.core.windows.net/nyc-mlflow-artifacts/{RUN_ID_MODEL}/artifacts/models'
# Carregar o artefato de inferência 
model = mlflow.sklearn.load_model(path_model)
    


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

        # Passa o valor de trip_distance
        features['trip_distance'] = trip['trip_distance']

        # Cria PU_DO_LocationID combinando os IDs de origem e destino
        features['PU_DO_LocationID'] = f"{trip['PULocationID']}_{trip['DOLocationID']}"

        # Converter para DataFrame
        features_df = pd.DataFrame([features])

        return features_df
    
    except KeyError as e:
        raise KeyError(f"❌ Chave ausente no dicionário de entrada: {e}")



# Criando a API Flask 
app = Flask('preds-trip-duration')

# Definindo o endpoint /predict, que aceita requisições POST, permitindo postar os dados para o servidor.
@app.route('/predict', methods=['POST'])
def predict():
    # Acessa os dados json da requisição POST (web services funcionam melhor com json)
    trip = request.get_json() 
    # Prepara os dados
    features = set_features(trip)

    # Faz a predição
    X = preprocessor.predict(features)
    pred = model.predict(X)

    # Convertendo para tipos Python nativos
    preds_trip_duration = float(pred[0])  # Garante que seja serializável
    model_version = str(RUN_ID_MODEL)  # Garante que não seja um objeto inesperado

    results = {
        'duration': preds_trip_duration,
        'model_version': model_version
    }

    return jsonify(results) # Retorna o dicionário de resultados em formato json

# Inicia o servidor Flask na porta 9696.
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

# Alternative to flask:
# In bash type: gunicorn --bind=0.0.0.0:9696 web_preds:app

