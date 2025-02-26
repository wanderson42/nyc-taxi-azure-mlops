import os
import io
import sys
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import asyncio
import uuid
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.context import get_run_context
from prefect.logging import get_run_logger
from prefect_azure import AzureBlobStorageCredentials
from azure.identity import DefaultAzureCredential
from prefect_azure.blob_storage import blob_storage_download
from prefect_azure.blob_storage import blob_storage_upload

################################################################################
#                            Previsões em Lote                                 # 
#------------------------------------------------------------------------------#
# Este programa realiza previsões em lote da "Duração da Viagem" de corridas de#
# táxi, um atributo geralmente presente no conjunto de dados, pois refere-se a # 
# corridas já concluídas. Apesar disso, o foco aqui é consolidar boas práticas #
# de MLOps e Cloud.                                                            #
# -----------------------------------------------------------------------------#
# Aqui usufruimos do framework prefect para gerenciar o modelo preditivo.      #
# De fato é possível desenvolver um código menos dependente da insfraestrutura #
# do prefect. No entanto, essa integração proporciona melhor gerenciamento do  # 
# modelo em produção, com monitoramento, agendamento e facilidade de reexecução#
# em caso de falhas, permitindo maior controle sobre a execução das tarefas.   #
###############################################################################


# Configura a URI do servidor remoto MLflow
TRACKING_SERVER_HOST = "http://191.235.81.94:5000"
mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
client = MlflowClient(tracking_uri=TRACKING_SERVER_HOST)


@task
async def load_data(run_date, taxi_type):
    """
    Carrega os dados de entrada do Azure Blob Storage.

    Esta função baixa um arquivo Parquet da Azure Blob Storage com base na data de execução 
    e no tipo de táxi especificado, retornando os dados como um fluxo de bytes.

    Args:
        run_date (datetime): Data de referência para o arquivo de entrada.
        taxi_type (str): Tipo de táxi (por exemplo, 'yellow' ou 'green').

    Returns:
        bytes: Conteúdo do arquivo Parquet baixado.

    Raises:
        Exception: Se ocorrer um erro durante o download do arquivo.
    """

    logger = get_run_logger()
    input_file = f'nyc-trip-data/{taxi_type}_tripdata_{run_date.year}-{str(run_date.month).zfill(2)}.parquet'

    #connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    #azure_credentials = AzureBlobStorageCredentials(connection_string=connection_string)

    # Usa DefaultAzureCredential para autenticação segura
    azure_credentials = AzureBlobStorageCredentials(
        account_url="https://mlflowartifacts01.blob.core.windows.net", 
        credential=DefaultAzureCredential()
    )

    # Baixando os dados da Azure Blob Storage como um fluxo de bytes 
    try:
        input_data = await blob_storage_download(
            container="mlflowcontainer",
            blob=input_file,
            blob_storage_credentials=azure_credentials
        )
    except Exception as e:
        logger.error(f"Erro ao baixar o arquivo {input_file}: {e}")
        raise
    
    return input_data 


@task(log_prints=True)
def ingest_data(file_parquet):
    """
    Prepara os dados de entrada para inferência.

    Converte um arquivo Parquet em pandas.DataFrame, verifica colunas obrigatórias, calcula a duração 
    das viagens, filtra valores inválidos e seleciona as colunas necessárias. Retorna os atributos 
    preditores (X) e o atributo a ser predito (y).

    Parameters:
    -----------
    file_parquet : str or bytes
        Caminho ou fluxo de bytes do arquivo Parquet.

    Returns:
    --------
    tuple
        X : pd.DataFrame - Atributos preditores.
        y : np.ndarray - Duração das viagens em minutos.
    """

    # Converte os bytes baixados em um objeto BytesIO
    pq_bytes = file_parquet
    pq_file = io.BytesIO(pq_bytes)
    
    # Converte os dados em pandas DataFrame
    X = pd.read_parquet(pq_file)

    # Garantir que X seja um DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Os dados de entrada devem ser fornecidos como DataFrame.")

    # Verificar features necessárias
    required_features = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'trip_distance', 'DOLocationID', 'PULocationID']
    missing_features = [col for col in required_features if col not in X.columns]
    if missing_features:
        raise ValueError(f"Faltam as colunas: {', '.join(missing_features)}")

    # Processamento de datas
    X['lpep_pickup_datetime'] = pd.to_datetime(X['lpep_pickup_datetime'], errors='coerce')
    X['lpep_dropoff_datetime'] = pd.to_datetime(X['lpep_dropoff_datetime'], errors='coerce')
    X['duration'] = (X['lpep_dropoff_datetime'] - X['lpep_pickup_datetime']).dt.total_seconds() / 60
    # Filtrar valores inválidos
    X = X[(X.duration >= 1) & (X.duration <= 60)]

    # Criar identificador único
    X['rideID'] = X['lpep_pickup_datetime'].dt.strftime("%Y/%m_") + X.index.astype(str)

    # Criando a feature PU_DO_LocationID
    X['PU_DO_LocationID'] = X['PULocationID'].astype(str) + '_' + X['DOLocationID'].astype(str)

    # Selecionar as features finais
    #final_features = ['rideID', 'PU_DO_LocationID', 'trip_distance', 'duration']

    # Atributos preditores
    #X = X[final_features]

    # Atributo a ser predito
    y = X.pop('duration').values

    return X, y

'''
@task
def load_models(run_id):
    """
    Carrega e recria os ojetos de pré-processamento e de treinamento 
    em tempo de execução a partir de uma run_id do MLflow. 

    Parameters:
    -----------
    run_id : str
        Identificador único do experimento no MLflow.

    Returns:
    --------
    tuple
        pipPreprocessor : mlflow.pyfunc - Modelo de pré-processamento registrado como PyFunc puro.
        pipTrain : mlflow.sklearn. - Modelo regressor otimizado para inferência registrado via Sklearn Python API (Pyfunc Flavor).
    """
    # Caminho do artefato referente ao pipeline de preprocessamento
    path_preprocessor = f'wasbs://mlflowcontainer@mlflowartifacts01.blob.core.windows.net/nyc-mlflow-artifacts/{run_id}/artifacts/preprocessor'
    # Carregar o artefato de preprocessamento como PyFuncModel
    pipPreprocessor = mlflow.pyfunc.load_model(path_preprocessor)

    # Caminho do artefato referente ao pipeline de treinamento do modelo de inferência
    path_model = f'wasbs://mlflowcontainer@mlflowartifacts01.blob.core.windows.net/nyc-mlflow-artifacts/{run_id}/artifacts/models'
    # Carregar o artefato de inferência 
    pipTrain = mlflow.sklearn.load_model(path_model)
    
    return pipPreprocessor, pipTrain
'''

@task
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

    # Definir nomes e aliases dos artefatos registrados
    preprocessor_registered = "nyc-taxi-preprocessor-prod"
    preprocessor_alias = "latest-preprocessor"
    model_registered = "nyc-taxi-model-prod"
    model_alias = "champion"

    # Carregar o preprocessador e o modelo do MLflow
    mlflow_preprocessor = mlflow.pyfunc.load_model(f"models:/{preprocessor_registered}@{preprocessor_alias}")
    mlflow_model = mlflow.sklearn.load_model(f"models:/{model_registered}@{model_alias}")

    # Obter o run_id associado aos artefatos
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_registered, model_alias)
    run_id = model_version.run_id

    return mlflow_preprocessor, mlflow_model, run_id


@task
async def save_data(X, y_preds, run_id, run_date, taxi_type):
    """
    Função assíncrona para salvar previsões no Azure Blob Storage.

    Esta função constrói um DataFrame a partir dos dados de entrada e previsões, 
    gera um arquivo Parquet em memória, e o envia para o Azure Blob Storage.

    Args:
        X (pd.DataFrame): Dados de entrada com identificadores únicos ('rideID').
        y (array-like): Valores reais do alvo (duração real da viagem).
        y_preds (array-like): Previsões do modelo (duração prevista da viagem).
        run_id (str): Identificador da versão do modelo.
        run_date (datetime): Data referente ao período em que os registro das viagens de taxi foram coletadas.
        taxi_type (str): Tipo de táxi (por exemplo, 'yellow' ou 'green').

    Returns:
        None
    """

    # Construindo o DataFrame das previsões
    # Construindo os dados de referência 
    X['model_version'] = run_id
    X['preds_trip_duration'] = y_preds

    # Gerando o arquivo Parquet em memória
    output_file = f'{taxi_type}_tripdata_{run_date.year}-{str(run_date.month).zfill(2)}.parquet'
    buffer = io.BytesIO()
    X.to_parquet(buffer, index=False)
    buffer.seek(0)  # Retornar ao início do arquivo em memória
    output_path = os.path.abspath(output_file)
    X.to_parquet(output_path, index=False)

    # Credenciais para acessar o azure blob store
    #connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    #azure_credentials = AzureBlobStorageCredentials(connection_string=connection_string)

    # Usa DefaultAzureCredential para autenticação segura
    azure_credentials = AzureBlobStorageCredentials(
        account_url="https://mlflowartifacts01.blob.core.windows.net", 
        credential=DefaultAzureCredential()
    )

    # Salvando o dataframe parquet no azure blob store 
    await blob_storage_upload(
        data=buffer.read(),
        container="mlflowcontainer",
        blob=f'nyc-preds-trip-duration/{output_file}',
        blob_storage_credentials=azure_credentials,
        overwrite=True,
    )
    # O await permite a execução assíncrona, para evitar bloqueios enquanto
    # a operação de salvamento dos dados está em andamento, 


@flow
async def predictions(
    taxi_type: str,
    run_date: datetime = None):
    """
    Executa o fluxo de previsões para dados de corridas de táxi.

    Este fluxo realiza as seguintes etapas:
    1. Carrega os dados de entrada para a data especificada no Azure Blob Storage.
    2. Aplica limpeza e preprocessamento nos dados.
    3. Carrega os modelos necessários a partir de artefatos do MLflow.
    4. Realiza previsões usando o modelo treinado.
    5. Salva os resultados no Azure Blob Storage.

    Args:
        taxi_type (str): Tipo de táxi (e.g., "yellow", "green").
        run_id (str): Identificador do modelo no MLflow.
        run_date (datetime, optional): Data de referência para a execução do fluxo. 
            Se não for fornecida, será utilizada a data de execução agendada.
    """

    logger = get_run_logger()
    # Se for uma tarefa agendada
    if run_date is None:
       # Buscando o contexto de execução, i.e. A data que a tarefa que foi agendada
       run_context = get_run_context()
       run_date = run_context.flow_run.expected_start_time

    # Uma vez que estamos realizando mensalmente previsões em lote
    # A execução sempre refere-se aos dados do mês anterior  
    
    #run_date = run_date - relativedelta(months=1) # subtraindo 1 mês

    logger.info(f"Carregando os artefatos registrados no MLflow...")
    # Carregar modelos de ELT e ML
    mlflow_preprocessor, mlflow_model, run_id = load_models()

    # Faz o download dos dados para inferência
    logger.info("Carregando os dados para inferência.")    
    input_raw_data = await load_data(run_date, taxi_type)

    # Limpando os dados de entrada
    logger.info("Aplicando uma limpeza nos Dados.") 
    X, y = ingest_data(input_raw_data)

    # Verificar se 'rideID' existe em X
    #if 'rideID' not in X.columns:
        # Gera identificadores únicos para cada linha
    #    X['rideID'] = [str(uuid.uuid4()) for _ in range(len(X))]  

    # Definir colunas numéricas e categóricas
    numerical_features=['trip_distance']
    categorical_features=['PU_DO_LocationID']

    logger.info(f"Aplicando a etapa de preprocessamento...")
    # Transformar os dados usando o método 'predict' do TranformerWrapper
    X_processed = mlflow_preprocessor.predict(X[numerical_features + categorical_features])

    # Realizar previsões com o modelo treinado
    logger.info("Realizando previsões com o modelo treinado.")
    y_preds = mlflow_model.predict(X_processed)
    print(y_preds)

    # Salvar os resultados
    await save_data(X, y_preds, run_id, run_date, taxi_type)
    logger.info(f"Salvando os resultados na Azure Blob Storage...")    





#-------------------------------begin-----------------------------------#
# uncoment for run in bash:
# python3.12 batch_score.py 'green' 2024 10 15 '71eed80fe08e4f4bbe882d06e19836b9'
 
def run():
    # Usando sys.arg para receber os inputs via linha de comando
    taxi_type = sys.argv[1] # ex: 'green'
    year = int(sys.argv[2]) # ex: 2024
    month = int(sys.argv[3]) # ex: 10
    day = int(sys.argv[4]) # ex: 15
    #run_id = sys.argv[5] # ex: '71eed80fe08e4f4bbe882d06e19836b9'

    asyncio.run(
        predictions(
        taxi_type = taxi_type,
        run_date = datetime(year=year, month = month, day = day)
        )
    )

if __name__ == '__main__':
    run()
#--------------------------------end------------------------------------#


#-------------------------------begin-----------------------------------#
# Uncoment for run in bash: python3.12 batch_score.py
'''
#run_id = '5fec3aa577c54407b982ab74fac8d1f0' 
year =  2024
month = 5
day = 15
taxi_type = 'green'



asyncio.run(
    predictions(
    taxi_type = taxi_type,
    run_date = datetime(year=year, month = month, day = day)
    )
)

'''
#--------------------------------end------------------------------------#