import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
#from scipy.sparse import csr_matrix
from prefect import flow, task, tags
from prefect_azure.blob_storage import AzureBlobStorageContainer
import mlflow
import pickle
import cloudpickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from prefect import task, flow
from prefect.logging import get_run_logger
from prefect.artifacts import create_table_artifact, create_link_artifact
from mlflow.models.signature import infer_signature
from mlflow.entities import LifecycleStage
from mlflow.tracking import MlflowClient
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor




################################################################################
#                      Extract Loading Transform Pipeline                      #
################################################################################
#---------------------------------- BEGIN -------------------------------------#

class extractLoader(BaseEstimator, TransformerMixin):
    """
    Transformer para carregar, limpar e pré-processar dados de táxi em formato Parquet.

    Compatível com pipelines do Scikit-Learn e integrável ao Prefect para workflows escaláveis.
    """
    # método construtor
    def __init__(self, file_path=None):
        """
        Inicializa o transformer com o caminho do arquivo externo.

        Parameters:
        -----------
        file_path : str ou list
            Caminho(s) do(s) arquivo(s) parquet a ser(em) carregado(s).
        """
        self.file_path = file_path

    # método privado para uso interno  
    def _remove_outliers(self, df, feature):
        """
        Remove outliers usando o método IQR para uma coluna específica.

        Parameters:
        -----------
        df : pd.DataFrame
            O DataFrame contendo os dados.
        feature : str
            O nome da coluna para remoção de outliers.

        Returns:
        --------
        df : pd.DataFrame
            DataFrame sem os outliers para a coluna especificada.
        """
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR

        return df[~((df[feature] > upper_limit) | (df[feature] < lower_limit))]

    def fit(self, X=None, y=None):
        """
        Método necessário para compatibilidade com a API do Scikit-learn.

        Parameters:
        -----------
        X : Ignorado
        y : Ignorado

        Returns:
        --------
        self : O próprio objeto.
        """
        return self

    def transform(self, X=None, y=False):
        """
        Carrega, limpa e pré-processa dados.

        Returns:
        --------
        df : pd.DataFrame
            DataFrame processado e pronto para uso.
        y : np.ndarray, opcional
            Atributo alvo extraído dos dados, se aplicável.

        Raises:
        -------
        ValueError:
            - Se o caminho do arquivo (`file_path`) não for especificado.
            - Se um dos arquivos fornecidos não contiver as colunas necessárias para criar o atributo alvo.
            - Se ocorrer um erro ao processar um arquivo.
            - Se nenhum arquivo válido for processado.        
        """
        if not self.file_path:
            raise ValueError("O caminho do arquivo (file_path) ou lista de arquivos deve ser especificado.")

        # Criando uma lista para lidar com múltiplos arquivos
        file_paths = self.file_path if isinstance(self.file_path, list) else [self.file_path]

        # Define as colunas obrigatórias e adicionais condicionais
        required_columns = ["fare_amount", "total_amount", "trip_distance", "tip_amount",
                            "congestion_surcharge", "DOLocationID", "PULocationID"]
        conditional_columns = ["lpep_pickup_datetime", "lpep_dropoff_datetime"]

        # Inicializa uma lista para armazenar DataFrames processados
        dataframes = []

        # Definir uma lista de dataframes para extrair somente os 
        # atributos de interesse do arquivo parquet, assim economizando memória.
        # Ao termino do laço, concatenamos tudo.  
        for path in file_paths:
            try:
                # Obter as colunas disponíveis no arquivo atual
                available_columns = pd.read_parquet(path, columns=None).columns

                # Verificar se as colunas condicionais estão presentes
                if not set(conditional_columns).issubset(available_columns):
                    raise ValueError(f"O arquivo '{path}' não contém as colunas necessárias para criar o atributo alvo.")

                # Determinar as colunas a carregar
                selected_columns = list(set(required_columns + conditional_columns) & set(available_columns))

                # Carregar os dados do arquivo atual
                df = pd.read_parquet(path, columns=selected_columns)

                # Convertendo as colunas para datetime e calculando a duração
                df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
                df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
                df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
                df['duration'] = df['duration'].apply(lambda time_diff: time_diff.total_seconds() / 60)
                # Criar rideID
                df['rideID'] = df['lpep_pickup_datetime'].apply(lambda dt: f"{dt.year:04d}/{dt.month:02d}_" ) + df.index.astype(str)

                # Remoção de outliers dos atributos preditores
                df = self._remove_outliers(df, "trip_distance")
                df = self._remove_outliers(df, "fare_amount")
                df = self._remove_outliers(df, "total_amount")
                df = self._remove_outliers(df, "duration")

                # Garantindo que IDs sejam tratados como strings
                df[["DOLocationID", "PULocationID"]] = df[["DOLocationID", "PULocationID"]].astype(str)

                # Filtrar os dados finais com base nos atributos obrigatórios
                df = df[['rideID'] + required_columns + ["duration"]]


                # Filtrar valores inválidos                
                df = df[df['fare_amount'] >= 1]
                df = df[df['duration'] >= 1]

                # Adicionar o DataFrame processado à lista
                dataframes.append(df)

            except Exception as e:
                raise ValueError(f"Erro ao processar o arquivo '{path}': {e}")

        # Concatenar todos os DataFrames
        if dataframes:
            df = pd.concat(dataframes, ignore_index=True)
        else:
            raise ValueError("Nenhum arquivo válido foi processado.")

        if df.columns.isin(["duration"]).any():
            # Extrair o atributo alvo
            y = df.pop('duration')

            # Transforma y em numpy.ndarray para compatibilidade com matrizes esparsas
            y = np.array(y)

            return df, y
        else:
            return df


# Tarefa para ajustar e executar o pipeline completo de ELT
# Contexto compartilhado
context = {}
@task()
def pipELT(
    file_path, 
    train_mode=True, 
    preprocessor=None,
    ):
    
    """
    Tarefa para executar o pipeline de ETL para dados de táxi, ajustando ou aplicando pré-processamento.

    Parameters:
    -----------
    file_path : str
        Caminho do arquivo (local ou URL) contendo os dados em formato Parquet.

    train_mode : bool, opcional, default=True
        Se True, realiza o ajuste do pipeline e retorna X e y.
        Se False, utiliza o pré-processador fornecido para transformar novos dados.

    preprocessor : ColumnTransformer, opcional, default=None
        Pré-processador ajustado para uso em modo de teste.

    Returns:
    --------
    X_processed : sparse matrix
        Dados transformados e prontos para modelagem.

    y : np.ndarray, opcional
        Atributo alvo (se presente nos dados). None se não houver atributo alvo.

    preprocessor : ColumnTransformer
        Pré-processador ajustado (retornado apenas no modo de treino).
    """
    logger = get_run_logger()

    # Definir colunas numéricas e categóricas
    num_columns = ["fare_amount", "total_amount", "trip_distance", "tip_amount", "congestion_surcharge"]
    nominal_columns = ["PULocationID", "DOLocationID", 'rideID']

    if train_mode:
        create_link_artifact(
            key="train",  # Chave indicando artefato de treino
            link="",  # Link fictício para dados de treino
            description="Protocolo ELT nos dados de treino - metodo fit_transform()"
        )        
        # Pipeline para atributos numéricos
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', RobustScaler())
        ])

        # Pipeline para atributos categóricos
        nominal_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combinação de pré-processadores para colunas numéricas e categóricas
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_columns),
            ('cat', nominal_pipeline, nominal_columns)
        ])

        # Construção do pipeline completo
        full_pipeline = Pipeline([
            ('data_loader_cleaner', extractLoader()),  # Classe anterior para carregar e limpar os dados
            ('preprocessor', preprocessor)
        ])

        # Ajustar o caminho do arquivo para o loader
        full_pipeline.set_params(data_loader_cleaner__file_path=file_path)

        # Carregar dados limpos e atributo alvo
        transformed_data = full_pipeline.named_steps['data_loader_cleaner'].transform(None)

        # Separar X (atributos) e y (alvo)
        if isinstance(transformed_data, tuple) and len(transformed_data) == 2:
            df, y = transformed_data
        else:
            df = transformed_data
            y = None

        # Ajustar o pré-processador e transformar os dados
        X_processed = full_pipeline.named_steps['preprocessor'].fit_transform(df)

        # Obter a data atual
        date = datetime.now(pytz.timezone("America/Sao_Paulo")).strftime("date-%Y-%m-%d-clock-%H-%M-%S")

        # Diretório onde o pipeline preprocessor será salvo
        preprocessor_directory = "./preprocessor/"
        os.makedirs(preprocessor_directory, exist_ok=True)  # Cria o diretório se não existir        

        # Nome do arquivo do pipeline
        elt_filename = f"elt.pkl"
        # Armazenar no contexto
        context["elt_filename"] = elt_filename

        # Salvar o pipeline localmente
        with open(elt_filename, "wb") as f:
            cloudpickle.dump(full_pipeline, f)
        logger.info(f"⚠️ Redundância -> Pré-processador salvo localmente como {elt_filename}")
 
        # Retornar os dados processados e o pré-processador ajustado para o teste
        return X_processed, y, full_pipeline.named_steps['preprocessor']

    else:
        create_link_artifact(
            key="test",  # Chave indicando artefato de treino
            link="",  # Link fictício para dados de treino
            description="Protocolo ELT nos dados de test - metodo transform()"
        )           
        # Modo de teste (aplicar transformações em novos dados)
        loader = extractLoader(file_path=file_path)
        transformed_data = loader.transform(X=None)

        # Separar X (atributos) e y (alvo)
        if isinstance(transformed_data, tuple) and len(transformed_data) == 2:
            df, y = transformed_data
        else:
            df = transformed_data
            y = None

        # Transformar os dados com o pré-processador já ajustado
        X_processed = preprocessor.transform(df)

        return X_processed, y
#----------------------------------- END --------------------------------------#

################################################################################
#                           Model Train Pipeline                               #
################################################################################
#---------------------------------- BEGIN -------------------------------------#

@task
def configure_mlflow(exp_name):
    """
    Configura o experimento no MLflow.

    Args:
        exp_name (str): Nome do experimento.
        artifact_location (str): Local no Azure Blob Storage para os artefatos.
        timezone (str): Fuso horário para timestamps.

    Returns:
        str: Nome do experimento configurado.
    """
    logger = get_run_logger()

    # Configura a URI do servidor MLflow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    logger.info(f"Tracking URI configurada para: '{mlflow.get_tracking_uri()}'")

    client = MlflowClient("http://127.0.0.1:5000")

    # Busca o experimento pelo nome
    experiment = client.get_experiment_by_name(exp_name)

    artifact_location = "wasbs://mlflowcontainer@mlflowartifacts01.blob.core.windows.net/"
    if experiment:
        if experiment.lifecycle_stage == LifecycleStage.DELETED:
            # Restaura experimento excluído
            logger.info(f"Restaurando experimento excluído: '{exp_name}' (ID: {experiment.experiment_id}).")
            client.restore_experiment(experiment.experiment_id)
        else:
            logger.info(f"⚠️ Experimento '{exp_name}' já existe e está ativo.")

        # Verifica se o artifact_location corresponde ao esperado
        if experiment.artifact_location != artifact_location:
            logger.warning(
                f"O experimento '{exp_name}' já existe, mas o "
                f"`artifact_location` registrado ({experiment.artifact_location}) "
                f"não corresponde ao especificado ({artifact_location})."
            )
            # Adiciona um aviso, mas não falha automaticamente
    else:
        # Cria um novo experimento se não existir
        logger.info(f"🆕 Criando novo experimento: '{exp_name}' com o artifact_location '{artifact_location}'.")
        experiment_id = mlflow.create_experiment(name=exp_name, artifact_location=artifact_location)
        logger.info(f"Novo experimento criado com ID: {experiment_id}")

    # Define o experimento como ativo
    mlflow.set_experiment(exp_name)
    logger.info(f"Experimento '{exp_name}' configurado como ativo.")

    # Configura o autolog para o MLflow
    mlflow.sklearn.autolog(log_datasets=False)

    return exp_name


@task(log_prints=True)
def train_model(model, param_grid, metric, n_splits, X_train, y_train):
    """
    Treina um modelo usando RandomizedSearchCV.

    Args:
        model: Modelo ou pipeline a ser treinado.
        param_grid (dict): Parâmetros para ajuste de hiperparâmetros.
        metric (str): Métrica de avaliação.
        n_splits (int): Número de splits para validação cruzada.
        X_train (array-like): Dados de treino.
        y_train (array-like): Alvo de treino.

    Returns:
        tuple: Melhor modelo treinado e resultados do RandomizedSearchCV.
    """
    warnings.filterwarnings("ignore", message=".*The total space of parameters.*")
    hp_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,
        scoring=metric,
        cv=n_splits,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    hp_search.fit(X_train, y_train)
    return hp_search.best_estimator_, hp_search


@task(log_prints=True)
def evaluate_and_log(
    run_name: str,
    model_name: str,
    best_model,
    hp_search,
    X_train,
    X_test,
    y_train,
    y_test,
    timezone: str,
    date: str = None,
):  
    """
    Avalia o modelo, registra resultados no MLflow e salva o modelo de forma padronizada.

    Args:
        run_name (str): Nome do experimento.
        model_name (str): Nome do modelo.
        best_model: Modelo treinado.
        hp_search: Resultados do RandomizedSearchCV.
        X_train (array-like): Dados de treino.
        X_test (array-like): Dados de teste.
        y_train (array-like): Atributo Alvo -  treino.
        y_test (array-like): Atributo Alvo - teste.
        timezone (str): Fuso horário.
        date (str): Data no formato 'YYYY-MM-DD' (opcional).
    """
    logger = get_run_logger()

    # Padronizar nome do modelo com base na data ou no nome do experimento
    if date is None:
        date = datetime.now(pytz.timezone(timezone)).strftime("date-%Y-%m-%d-clock-%H-%M-%S")

    # Diretório onde o modelo será salvo
    model_dir = "./models/"
    os.makedirs(model_dir, exist_ok=True)  # Cria o diretório se não existir

    model_filename = f"{model_dir}{run_name}-{date}.pkl"  

    # Predições
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Métricas
    metrics_train = {
        "rmse_train": root_mean_squared_error(y_train, y_pred_train),
        "r2_train": r2_score(y_train, y_pred_train)
    }
    metrics_test = {
        "rmse_test": root_mean_squared_error(y_test, y_pred_test),
        "r2_test": r2_score(y_test, y_pred_test)
    }

    # Log das métricas no MLflow
    for key, value in metrics_train.items():
        mlflow.log_metric(key, value)
    for key, value in metrics_test.items():
        mlflow.log_metric(key, value)

    # Log dos melhores parâmetros
    for key, value in hp_search.best_params_.items():
        mlflow.log_param(key, value)

    # Gerar assinatura e exemplo de entrada
    input_example = X_train[:1]
    signature = infer_signature(X_train, best_model.predict(X_train))

    # Salvar o modelo localmente
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"⚠️ Redundância -> Modelo salvo localmente como {model_filename}")

    # Registrar o modelo no MLflow (onde será salvo na nuvem)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="models",
        input_example=input_example,
        signature=signature
    )
    # Registrar o pré-processador que esta fora do escopo dessa função, através de um gerenciador de contexto
    elt_filename = context.get("elt_filename")  # Obter o valor do contexto
    mlflow.log_artifact(elt_filename)

    logger.info(f"⚠️ Pré-processador salvo na azure cloud e registrado no MLflow com o nome de experimento {elt_filename}")    
    logger.info(f"⚠️ Modelo salvo na azure cloud e registrado no MLflow com o nome de experimento {run_name}")

    # Criar chave sanitizada para o table artifact
    sanitized_key = f"{run_name}-metrics-table".lower().replace("_", "-")

    # Criar a tabela de métricas como artifact para o prefect
    table_data = [
        {"Dataset": "Train", "RMSE": metrics_train["rmse_train"], "R2": metrics_train["r2_train"]},
        {"Dataset": "Test", "RMSE": metrics_test["rmse_test"], "R2": metrics_test["r2_test"]}
    ]
    table_markdown = create_table_artifact(
        key=sanitized_key,
        table=table_data,
        description=f"Métricas de desempenho do modelo {model_name}."
    )
    logger.info(f"Tabela de métricas criada:\n{table_markdown}")

    return {
        "model_name": model_name,
        "run_name": run_name,
        "best_model": best_model,
        "best_params": hp_search.best_params_,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "model_filename": model_filename
    }


@task(log_prints=True)
def pipTrain(exp_name, models, params, metric, n_splits, X_train, X_test, y_train, y_test, timezone="America/Sao_Paulo"):
    """
    Pipeline principal para o treinamento de modelos.

    Args:
        exp_name (str): Nome do experimento.
        models (list): Lista de modelos ou pipelines.
        params (list): Lista de parâmetros.
        metric (str): Métrica de avaliação.
        n_splits (int): Número de splits para validação cruzada.
        X_train (array-like): Dados de treino.
        X_test (array-like): Dados de teste.
        y_train (array-like): Alvo de treino.
        y_test (array-like): Alvo de teste.
        timezone (str): Fuso horário.
    """

    configure_mlflow(exp_name)
 
    logger = get_run_logger()

    logger = get_run_logger()
    if len(models) != len(params):
        raise ValueError("O número de modelos deve corresponder ao número de conjuntos de parâmetros.")

    model_names = [
        model.steps[-1][0] if hasattr(model, "steps") else type(model).__name__ for model in models
    ]
    logger.info(f"Modelos configurados: {model_names}")

    results = []
    for model, param_grid, model_name in zip(models, params, model_names):
        run_name = f"{exp_name}-{model_name}"
        with mlflow.start_run(run_name=run_name):
            # Treinar o modelo
            best_model, hp_search = train_model(model, param_grid, metric, n_splits, X_train, y_train)

            # Avaliar e salvar resultados, incluindo o pré-processador
            result = evaluate_and_log(
                run_name=run_name,
                model_name=model_name,
                best_model=best_model,
                hp_search=hp_search,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                timezone=timezone
            )
            results.append(result)

    logger.info("✅ Pipeline de treinamento concluído!")
    return results
#----------------------------------- END --------------------------------------#

@task()
def azure_fetch_data(date: str):
    """
    Busca os dados de treino e teste numa azure blob store com base na data fornecida.

    Args:
        date (str): Data no formato 'YYYY-MM-DD'.

    Returns:
        tuple: Dados de treino e teste.
    """    
    processed_date = datetime.strptime(date, "%Y-%m-%d")

    # String para determinar a data dos dados de treino (mês atual)
    train_date = processed_date - relativedelta(months=0)
    # String para determinar a data dos dados de teste (mês subsequente)
    test_date = processed_date + relativedelta(months=1)
    
    # Gera o nome dos arquivos de treino e teste baseados na data fornecida.
    train_file = f"green_tripdata_{train_date.year}-{str(train_date.month).zfill(2)}.parquet"
    test_file = f"green_tripdata_{test_date.year}-{str(test_date.month).zfill(2)}.parquet"

    # Classe do Prefect que fornece um método para efetuar uma 
    # transferência segura de arquivos a partir de um Azure Blob Container 
    block = AzureBlobStorageContainer.load("azure-blob-container")

    # Loop para baixar os arquivos de treino e teste localizados na um azure blob storage
    files = [train_file, test_file]
    current_dir = os.getcwd()
    for file in files:
        block.download_object_to_path(
            from_path=os.path.join("nyc-trip-data/", file),
            to_path=os.path.join(current_dir, file)
        )

    return train_file, test_file



@flow(log_prints=True)
def main_flow(date: str = None):
    """
    Fluxo principal que treina modelos com base em dados dinâmicos por data.

    Args:
        date (str): Data no formato 'YYYY-MM-DD'.
    """
    logger = get_run_logger()  

    # Definindo a data padrão como hoje, se não fornecida
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    # Obtendo-se os dados de treino e teste
    train_file, test_file = azure_fetch_data(date)

    logger.info(f"🎲 Nome dos dados de treino:\n{train_file}")
    logger.info(f"🎲 Nome dos dados de teste:\n{test_file}")


    # Execução do fluxo no modo de treino
    X_train_processed, y_train, preprocessor = pipELT(train_file, train_mode=True)
    # Execução do fluxo no modo de teste
    X_test_processed, y_test = pipELT(test_file, train_mode=False, preprocessor=preprocessor)

    # Criar os modelos e hiperparâmetros
    models = [
        #Pipeline([('LinearRegression', LinearRegression())]),
        #Pipeline([('Ridge', Ridge())]),
        #Pipeline([('Lasso', Lasso())]),
        Pipeline([('xgb-regressor', XGBRegressor())])
    ]

    # Hiperparâmetros para os modelos
    params = [
        #{},  # Linear Regression não precisa de hiperparâmetros para esta abordagem
        #{'Ridge__alpha': [0.01, 0.1, 1, 5, 10]},
        #{'Lasso__alpha': [0.01, 0.1, 1, 5, 10]},
        {
            'xgb-regressor__n_estimators': [200, 300, 400],
            'xgb-regressor__learning_rate': [0.05, 0.1, 0.15],
            'xgb-regressor__max_depth': [6, 8, 10],
            'xgb-regressor__min_child_weight': [3, 5, 7],
            'xgb-regressor__gamma': [0.3, 0.5, 0.7],
            'xgb-regressor__subsample': [0.6, 0.8, 1.0],
            'xgb-regressor__colsample_bytree': [0.4, 0.5, 0.6],
            'xgb-regressor__reg_alpha': [0, 0.5, 1],
            'xgb-regressor__reg_lambda': [0.5, 1, 1.5],
            'xgb-regressor__seed': [42]
        }
    ]


    # Executar o fluxo de treinamento
    pipTrain(
        exp_name=f"nyc-taxi-driver",
        models=models,
        params=params,
        metric="r2",
        n_splits=3,
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_test=y_test
    )
    print(f"Fluxo principal concluído para a data: {date}")

def menu():
    print("=" * 60)
    print("                 MLOps Workflow Setup                  ")
    print("=" * 60)
    print("            Experiment Tracking - Mlflow"               )
    print("-" * 60)
    print(f"{'Tracking Server:':<20} {'Yes, Local Server.'}")
    print(f"{'Backend Store:':<20} {'Azure PostgreSQL.'}")
    print(f"{'Artifacts Store:':<20} {'Azure Blob Storage.'}")
    print("=" * 60)
    print("      Workflow Orchestration Engine - Prefect"          )
    print("-" * 60)    
    print(f"{'Workflow Orchestration Server:':<2} {'Yes, Remote Server (Azure Virtual Machines).'}")
    print("=" * 60)

menu()

main_flow('2024-01-15')
