import mlflow

TRACKING_SERVER_HOST = "http://191.235.81.94:5000"
mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
mlflow.set_registry_uri(TRACKING_SERVER_HOST)

client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_SERVER_HOST)

model_registered = "nyc-taxi-model-prod"
model_alias = "champion"

model_version = client.get_model_version_by_alias(model_registered, model_alias)
model_id = model_version.run_id
# Obtém os detalhes da execução pelo run_id
run_info = client.get_run(model_id)
# Obtém o nome do modelo a partir das tags do experimento (caso tenha sido registrado como artefato)
model_name = run_info.data.tags.get("mlflow.runName", "Nome do modelo não encontrado")
RUN_ID = f"{model_name} (id: {model_id})"

print(RUN_ID)