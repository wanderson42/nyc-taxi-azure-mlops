#!/bin/bash

# Carregar variáveis de ambiente do arquivo .env
source .env

# Iniciar o servidor MLflow
mlflow server \
    --host $MLFLOW_SERVER_HOST \
    --port $MLFLOW_SERVER_PORT \
    --backend-store-uri "postgresql://$MLFLOW_DB_USERNAME:$MLFLOW_DB_PASSWORD@$MLFLOW_DB_HOSTNAME/$MLFLOW_DB_DBNAME" \
    --default-artifact-root "$MLFLOW_ARTIFACT_STORE_URI" \

#MLFLOW_TRACKING_URI=https://191.235.81.94:5000  # Replace with remote host name or IP address in an actual environment
