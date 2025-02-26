# Use uma imagem Python base
FROM python:3.12.7-slim

# Instala dependências básicas do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ bookworm main" > /etc/apt/sources.list.d/azure-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends azure-cli \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Define o diretório de trabalho
WORKDIR /app  

# Instala pipenv e atualiza o pip
RUN pip install --no-cache-dir -U pip pipenv

# Copia apenas os arquivos necessários para instalar dependências
COPY ["Pipfile", "Pipfile.lock", "web_preds_mlflow_server.py", "./"]

# Remove qualquer cache antigo antes da instalação
RUN pipenv install --system --deploy --clear

# Expõe a porta do servidor
EXPOSE 9696  

# Usa Gunicorn para melhor desempenho
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "web_preds_mlflow_server:app", "--timeout", "120"]
