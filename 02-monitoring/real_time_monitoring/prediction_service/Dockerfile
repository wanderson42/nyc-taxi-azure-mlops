# syntax=docker/dockerfile:1

# Use uma imagem Python base
FROM python:3.12.7-slim

# Instala dependências básicas do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ bookworm main" > /etc/apt/sources.list.d/azure-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends azure-cli \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

#RUN pip3 install evidently==0.2.8

COPY app.py .
COPY lin_reg.bin .

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app:app", "--timeout", "120"]

#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=9696"]
