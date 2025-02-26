# syntax=docker/dockerfile:1

# Use uma imagem Python base
FROM python:3.12.7-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY app.py .

EXPOSE 8085

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8085", "app:app", "--timeout", "120"]
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8085"]
