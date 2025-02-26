## Deploying a model as a web-service

* Creating a virtual environment with Pipenv
* Creating a script for predictiong 
* Putting the script into a Flask app
* Packaging the app to Docker
* Its necessary activate the mlflow tracking server in azure virtual machine

```bash
pipenv requirements > requirements.txt
```

```bash
docker build -t trip-duration-preds-mlflow-tracking-service:v2 .
```

```bash
docker run -it --rm -p 9696:9696 --network=host   -v ~/.azure:/root/.azure   trip-duration-preds-mlflow-tracking-server:v1
```
