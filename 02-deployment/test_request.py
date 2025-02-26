# Only test. It's not necessay for serving in production scenarios 

import requests # Biblioteca que permite o uso do método POST

trip = {
    "PULocationID": 65,
    "DOLocationID": 170,
    "trip_distance": 8.54
}

url = 'http://localhost:9696/predict' # A rota utilizada para previsões

response = requests.post(url, json=trip) #  POST os dados da corrida de taxi no formato json

print(response.json()) # Obtêm-se a resposta do web service
