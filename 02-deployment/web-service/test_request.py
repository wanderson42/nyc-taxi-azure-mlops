# Only test. It's not necessay for serving in production scenarios 

import requests

trip = {
    "PULocationID": 65,
    "DOLocationID": 170,
    "trip_distance": 6.54
}

url = 'http://localhost:9696/predict'

response = requests.post(url, json=trip)
print(response.json())
