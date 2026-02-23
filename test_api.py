import requests

url = "http://127.0.0.1:5000/recommend/api"
data = {
    "shipping_category": "Domestic",
    "fragile": 1,
    "weight": 10,
    "volume": 2
}

response = requests.post(url, json=data)
print(response.json())
