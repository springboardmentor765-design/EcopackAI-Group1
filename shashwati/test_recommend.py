import requests

url = "http://127.0.0.1:5000/api/recommend"

payload = {
    "product_weight": 1.5,
    "volume_m3": 0.02,
    "fragile": 1,
    "shipping_category": "Domestic",
    "shelf_life_days": 180
}

response = requests.post(url, json=payload)
print(response.json())
