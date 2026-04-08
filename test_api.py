import requests

url = "http://127.0.0.1:5000/predict"
data = {"review": "This product is amazing and works perfectly!"}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())