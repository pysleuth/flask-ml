import requests

json = {
    "cylinder": 8,
    "displacement": 300,
    "horsepower": 78,
    "weight": 3_000,
    "acceleration": 20,
    "year": 76,
    "origin": 1,
}

r = requests.post("http://localhost:5000/api", json=json)
if r.status_code == 200:
    print(f"Works: {r.text}")
else:
    print(f"Fail: {r.text}")