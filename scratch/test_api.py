
import requests
import json

url = "http://localhost:8000/api/chat"
data = {"query": "Expense Ratio of Axis Large Cap Fund"}
try:
    response = requests.post(url, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
