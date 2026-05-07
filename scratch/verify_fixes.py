import requests
import json
import sys

# Set encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

BASE_URL = "http://127.0.0.1:8000"

queries = [
    "What is the AUM of ICICI Prudential Top 100 Fund?",
    "What is the NAV of ICICI Prudential Infrastructure Fund?",
    "What is the AUM of ICICI Prudential Pharma Healthcare and Diagnostics Fund?"
]

print("Verifying fixes on port 8000...")
for q in queries:
    print(f"\nQuery: {q}")
    try:
        response = requests.post(f"{BASE_URL}/api/chat", json={"query": q})
        if response.status_code == 200:
            data = response.json()
            # Use repr to avoid encoding issues or use the codecs fix above
            print(f"Response: {data['text']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")
