import urllib.request
import urllib.parse
import json
import time

API_URL = "http://127.0.0.1:8000/api/search"

def test_search(query):
    print(f"\n--- Testing query: '{query}' ---")
    url = f"{API_URL}?q={urllib.parse.quote(query)}&top_k=2"
    try:
        t0 = time.time()
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode())
            t1 = time.time()
            results = data.get("results", [])
            print(f"Search completed in {t1-t0:.3f}s. Found {len(results)} results.")
    except Exception as e:
        print(f"Request failed: {e}")

print("Testing API latency...")
test_search("first query to load model")
test_search("second query")
test_search("third query")
test_search("fourth query")
