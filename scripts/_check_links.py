import urllib.request
import urllib.parse
import json

API_URL = "http://127.0.0.1:8000/api/search"

def check_links():
    query = "binary search invariant"
    url = f"{API_URL}?q={urllib.parse.quote(query)}&top_k=2"
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            results = data.get("results", [])
            
            print(f"Results for '{query}':")
            for i, r in enumerate(results):
                video_id = r.get('video_id', '')
                jump_link = r.get('jump_link', '')
                print(f"\nResult {i+1}:")
                print(f"Video ID: {video_id}")
                print(f"Jump Link: {jump_link}")
                
    except Exception as e:
        print(f"Error: {e}")

check_links()
