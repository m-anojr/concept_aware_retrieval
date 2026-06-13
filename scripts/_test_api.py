import urllib.request
import urllib.parse
import json
import time

API_URL = "http://127.0.0.1:8000/api/search"

def test_search(query):
    print(f"\n--- Testing query: '{query}' ---")
    url = f"{API_URL}?q={urllib.parse.quote(query)}&top_k=3"
    try:
        t0 = time.time()
        # Increased timeout to 60 seconds for the first request
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode())
            t1 = time.time()
            
            if "error" in data and data["error"]:
                print(f"Error returned by API: {data['error']}")
                return False
                
            results = data.get("results", [])
            print(f"Search completed in {t1-t0:.2f}s. Found {len(results)} results.")
            
            for i, r in enumerate(results):
                print(f"[{i+1}] Score: {r.get('score', 0):.3f} | Mode: {r.get('search_mode', 'N/A')}")
                print(f"    Video: {r.get('video_id', '')} ({r.get('start_time_str', '')} - {r.get('end_time_str', '')})")
                
                # Show snippets to verify content
                transcript = r.get('transcript_text', '')
                if transcript:
                    print(f"    Transcript: {transcript[:100]}...")
                
                ocr = r.get('ocr_text', '')
                if ocr:
                    print(f"    OCR: {ocr[:100]}...")
            return True
            
    except Exception as e:
        print(f"Request failed: {e}")
        return False

# Test 1: First request might be slow due to model loading
print("Starting API tests. First request might take up to 30-40s as models load...")
test_search("binary search invariant")

# Test 2: Second request should be fast
test_search("GCD algorithm Euclid")

# Test 3: Test from the second video
test_search("queue data structure")

print("\nTesting complete.")
