import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stage2_retrieval.retrieval import LectureRetriever

print("Loading retriever...")
retriever = LectureRetriever()

queries = ["binary search invariant", "GCD algorithm Euclid", "queue data structure", "er model attributes"]

for q in queries:
    print(f"\n--- Query: '{q}' ---")
    results = retriever.search(q, top_k=3)
    for i, r in enumerate(results):
        print(f"[{i+1}] Score: {r['score']:.3f} | Mode: {r['search_mode']}")
        print(f"    Video: {r['video_id']} ({r['start_time_str']} - {r['end_time_str']})")
        print(f"    Transcript: {r['transcript_text'][:100]}...")
