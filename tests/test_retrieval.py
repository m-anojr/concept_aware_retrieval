import unittest

from src.stage2_retrieval.retrieval import LectureRetriever, MIN_RELEVANCE_SCORE


class FinalizeResultsTests(unittest.TestCase):
    def test_discards_results_below_relevance_threshold(self):
        retriever = LectureRetriever.__new__(LectureRetriever)
        results = [
            {"score": MIN_RELEVANCE_SCORE - 0.01, "video_id": "1", "start_time": 10},
            {"score": MIN_RELEVANCE_SCORE, "video_id": "2", "start_time": 20},
        ]

        finalized = retriever._finalize_results(results, "/videos/{video_id}.mp4#t={start}")

        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["video_id"], "2")
        self.assertEqual(finalized[0]["jump_link"], "/videos/2.mp4#t=20")


if __name__ == "__main__":
    unittest.main()
