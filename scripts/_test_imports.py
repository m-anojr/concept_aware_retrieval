import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction.feature_pipeline import build_feature_stream
from src.stage1_boundary.segment import segments_from_scores
from src.stage1_boundary.pseudo_labels import pseudo_boundaries_from_signals
print("All imports OK")
