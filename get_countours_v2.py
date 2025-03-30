import cv2
import numpy as np
from typing import List, Dict, Tuple
from grade_contur import ContourEvaluationResult, evaluate_contour

previous_positions = []

class ContourTracker:
    def __init__(self, static_threshold: int = 4, refresh_interval: int = 1000):
        self.static_threshold = static_threshold
        self.refresh_interval = refresh_interval
        self.frame_counter = 0
        self.position_history: Dict[Tuple[int, int], int] = {}
        self.static_contours: Dict[Tuple[int, int], bool] = {}

    def is_centers_equal(self, center1: Tuple[int, int], center2: Tuple[int, int], tolerance: int = 5) -> bool:
        """Check if two centers are within tolerance distance of each other."""
        x1, y1 = center1
        x2, y2 = center2
        return abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance

    def refresh_tracker(self):
        """Reset all tracking data periodically to recover from false positives."""
        self.position_history.clear()
        self.static_contours.clear()
        self.frame_counter = 0
        # print("Tracker refreshed - all static markings cleared")

    def find_top_candidates(self, contours: List[np.ndarray], top_n: int = 10) -> List[ContourEvaluationResult]:
        # Increment frame counter
        self.frame_counter += 1

        # Check if we need to refresh
        if self.frame_counter >= self.refresh_interval:
            self.refresh_tracker()

        candidates = []
        positions_seen_this_frame = set()

        for contour in contours:
            result = evaluate_contour(contour, previous_positions)
            candidates.append(result)

            current_center = result.center
            positions_seen_this_frame.add(current_center)

            if current_center in self.static_contours:
                continue

            if current_center in self.position_history:
                self.position_history[current_center] += 1
                if self.position_history[current_center] >= self.static_threshold:
                    self.static_contours[current_center] = True
            else:
                self.position_history[current_center] = 1

        # Reset counters for positions not seen in this frame
        all_positions = set(self.position_history.keys())
        positions_to_reset = all_positions - positions_seen_this_frame
        for pos in positions_to_reset:
            # if pos not in self.static_contours:
            self.position_history[pos] = 0

        # Clean up positions with zero count
        self.position_history = {k: v for k, v in self.position_history.items() if v > 0}

        # Sort candidates by score and filter out static ones
        dynamic_candidates = [c for c in candidates if c.center not in self.static_contours]

        sorted_candidates = sorted(dynamic_candidates, key=lambda x: x.score, reverse=True)[:top_n]
        if sorted_candidates:
            previous_positions.append(sorted_candidates[0].center)

        return sorted_candidates

    def is_static(self, center: Tuple[int, int]) -> bool:
        """Check if a position is marked as static."""
        return center in self.static_contours

    def get_streak(self, center: Tuple[int, int]) -> int:
        """Get the current streak for a position."""
        return self.position_history.get(center, 0)

    def get_frame_count(self) -> int:
        """Get current frame count until next refresh."""
        return self.frame_counter

def get_contours(grayscale_image: np.ndarray) -> np.ndarray:
    """
    Detect contours in a grayscale image and return the output image with contours drawn on it.
    """
    # Detect contours in the mask
    contours, _ = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours
