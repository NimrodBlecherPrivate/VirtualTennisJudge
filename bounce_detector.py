import os
from typing import Tuple, List
import numpy as np
import catboost as ctb
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial import distance

MODEL = "ctb_regr_bounce.cbm"

debug_mode = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes", "on")


class BounceDetector:
    def __init__(self, path_model=None):
        self.model = ctb.CatBoostRegressor()
        self.threshold = 0.10
        if path_model:
            self.load_model(path_model)

    def load_model(self, path_model):
        self.model.load_model(path_model)

    def prepare_features(self, x_ball, y_ball):
        labels = pd.DataFrame({"frame": range(len(x_ball)), "x-coordinate": x_ball, "y-coordinate": y_ball})

        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels["x_lag_{}".format(i)] = labels["x-coordinate"].shift(i)
            labels["x_lag_inv_{}".format(i)] = labels["x-coordinate"].shift(-i)
            labels["y_lag_{}".format(i)] = labels["y-coordinate"].shift(i)
            labels["y_lag_inv_{}".format(i)] = labels["y-coordinate"].shift(-i)
            labels["x_diff_{}".format(i)] = abs(labels["x_lag_{}".format(i)] - labels["x-coordinate"])
            labels["y_diff_{}".format(i)] = labels["y_lag_{}".format(i)] - labels["y-coordinate"]
            labels["x_diff_inv_{}".format(i)] = abs(labels["x_lag_inv_{}".format(i)] - labels["x-coordinate"])
            labels["y_diff_inv_{}".format(i)] = labels["y_lag_inv_{}".format(i)] - labels["y-coordinate"]
            labels["x_div_{}".format(i)] = abs(
                labels["x_diff_{}".format(i)] / (labels["x_diff_inv_{}".format(i)] + eps)
            )
            labels["y_div_{}".format(i)] = labels["y_diff_{}".format(i)] / (labels["y_diff_inv_{}".format(i)] + eps)

        for i in range(1, num):
            labels = labels[labels["x_lag_{}".format(i)].notna()]
            labels = labels[labels["x_lag_inv_{}".format(i)].notna()]
        labels = labels[labels["x-coordinate"].notna()]

        colnames_x = (
            ["x_diff_{}".format(i) for i in range(1, num)]
            + ["x_diff_inv_{}".format(i) for i in range(1, num)]
            + ["x_div_{}".format(i) for i in range(1, num)]
        )
        colnames_y = (
            ["y_diff_{}".format(i) for i in range(1, num)]
            + ["y_diff_inv_{}".format(i) for i in range(1, num)]
            + ["y_div_{}".format(i) for i in range(1, num)]
        )
        colnames = colnames_x + colnames_y

        features = labels[colnames]
        return features, list(labels["frame"])

    def predict(self, x_ball, y_ball, smooth=True):
        if smooth:
            x_ball, y_ball = self.smooth_predictions(x_ball, y_ball)
        features, num_frames = self.prepare_features(x_ball, y_ball)
        preds = self.model.predict(features)
        ind_bounce = np.where(preds > self.threshold)[0]
        if len(ind_bounce) > 0:
            ind_bounce = self.postprocess(ind_bounce, preds)
        frames_bounce = [num_frames[x] for x in ind_bounce]
        return frames_bounce

    def smooth_predictions(self, x_ball, y_ball):
        # We need to clear out mishits in our ball detection
        # Therefor, in any chunk of 5 frames, if the middle frame x coordinate,
        # is very far from the average of the 4 frames before and after it, we need to replace it with None
        # Then, we need to interpolate the None values

        # First, detect outliers in chunks of 5 frames
        # window_size = 2  # 2 frames before and after (total of 5 frames per window)
        # outlier_threshold = 400  # Pixels distance threshold for determining outliers

        # # Create copies to avoid modifying the originals during the iteration
        # x_ball_copy = x_ball.copy()
        # y_ball_copy = y_ball.copy()

        # # Check each frame that has enough frames before and after it
        # for i in range(window_size, len(x_ball) - window_size):
        #     # Skip frames that are already None
        #     if x_ball[i] is None or y_ball[i] is None:
        #         continue

        #     # Get surrounding frames (exclude the current frame)
        #     surrounding_x = [x_ball[j] for j in range(i-window_size, i+window_size+1) if j != i and x_ball[j] is not None]
        #     surrounding_y = [y_ball[j] for j in range(i-window_size, i+window_size+1) if j != i and y_ball[j] is not None]

        #     # Skip if we don't have enough surrounding frames for comparison
        #     if len(surrounding_x) < 2 or len(surrounding_y) < 2:
        #         continue

        #     # Calculate average position of surrounding frames
        #     avg_x = sum(surrounding_x) / len(surrounding_x)
        #     avg_y = sum(surrounding_y) / len(surrounding_y)

        #     # Calculate distance from current frame to average position
        #     current_dist = distance.euclidean((x_ball[i], y_ball[i]), (avg_x, avg_y))

        #     # If the distance exceeds threshold, mark as None (outlier)
        #     if current_dist > outlier_threshold:
        #         x_ball_copy[i] = None
        #         y_ball_copy[i] = None

        # # Update the original arrays with the outlier-removed versions
        # x_ball = x_ball_copy
        # y_ball = y_ball_copy

        # Now interpolate missing values
        is_none = [int(x is None) for x in x_ball]
        interp = 3
        counter = 0
        for num in range(interp, len(x_ball) - 1):
            if not x_ball[num] and sum(is_none[num - interp : num]) == 0 and counter < 3:
                x_ext, y_ext = self.extrapolate(x_ball[num - interp : num], y_ball[num - interp : num])
                x_ball[num] = x_ext
                y_ball[num] = y_ext
                is_none[num] = 0
                if x_ball[num + 1]:
                    dist = distance.euclidean((x_ext, y_ext), (x_ball[num + 1], y_ball[num + 1]))
                    if dist > 80:
                        x_ball[num + 1], y_ball[num + 1], is_none[num + 1] = None, None, 1
                counter += 1
            else:
                counter = 0
        return x_ball, y_ball

    def extrapolate(self, x_coords, y_coords):
        xs = list(range(len(x_coords)))
        func_x = CubicSpline(xs, x_coords, bc_type="natural")
        x_ext = func_x(len(x_coords))
        func_y = CubicSpline(xs, y_coords, bc_type="natural")
        y_ext = func_y(len(x_coords))
        return float(x_ext), float(y_ext)

    def postprocess(self, ind_bounce, preds):
        ind_bounce_filtered = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            if (ind_bounce[i] - ind_bounce[i - 1]) != 1:
                cur_ind = ind_bounce[i]
                ind_bounce_filtered.append(cur_ind)
            elif preds[ind_bounce[i]] > preds[ind_bounce[i - 1]]:
                ind_bounce_filtered[-1] = ind_bounce[i]
        return ind_bounce_filtered


class TrackingBallCoordinates:
    """Saved (x,y) of the ball in order to predict the bounce"""

    def __init__(self, path_model: str = MODEL) -> None:
        self.Detector = BounceDetector(path_model)
        self.x_y_to_frame = {}
        self.new_to_old_index = {}
        self.clear_coordinates()

    def add_coordinates(self, coordinates: Tuple[float, float], frame_idx: int) -> None:
        """Add new ball coordinates and frame index to tracking history.

        Args:
            coordinates: Tuple of (x,y) ball coordinates
            frame_idx: Frame index for these coordinates
        """
        x, y = coordinates if coordinates else (None, None)
        self.x_ball.append(x)
        self.y_ball.append(y)
        self.frames_indexses.append(frame_idx)
        self.x_y_to_frame[(x, y)] = frame_idx

    def get_coordinates(self, frame_idx: int) -> Tuple[float, float]:
        """Get ball coordinates for a specific frame index.

        Args:
            frame_idx: Frame index to retrieve coordinates for

        Returns:
            Tuple of (x,y) coordinates for requested frame
        """
        return int(self.x_ball[frame_idx]), int(self.y_ball[frame_idx])

    def clear_coordinates(self) -> None:
        """Reset all coordinate tracking lists."""
        self.x_ball: List[float] = []
        self.y_ball: List[float] = []
        self.frames_indexses: List[int] = []


    def prepare_features(self):
        """
        Splits the x, y coordinates into lists, separated by (None, None).
        Example: [(1,1), (2,2), (None, None), (3,3), (4,4)] -> [[(1,1), (2,2)], [(3,3), (4,4)]]
        We need to reserve a map between new and old indexes, as we want to mark the bounce in frame X, not in the new index X
        """
        x_y_coords = list(zip(self.x_ball, self.y_ball))
        split_coords = []
        current_segment = []

        new_index = 0  # Track the new index after splitting

        if debug_mode:
            print(f"x, y coords {x_y_coords}")

        for old_index, coord in enumerate(x_y_coords):
            if coord == (None, None):
                if current_segment and len(current_segment) >= 3:  # Only append if segment has enough points
                    split_coords.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(coord)
                # Map the new index to the old index
                self.new_to_old_index[new_index] = old_index
                new_index += 1

        if current_segment and len(current_segment) >= 3:  # Append the last segment if it has enough points
            split_coords.append(current_segment)

        return split_coords

    # def is_coordiantes_close(
    #     self, coord1: Tuple[float, float], coord2: Tuple[float, float], threshold: float = 100
    # ) -> bool:
    #     """Check if two sets of coordinates are close to each other.

    #     Args:
    #         coord1: First set of coordinates
    #         coord2: Second set of coordinates
    #         threshold: Maximum distance allowed between coordinates

    #     Returns:
    #         bool: True if coordinates are close, False otherwise
    #     """
    #     return distance.euclidean(coord1, coord2) < threshold

    # def is_index_valid(self, index: int, threshold) -> bool:
    #     """Check if the suuroding 8 samples of this index are close to each other, meaning the samples determined this was a bounce are valid"""
    #     strikes = 0
    #     if index < 3 or index > len(self.x_ball) - 4:
    #         return False
    #     for i in range(index - 3, index + 4):
    #         if not self.is_coordiantes_close(
    #             (self.x_ball[index], self.y_ball[index]), (self.x_ball[i], self.y_ball[i])
    #         ):
    #             strikes += 1
    #     # return strikes < threshold
    #     return True

    def predict_bounce(self, window_size: int = 3, distance_threshold: float = 1000000.0) -> List[int]:
        """
        Predict bounce frames and filter out false positives by checking surrounding coordinates.

        Args:
            window_size: Number of frames to check before and after bounce
            distance_threshold: Maximum allowed average distance for valid bounce

        Returns:
            List[int]: Filtered list of frame indices where bounces were detected
        """
        # Get initial bounce predictions
        # bounce_predictions = []
        # feature_chunks = self.prepare_features()
        # if debug_mode:
        #     print(f"after preparation {feature_chunks}")

        # for chunk in feature_chunks:
        #     if not chunk or len(chunk) < 3:
        #         continue  # Skip empty chunks

        #     x_coords, y_coords = zip(*chunk)
        #     # Predict bounces for the current chunk of coordinates
        #     bounces = self.Detector.predict(x_coords, y_coords, smooth=False)
        #     bounce_predictions.extend(bounces)

        # frames_bounce  = [self.new_to_old_index[x] for x in bounce_predictions]
        # return frames_bounce
        return self.Detector.predict(self.x_ball, self.y_ball, smooth=True)

        # valid_bounces = []
        # strikes = 0

        # for bounce_frame in bounce_predictions:
        #     try:
        #         # Skip if bounce coordinates are None
        #         if self.x_ball[bounce_frame] is None or self.y_ball[bounce_frame] is None:
        #             continue

        #         bounce_coord = (self.x_ball[bounce_frame], self.y_ball[bounce_frame])

        #         if not self.is_index_valid(bounce_frame, 2):
        #             continue
        #         # Get valid coordinates before bounce
        #         pre_bounce_coords = []
        #         for i in range(bounce_frame - window_size, bounce_frame):
        #             if i >= 0 and i < len(self.x_ball) and self.x_ball[i] is not None and self.y_ball[i] is not None:
        #                 pre_bounce_coords.append((self.x_ball[i], self.y_ball[i]))
        #             else:
        #                 strikes += 1

        #         # Get valid coordinates after bounce
        #         post_bounce_coords = []
        #         for i in range(bounce_frame + 1, bounce_frame + window_size + 1):
        #             if i < len(self.x_ball) and self.x_ball[i] is not None and self.y_ball[i] is not None:
        #                 post_bounce_coords.append((self.x_ball[i], self.y_ball[i]))
        #             else:
        #                 strikes += 1

        #         if strikes > 2:
        #             # logging.debug(f"Skipping frame {bounce_frame}: too many missing coordinates")
        #             strikes = 0
        #             continue

        #         # Skip if not enough valid coordinates
        #         if len(pre_bounce_coords) < 2 or len(post_bounce_coords) < 2:
        #             # logging.debug(f"Skipping frame {bounce_frame}: insufficient valid coordinates")
        #             continue

        #         # Calculate distances for valid coordinates
        #         pre_bounce_distances = [distance.euclidean(coord, bounce_coord) for coord in pre_bounce_coords]

        #         post_bounce_distances = [distance.euclidean(coord, bounce_coord) for coord in post_bounce_coords]

        #         # Calculate average distance
        #         avg_pre_distance = sum(pre_bounce_distances) / len(pre_bounce_distances)
        #         avg_post_distance = sum(post_bounce_distances) / len(post_bounce_distances)
        #         avg_distance = (avg_pre_distance + avg_post_distance) / 2

        #         # If average distance is within threshold, consider it a valid bounce
        #         if avg_distance < distance_threshold:
        #             valid_bounces.append(bounce_frame)
        #             # logging.debug(f"Valid bounce at frame {bounce_frame} with avg distance {avg_distance:.2f}")

        #     except Exception as e:
        #         # logging.warning(f"Error processing bounce frame {bounce_frame}: {e}")
        #         continue

        # return valid_bounces


# test the class
if __name__ == "__main__":
    bounce_detector = TrackingBallCoordinates()

    # read centers.txt, choose specific frames out of there, and test the model
    # format of centers.txt: frame_idx, x, y

    # range_to_test = range(170, 180)
    # frames_to_test = []
    # with open("centers.txt", "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         frame_idx, x, y = line.split(",")
    #         if int(frame_idx) in range_to_test:
    #             bounce_detector.add_coordinates((int(x), int(y)), int(frame_idx))
    bounce_detector.add_coordinates((2025, 1350), 0)
    bounce_detector.add_coordinates((2031, 1434), 1)
    bounce_detector.add_coordinates((2037, 1500), 2)
    bounce_detector.add_coordinates((2042, 1590), 3)
    bounce_detector.add_coordinates((2045, 1545), 4)
    bounce_detector.add_coordinates((2047, 1502), 5)
    bounce_detector.add_coordinates((2050, 1459), 6)
    bounce_detector.add_coordinates((2052, 1416), 7)

    print(bounce_detector.x_ball)
    print(bounce_detector.y_ball)
    print(bounce_detector.predict_bounce())
