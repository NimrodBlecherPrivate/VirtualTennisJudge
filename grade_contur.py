from dataclasses import dataclass
import cv2
import numpy as np
import math
from typing import Tuple, Optional


@dataclass
class ContourEvaluationResult:
    score: float
    center: Tuple[int, int]
    radius: float
    circularity: float
    distance: float
    velocity: Optional[float]
    area: float

    def __eq__(self, other):
        if not isinstance(other, ContourEvaluationResult):
            return False
        return self.center == other.center

    def __hash__(self):
        return hash(self.center)

    def __str__(self):
        # Helper function to format values or return 'N/A' if None
        def format_value(value, fmt=".2f"):
            return f"{value:{fmt}}" if value is not None else "N/A"

        # Use the helper function to format each attribute
        return (
            f"Score: {format_value(self.score)}, "
            f"Center: {self.center}, "
            f"Radius: {format_value(self.radius)}, "
            f"Circularity: {format_value(self.circularity)}, "
            f"Distance: {format_value(self.distance)}, "
            f"Area: {format_value(self.area)}"
        )


# Function to calculate the velocity of an object based on its previous and current positions
def calculate_velocity(
    prev_position: Optional[Tuple[int, int]],
    current_position: Tuple[int, int],
    frames_diff: int,
) -> Optional[float]:
    """
    Calculate the velocity of the ball based on the previous and current positions.
    The velocity is computed as the Euclidean distance between the positions divided by the frames difference.

    Args:
        prev_position (Optional[Tuple[int, int]]): Previous position of the object (None if not available).
        current_position (Tuple[int, int]): Current position of the object.
        frames_diff (int): The difference in frames between the previous and current positions.

    Returns:
        Optional[float]: The calculated velocity, or None if no previous position exists.
    """
    if prev_position is None:
        return None  # No previous position, return None

    # Calculate the Euclidean distance between the previous and current positions
    distance = math.sqrt((current_position[0] - prev_position[0]) ** 2 + (current_position[1] - prev_position[1]) ** 2)

    # Calculate velocity as distance divided by the number of frames
    return distance / frames_diff


# Function to calculate the circularity of a contour
def calculate_circularity(contour: np.ndarray) -> float:
    """
    Calculate the circularity of a contour. The more circular the contour, the higher the score.
    Circularity is calculated as 4 * π * (Area / Perimeter^2).

    Args:
        contour (np.ndarray): The contour whose circularity is to be calculated.

    Returns:
        float: The circularity score of the contour.
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if perimeter == 0:  # Avoid division by zero
        return 0

    area_score = (4 * np.pi * (area / (perimeter**2))) * 1.5

    # Minimum Enclosing Circle (Method 2)

    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius**2)
    contour_area = cv2.contourArea(contour)
    circularity_ratio = (contour_area / circle_area) * 1.5

    # Bounding Rectangle Aspect Ratio (Method 3)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    circularity_grade = area_score * circularity_ratio * aspect_ratio

    return circularity_grade


# Function to calculate the Euclidean distance between the center of the current contour and the previous position
def calculate_distance(center: Tuple[int, int], prev_position: Optional[Tuple[int, int]]) -> float:
    """
    Calculate the Euclidean distance between the center of the current contour and the previous position.

    Args:
        center (Tuple[int, int]): The center of the current contour.
        prev_position (Optional[Tuple[int, int]]): The previous position of the object.

    Returns:
        float: The Euclidean distance, or a large number if no previous position exists.
    """
    if prev_position is None:
        return float("inf")  # No previous position, return a large number

    return math.sqrt((center[0] - prev_position[0]) ** 2 + (center[1] - prev_position[1]) ** 2)


def get_median_of_last_five_positions(positions: list[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Get the median of the last five positions to smooth the position prediction.

    Args:
        positions (list[Tuple[int, int]]): List of the last five positions.

    Returns:
        Tuple[int, int]: The median position of the last five positions.
    """
    if len(positions) < 5:
        if not positions:
            return None
        return positions[-1]  # Return the last position if there are less than five positions

    last_five = positions[-5:]
    x_values = [pos[0] for pos in last_five]
    y_values = [pos[1] for pos in last_five]

    return np.median(x_values), np.median(y_values)


# Function to evaluate the quality of a contour based on circularity, distance, and velocity
def evaluate_contour(
    contour: np.ndarray,
    previous_positions: list[Optional[Tuple[int, int]]] = [],
) -> ContourEvaluationResult:
    """
    Evaluate a contour by its circularity, distance from the previous position, and velocity.

    Args:
        contour (np.ndarray): The contour to be evaluated.
        prev_position (Optional[Tuple[int, int]]): The previous position of the object.

    Returns:
        ContourEvaluationResult: A dataclass containing the evaluation result including score, center, radius,
                                 circularity, distance, velocity, and area.
    """
    area = cv2.contourArea(contour)
    circularity = calculate_circularity(contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))

    prev_position, distance = None, None
    if previous_positions:
        distance = calculate_distance(center, previous_positions[-1]) if previous_positions else float("inf")
        if distance > 100 and not distance == float("inf"):
            prev_position = get_median_of_last_five_positions(previous_positions)
            # Calculate distance from the previous position
            distance = calculate_distance(center, prev_position)

    # Calculate velocity
    velocity = calculate_velocity(prev_position, center, 1)

    # Calculate distance score (inversely proportional to distance)
    if distance:
        distance_score = 1 / distance
        if 20 < distance < 90 and circularity > 1:
            distance_score *= 3
    else:
        distance_score = 0
    # Calculate final score: weighted sum of circularity and distance score
    score = 4 * circularity + 10 * distance_score

    # If the velocity is low, it might not be the ball (optional condition)
    # if velocity and 0.5 < velocity < 40:
    #     # print("Velocity is very low, probably not the ball")

    if (area < 300 and center[1] > 1300) or (area < 150 and center[1] > 1000) or (area < 50 and center[1] > 800):
        score /= 5

    if area < 30 or area > 1500:
        # print(f"Area {area} is not in the valid range")
        score /= 10

    return ContourEvaluationResult(score, center, radius, circularity, distance, velocity, area)
