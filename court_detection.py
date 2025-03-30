from enum import Enum
import os
from typing import Any, List
import cv2
import numpy as np

Line = tuple[int, int, int, int]

debug_mode = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes", "on")


# results Enum, in ,out, etc...
class Results(Enum):
    IN = "in"
    OUT = "out"
    SERVE_IN = "serve in"
    SERVE_OUT = "serve out"
    ERROR = "error"


class InOrOut:
    def __init__(self, court_frame_path, corners_dir_path):
        # Load the binary image and the template
        court_image = cv2.imread(court_frame_path, cv2.IMREAD_GRAYSCALE)

        # Mask out top 30% of the image
        court_image[: int(court_image.shape[0] * 0.3), :] = 0

        templates_dir = corners_dir_path
        # go thorugh all files in the directory and imread them
        templates = [
            (cv2.imread(f"{templates_dir}/{file}", cv2.IMREAD_GRAYSCALE), file) for file in os.listdir(templates_dir)
        ]

        corners = {}

        def tmpelate_matching_and_marking(template, template_name, image, output_image):
            # Ensure the images are binary (black and white only)
            _, image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
            _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)

            if debug_mode:
                cv2.imwrite("try/court_binary.jpg", image)

            # Perform template matching
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

            # Find the location of the best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc

            # Calculate the center of the match
            template_h, template_w = template.shape
            center_x = top_left[0] + template_w // 2
            center_y = top_left[1] + template_h // 2
            corners[template_name] = (center_x, center_y)
            # Mark the center of the match in red. use A big circle
            cv2.circle(output_image, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(
                output_image, template_name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

        output_image = cv2.cvtColor(court_image, cv2.COLOR_GRAY2BGR)
        for template, name in templates:
            tmpelate_matching_and_marking(template, name, court_image, output_image)

        cv2.imwrite("output.jpg", output_image)

        bottom_line = (corners["bottom_left_corner.png"], corners["bottom_right_corner.png"])
        bottom_line = (
            corners["bottom_left_corner.png"][0],
            corners["bottom_left_corner.png"][1],
            corners["bottom_right_corner.png"][0],
            corners["bottom_right_corner.png"][1],
        )
        top_line = (corners["upper_left_corner.png"], corners["upper_right_corner.png"])
        top_line = (
            corners["upper_left_corner.png"][0],
            corners["upper_left_corner.png"][1],
            corners["upper_right_corner.png"][0],
            corners["upper_right_corner.png"][1],
        )

        left_line = (corners["upper_left_corner.png"], corners["bottom_left_corner.png"])
        left_line = (
            corners["upper_left_corner.png"][0],
            corners["upper_left_corner.png"][1],
            corners["bottom_left_corner.png"][0],
            corners["bottom_left_corner.png"][1],
        )

        right_line = (corners["upper_right_corner.png"], corners["bottom_right_corner.png"])
        right_line = (
            corners["upper_right_corner.png"][0],
            corners["upper_right_corner.png"][1],
            corners["bottom_right_corner.png"][0],
            corners["bottom_right_corner.png"][1],
        )

        middle_point = (
            int((corners["bottom_left_corner.png"][0] + corners["bottom_right_corner.png"][0]) / 2),
            int((corners["upper_left_corner.png"][1] + corners["bottom_left_corner.png"][1]) / 2),
        )

        serve_line = (middle_point[0], middle_point[1], corners["t_point.png"][0], corners["t_point.png"][1])

        upper_line = (0, 400, court_image.shape[1], 400)

        lines_dict = {
            "top": top_line,
            "bottom": bottom_line,
            "left": left_line,
            "right": right_line,
            "serve": serve_line,
            "upper": upper_line,
        }

        self.lines_dict = lines_dict
        # Mark Corneres in bold  Green
        output_image = cv2.cvtColor(court_image, cv2.COLOR_GRAY2BGR)
        for name, point in lines_dict.items():
            cv2.circle(output_image, (point[0], point[1]), 5, (0, 255, 0), -1)
            cv2.putText(output_image, name, (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(output_image, (point[2], point[3]), 5, (0, 255, 0), -1)
            cv2.putText(output_image, name, (point[2], point[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.line(output_image, (point[0], point[1]), (point[2], point[3]), (0, 255, 0), 2)

    @staticmethod
    def calculate_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    @staticmethod
    def point_to_line_distance(point, line_start, line_end):
        # Convert to numpy arrays for easier calculation
        point = np.array(point)
        line_start = np.array(line_start)
        line_end = np.array(line_end)

        # Calculate perpendicular distance
        numerator = abs(np.cross(line_end - line_start, point - line_start))
        denominator = np.linalg.norm(line_end - line_start)
        return numerator / denominator

    @staticmethod
    def point_to_line_segment_distance(point: tuple, line_start: tuple, line_end: tuple) -> float:
        """
        Calculate the shortest distance from a point to a line segment.

        Args:
            point: (x, y) coordinates of the point
            line_start: (x1, y1) start point of line segment
            line_end: (x2, y2) end point of line segment
        """
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Calculate the length of the line segment squared
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

        if line_length_sq == 0:
            # Line segment is actually a point
            return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5

        # Calculate projection of point onto line
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))

        # Calculate nearest point on line segment
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        # Return distance to nearest point
        return ((x - proj_x) ** 2 + (y - proj_y) ** 2) ** 0.5

    def get_minimum_distance(self, x, y) -> float:
        """Calculate the distance from a point to the closest line."""
        distances = []
        for line in self.lines_dict.values():
            x1, y1, x2, y2 = line
            # Calculate the distance from the point to the line
            distance = self.point_to_line_segment_distance((x, y), (x1, y1), (x2, y2))
            distances.append(distance)

        if debug_mode:
            print("Distances: ")
            for line_name, distance in zip(self.lines_dict.keys(), distances):
                print(f"{line_name}: {distance}")
        return min(distances)

    def determine_ball_position(self, x_ball, y_ball, radius) -> Results:
        """
        Determine the position of the ball relative to court lines.

        Args:
            x_ball (float): X-coordinate of the ball's center.
            y_ball (float): Y-coordinate of the ball's center.
            radius (float): Radius of the ball.
            lines (dict): Dictionary with line identifiers as keys and line coordinates (x1, y1, x2, y2) as values.

        Returns:
            str: One of "in", "out", "serve in", "serve out", or "couples in".
        """

        def is_ball_between_lines_vertically(x, y, radius, line1, line2):
            """Check if the ball is between two lines."""
            if (line1[2] - line1[0]) != 0:
                m1 = (line1[3] - line1[1]) / (line1[2] - line1[0])
            else:
                x_min = min(line1[0], line2[2])
                x_max = max(line1[0], line2[2])
                return (x - radius) >= x_min and (x + radius) <= x_max
            if (line2[2] - line2[0]) != 0:
                m2 = (line2[3] - line2[1]) / (line2[2] - line2[0])
            else:
                x_min = min(line1[0], line2[2])
                x_max = max(line1[0], line2[2])
                return (x - radius) >= x_min and (x + radius) <= x_max
            y1_at_x = line1[1] + m1 * (x - line1[2])
            y2_at_x = line2[1] + m2 * (x - line2[2])
            y_min = min(y1_at_x, y2_at_x)
            y_max = max(y1_at_x, y2_at_x)
            return (y - radius) >= y_min and (y + radius) <= y_max

        def is_ball_between_lines_horizontically(x, y, radius, line1, line2):
            """Check if the ball is between two lines."""
            if (line1[2] - line1[0]) != 0:
                m1 = (line1[3] - line1[1]) / (line1[2] - line1[0])
            else:
                x_min = min(line1[0], line2[2])
                x_max = max(line1[0], line2[2])
                return (x - radius) >= x_min and (x + radius) <= x_max
            if (line2[2] - line2[0]) != 0:
                m2 = (line2[3] - line2[1]) / (line2[2] - line2[0])
            else:
                x_min = min(line1[0], line2[2])
                x_max = max(line1[0], line2[2])
                return (x - radius) >= x_min and (x + radius) <= x_max
            x1_at_y = line1[0] + (y - line1[1]) / m1
            x2_at_y = line2[0] + (y - line2[1]) / m2
            x_min = min(x1_at_y, x2_at_y)
            x_max = max(x1_at_y, x2_at_y)
            return (x - radius) >= x_min and (x + radius) <= x_max

        try:
            top_line = self.lines_dict["top"]
            bottom_line = self.lines_dict["bottom"]
            left_line = self.lines_dict["left"]
            right_line = self.lines_dict["right"]
            serve_line = self.lines_dict["serve"]
            upper_line = self.lines_dict["upper"]

            # Check for "serve in left"
            if is_ball_between_lines_vertically(
                x_ball, y_ball, radius, upper_line, top_line
            ) and is_ball_between_lines_horizontically(x_ball, y_ball, radius, left_line, serve_line):
                return Results.SERVE_IN
            # Check for "serve in right"
            if is_ball_between_lines_vertically(
                x_ball, y_ball, radius, upper_line, top_line
            ) and is_ball_between_lines_horizontically(x_ball, y_ball, radius, serve_line, right_line):
                return Results.SERVE_IN

            # Check for "in"
            if is_ball_between_lines_vertically(
                x_ball, y_ball, radius, upper_line, bottom_line
            ) and is_ball_between_lines_horizontically(x_ball, y_ball, radius, left_line, right_line):
                return Results.IN

            return Results.OUT
        except Exception as e:
            print(f"Error determining ball position: {e}")
            return Results.ERROR


if __name__ == "__main__":
    # I just need output.jpg, which is the court with corners marked
    in_or_out = InOrOut("court_frame.jpg", "corners")
    print(in_or_out.determine_ball_position(100, 100, 10))