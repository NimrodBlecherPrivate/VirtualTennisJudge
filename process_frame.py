import os
from typing import Any, List
import cv2
import numpy as np
import tools
import get_countours_v2 as gc
import cProfile
import pstats

debug_mode = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes", "on")
verbose = os.getenv("VERBOSE", "0").lower() in ("1", "true", "yes", "on")

colors = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "purple": (255, 0, 255),
    "orange": (0, 165, 255),
    "brown": (42, 42, 165),
    "pink": (147, 20, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "teal": (128, 128, 0),
    "lavender": (250, 230, 230),
}



"""
Trapezoid Point Adjustment Guide
===============================

Format: [x, y] where:
- x moves point horizontally (← 0 to width →)
- y moves point vertically   (↑ 0 to height ↓)

                          TOP
    [0.1*width, 0.35*height]    [0.9*width, 0.35*height]
                ●──────────────────────●
               ╱                        ╲
              ╱                          ╲
             ╱                            ╲
            ╱                              ╲
           ●────────────────────────────────●
    [0, 0.8*height]                  [width, 0.8*height]
                        BOTTOM

Quick Reference:
---------------
• To move a point UP:    Decrease y value (e.g., 0.8 → 0.7)
• To move a point DOWN:  Increase y value (e.g., 0.35 → 0.4)
• To move a point LEFT:  Decrease x value (e.g., 0.9 → 0.8)
• To move a point RIGHT: Increase x value (e.g., 0.1 → 0.2)

Common Adjustments:
-----------------
1. Make trapezoid wider at top:
   - Top left:  Decrease x (0.1 → 0.05)
   - Top right: Increase x (0.9 → 0.95)

2. Make trapezoid taller:
   - Top points: Decrease y (0.35 → 0.3)
   - Bottom points: Increase y (0.8 → 0.85)
   
"""


def cut_court(frame: np.ndarray) -> np.ndarray:
    """
    Mask out irrelevant areas of the tennis court frame:
    1. Create a trapezoid mask for the main court area
    2. Add black triangles in the top corners to remove other courts/benches

    Args:
        frame (np.ndarray): Input frame to mask

    Returns:
        np.ndarray: Masked frame with only relevant court area visible
    """
    height, width = frame.shape[:2]

    # Create a black mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Define trapezoid points for court area
    trapezoid_points = np.array(
        [
            [0, int(height * 0.95)],  # bottom left
            [int(width * 0.20), int(height * 0.6)],  # top left
            [int(width * 0.65), int(height * 0.6)],  # top right
            [width, int(height * 0.95)],  # bottom right
        ],
        dtype=np.int32,
    )

    # Instruction on how to change the trapezoid points, e.g  - to take top right corner a bit higher and more to the right

    # Fill trapezoid area with white
    if debug_mode:
        cv2.fillPoly(mask, [trapezoid_points], 255)

    # Define and fill top corner triangles with black (they're already black in the mask)
    # left_triangle = np.array(
    #     [
    #         [0, 0],  # top left corner
    #         [int(width * 0.3), 0],  # top right point
    #         [0, int(height * 0.4)],  # bottom left point
    #     ],
    #     dtype=np.int32,
    # )

    # right_triangle = np.array(
    #     [
    #         [width, 0],  # top right corner
    #         [int(width * 0.7), 0],  # top left point
    #         [width, int(height * 0.4)],  # bottom right point
    #     ],
    #     dtype=np.int32,
    # )

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Draw the boundaries for visualization (optional)
    # if debug_mode:
    #     cv2.polylines(masked_frame, [trapezoid_points], True, (255, 255, 255), 2)
    # cv2.fillPoly(masked_frame, [left_triangle], (0, 0, 0))
    # cv2.fillPoly(masked_frame, [right_triangle], (0, 0, 0))

    return masked_frame


def create_objects_binary_picture(
    frame: np.ndarray, frame_index: int, blur_kernel=(35, 35), threshold=30
) -> np.ndarray:
    """
    Identify regions where the color differs significantly from the background,
    while ignoring shadow regions (dark pixels).

    Args:
        frame (np.ndarray): Input BGR frame (image) as a NumPy array.
        blur_kernel (tuple): Kernel size for background smoothing (Gaussian blur).
        threshold (int): Color difference threshold.

    Returns:
        np.ndarray: Binary mask highlighting regions that differ from the background.
    """
    if frame is None:
        raise ValueError("Input frame is None")

    frame_height, frame_width = frame.shape[:2]

    # Process input frame
    blurred = cv2.GaussianBlur(frame, blur_kernel, 0)
    
    if debug_mode:
        cv2.imwrite(f"try/blurred_{frame_index}.jpg", blurred)
        
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_mask = cv2.inRange(gray_scale, 220, 255)

    if debug_mode:
        cv2.imwrite(f"try/gray_mask_{frame_index}.jpg", gray_mask)

    # Calculate color difference mask
    color_difference = cv2.absdiff(frame, blurred)
    _, binary_mask = cv2.threshold(
        cv2.cvtColor(color_difference, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY
    )
    final_mask = cv2.bitwise_and(binary_mask, gray_mask)

    if debug_mode:
        cv2.imwrite(f"try/final_mask_{frame_index}.jpg", final_mask)

    # Process court reference image
    court = cv2.resize(tools.load_image("court_frame/court.jpg"), (frame_width, frame_height))

    _, binary_court = cv2.threshold(cv2.cvtColor(court, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)

    if final_mask.shape != binary_court.shape:
        raise ValueError(f"Shape mismatch: final_mask {final_mask.shape} != binary_court {binary_court.shape}")

    # Remove court markings from mask
    if debug_mode:
        cv2.imwrite(f"try/binary_court_{frame_index}.jpg", binary_court)
    gray_scale_subbed = cv2.subtract(final_mask, binary_court)
    if debug_mode:
        cv2.imwrite(f"try/gray_scale_subbed_{frame_index}.jpg", gray_scale_subbed)
    return cv2.inRange(gray_scale_subbed, 200, 255)


def build_diff_from_court_image(frame: np.ndarray, court: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Identify regions where the color differs significantly from the background,
    while ignoring shadow regions (dark pixels).

    Args:
        frame: Input video frame as numpy array in BGR format
        court: Grayscale background court image
        threshold: Color difference threshold value (default=30)

    Returns:
        Binary mask highlighting regions that differ from the background.
    """
    court= cv2.resize(court, (frame.shape[1], frame.shape[0]))
    
    # Convert frame to grayscale
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create mask for very bright pixels (230-255)
    gray_mask = cv2.inRange(gray_scale, 220, 255)

    # Calculate absolute difference between frame and background
    color_difference = cv2.absdiff(gray_scale, court)
    
    # Create binary mask from color differences
    _, binary_mask = cv2.threshold(color_difference, threshold, 255, cv2.THRESH_BINARY)

    # Combine masks and subtract court background
    final_mask = cv2.bitwise_and(binary_mask, gray_mask)
    gray_scale_subbed = cv2.subtract(final_mask, court)

    if debug_mode and verbose:
        cv2.imwrite("try/gray_mask.jpg", gray_mask)
        cv2.imwrite("try/color_difference.jpg", color_difference)
        cv2.imwrite("try/binary_mask.jpg", binary_mask)
        cv2.imwrite("try/final_mask.jpg", final_mask)
        cv2.imwrite("try/gray_scale_subbed.jpg", gray_scale_subbed)

    return gray_scale_subbed


def remove_noise(filtered_frame: np.ndarray) -> np.ndarray:
    """
    Removes noise and fills in broken shapes in the given frame using morphological operations.
    """
    # Initial noise removal using small opening operation
    small_opening_kernel = np.ones((2, 2), np.uint8)
    filtered_frame = cv2.morphologyEx(filtered_frame, cv2.MORPH_OPEN, small_opening_kernel)

    # Fill small holes and connect nearby shapes using closing operation
    closing_kernel = np.ones((4, 4), np.uint8)
    filtered_frame = cv2.morphologyEx(filtered_frame, cv2.MORPH_CLOSE, closing_kernel)

    # Further refine by removing small vertical noise
    small_vertical_kernel = np.ones((2, 1), np.uint8)
    filtered_frame = cv2.morphologyEx(filtered_frame, cv2.MORPH_OPEN, small_vertical_kernel)

    # Further refine by removing small horizontal noise
    small_horizontal_kernel = np.ones((1, 2), np.uint8)
    filtered_frame = cv2.morphologyEx(filtered_frame, cv2.MORPH_OPEN, small_horizontal_kernel)

    # Additional dilation to fill in larger gaps
    dilation_kernel = np.ones((2, 2), np.uint8)
    filtered_frame = cv2.dilate(filtered_frame, dilation_kernel, iterations=1)

    return filtered_frame


def mark_contours(frame: np.ndarray, contours: List[Any], top: int) -> np.ndarray:
    # print top contours, each with a different color
    for i, top_contour in enumerate(contours[:top]):
        if debug_mode:
            print(top_contour)
        color = colors[list(colors.keys())[i]]
        # Use the origin frame, mark the top contours with a red circle, and the score above the circle
        cv2.circle(frame, top_contour.center, int(top_contour.radius), color, 2)
        cv2.putText(frame, str(top_contour.score), top_contour.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def proccess_frame(
    frame: np.ndarray = None,
    frame_idx: int = None,
    processed_frames: List = None,
    output_path: str = None,
    debug: bool = False,
    tracker: gc.ContourTracker = None,
    court: np.ndarray = None,
    video_out: cv2.VideoWriter = None,
):
    if tracker is None:
        print("Error: tracker is None")
        exit(1)

    #bla = create_objects_binary_picture(frame=frame, frame_index=frame_idx)

    frame = cut_court(frame)

    filtered_frame = build_diff_from_court_image(frame, court)
    

    if debug_mode:
        # save filtered frame for debugging, before morphological operations
        filtered_path = f"{output_path}_filtered_befor_morph.jpg"
        cv2.imwrite(filtered_path, filtered_frame)

    # # Calculate histogram
    # hist = cv2.calcHist([filtered_frame], [0], None, [256], [0, 256])

    # # Print basic statistics
    # print(f"Min value: {np.min(filtered_frame)}")
    # print(f"Max value: {np.max(filtered_frame)}")
    # print(f"Mean value: {np.mean(filtered_frame):.2f}")
    # print(f"Median value: {np.median(filtered_frame)}")

    # # Print counts in specific ranges
    # print(f"Pixels with value < 80: {np.sum(filtered_frame < 80)}")
    # print(f"Pixels with value 80-90: {np.sum((filtered_frame >= 80) & (filtered_frame <= 90))}")
    # print(f"Pixels with value > 90: {np.sum(filtered_frame > 90)}")

    # # Simple text histogram for key ranges
    # # Group values in ranges of 10 for readability
    # ranges = [(i, i+9) for i in range(0, 250, 10)]
    # ranges.append((250, 255))  # Last range

    # print("\nSimplified histogram:")
    # for start, end in ranges:
    #     count = np.sum((filtered_frame >= start) & (filtered_frame <= end))
    #     percentage = (count / filtered_frame.size) * 100
    #     bar = "#" * int(percentage * 2)  # Scale the bar length
    #     print(f"{start:3d}-{end:3d}: {bar} ({percentage:.2f}%, {count} pixels)")

    # # Apply adaptive threshold
    # binary = cv2.threshold(filtered_frame, 210, 255, cv2.THRESH_BINARY)[1]

    # # Use this binary image as a mask
    # filtered_frame[binary == 0] = 255
    # cv2.imwrite(f"{output_path}_filtered_n1.jpg", filtered_frame)

    filtered_frame = remove_noise(filtered_frame)

    contours = gc.get_contours(filtered_frame)

    # save filtered frame for debugging
    if debug_mode:
        filtered_path = f"{output_path}_filtered_final.jpg"
        cv2.imwrite(filtered_path, filtered_frame)

    top_cotours = tracker.find_top_candidates(contours, top_n=1)
    if not top_cotours:
        return None
    if debug_mode and top_cotours:
        with open("centers.txt", "a") as f:
            f.write(f"{frame_idx},{top_cotours[0].center[0]},{top_cotours[0].center[1]}\n")

    mark_contours(frame, top_cotours, top=1)

    if debug:
        cv2.imwrite(f"{output_path}.jpg", frame)

    if video_out is not None:
        video_out.write(frame)
    top_contour = top_cotours[0]
    if top_contour.area < 30 or top_contour.area > 1500 or top_contour.circularity < 0.75:
        return None
    res = top_contour.center if top_cotours else None
    return res


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    tracker = gc.ContourTracker()
    court = tools.load_image("court_frame/court2.jpg")
    court_grey = cv2.cvtColor(court, cv2.COLOR_BGR2GRAY)
    for i in list(range(545,570,1)):
        frame = tools.load_image(f"frames/{i}.jpg")
        proccess_frame(
            frame=frame,
            frame_idx=i,
            output_path=f"try/frame{i}.jpg",
            debug=True,
            tracker=tracker,
            court=court_grey,
        )
