import cProfile
import os
import pstats
import re
import time
import multiprocessing
import cv2
import process_frame as pf
import tools
import numpy as np
from get_countours_v2 import ContourTracker
import concurrent.futures
from bounce_detector import TrackingBallCoordinates
from court_detection import InOrOut, Results

debug_mode = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes", "on")

# Function to sort filenames numerically
def sort_numerically(filenames):
    res = sorted(filenames, key=lambda x: int(re.search(r"\d+", x).group()))
    print(res)
    return res


def load_frames_in_batches(frames_path, batch_size=100, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    frames_nd_array = []
    for i in range(0, len(frames_path), batch_size):
        batch = frames_path[i : i + batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_frames = list(executor.map(tools.load_image, batch))
            frames_nd_array.extend(batch_frames)

    return frames_nd_array


def load_frames_with_multiprocessing(frames_path, batch_size=None):
    if batch_size is None:
        batch_size = len(frames_path)

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        frames_nd_array = pool.map(tools.load_image, frames_path, chunksize=batch_size)

    return frames_nd_array


# Absolute minimal parallelism
def load_frames_minimal_parallel(frames_path):
    with multiprocessing.Pool(processes=3) as pool:
        frames_nd_array = pool.map(tools.load_image, frames_path)
    return frames_nd_array


def proccess_frames(
    frames_dir: str,
    processed_frames_dir: str,
    tracker: ContourTracker,
    court: np.ndarray,
    video_out: cv2.VideoWriter,
    detector: TrackingBallCoordinates,
    debug: bool = False,
) -> None:
    """
    Load a list of frames from the specified directory path.

    Args:
        frames_path (str): Path to the directory containing the frames.

    Returns:
        List[np.ndarray]: List of loaded frames.
    """
    frames = sort_numerically(os.listdir(frames_dir))
    frames_path = [os.path.join(frames_dir, frame) for frame in frames]
    if debug_mode:
        print(f"frames_path: {frames_path}")

    for index, frame in enumerate(frames_path):
        if debug_mode:
            print(f"Processing frame {index}")
        coordinates = pf.proccess_frame(
            tools.load_image(frame),
            index,
            processed_frames=[],
            debug=debug,
            output_path=f"{processed_frames_dir}/frame{index}",
            tracker=tracker,
            court=court,
            video_out=video_out,
        )
        detector.add_coordinates(coordinates, index)
    # do tools.save_image(frame, proccessed_frames_dir) for each frame in proccessed_frames, using a thread pool
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.map(tools.save_image, proccessed_frames, [proccess_frames_dir] * len(proccessed_frames))


def get_video_creator(court: str, output_path: str) -> None:
    """
    Create a video from a list of frames.

    Args:
        frames_dir (str): Path to the directory containing the frames.
        output_path (str): Path to save the output video.
    """
    court = tools.load_image(court)
    height, width, _ = court.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 60, (width, height))

    return out


def add_banner_to_frame(bounce_frame, result):
    # Set rectangle parameters
    rect_color = (0, 0, 0)  # Black rectangle

    def add_banner(bounce_frame: np.ndarray, result: Results) -> np.ndarray:
        """Add colored banner with IN/OUT text to frame."""
        text_color = (0, 255, 0) if result != Results.OUT else (0, 0, 255)  # Green for IN, Red otherwise
        rect_coords = ((5, 5), (250, 60))  # Rectangle coordinates: (top_left, bottom_right)

        # Draw filled rectangle banner
        cv2.rectangle(bounce_frame, rect_coords[0], rect_coords[1], rect_color, thickness=-1)

        # Add text overlay
        text_params = {
            "text": result.value,
            "org": (15, 45),
            "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale": 1.5,
            "color": text_color,
            "thickness": 2,
        }
        cv2.putText(bounce_frame, **text_params)

        return bounce_frame

    return add_banner(bounce_frame, result)


def run(frames_dir, bounce_frames_dir, proccess_frames_dir, output_path, court_frame_path, corners_dir_path):
    print("Starting video processing...")
    # Load frames
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()

    # Initialize tracking objects
    tracker = ContourTracker()
    detector = TrackingBallCoordinates()
    in_or_out = InOrOut(court_frame_path=court_frame_path, corners_dir_path=corners_dir_path)
    video_out = get_video_creator(court_frame_path, output_path)

    court_grey = tools.load_image(court_frame_path, greyscale=True)
    # Process video frames
    proccess_frames(frames_dir, proccess_frames_dir, tracker, court_grey, video_out, detector, debug=True)

    # Analyze bounce frames
    for frame in detector.predict_bounce():
        # min_dist = np.inf
        x, y = detector.get_coordinates(frame)

        # # Check How far is the ball from the relevant lines, in order to not include obvious bounces
        # # distance_to_closest_line = in_or_out.get_minimum_distance(x, y)
        # # print(f"Distance to closest line: {distance_to_closest_line}")

        # # if distance_to_closest_line > 150:
        # #     print(f"Skipping frame {frame} due to ball being too far from the lines")
        # #     continue

        # Load and process bounce frame
        frame_path = os.path.join(frames_dir, f"{frame}.jpg")
        bounce_frame = tools.load_image(frame_path)
        # lines = in_or_out.lines_dict
        # for line_name, line in lines.items():
        #     x1, y1, x2, y2 = line
        #     cv2.line(bounce_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     # show the distance from the ball to the line
        #     distance = in_or_out.point_to_line_distance((x, y), (x1, y1), (x2, y2))
        #     min_dist = min(distance, min_dist)
        #     if debug_mode:
        #         print(f"Distance to {line_name}: {distance:.1f}")
        #         # literally draw the line from the ball location to the line
        #         cv2.line(bounce_frame, (x, y), (int((x1 + x2) / 2), int((y1 + y2) / 2)), (0, 0, 255), 2)
        # if distance > 1000:
        #     print('The ball bounce either deep inside the court or very far out, therefor skipping')
        #     continue
        if debug_mode:
            print(f"Frame {frame} includes a bounce, X: {x}, Y: {y}")
        # Add bounce markers
        bounce_text = f"Bounce, X: {x}, Y: {y}"
        cv2.putText(bounce_frame, bounce_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(bounce_frame, (x, y), 10, (0, 0, 255), 2)

        # Add IN/OUT banner
        result = in_or_out.determine_ball_position(x, y, 10)
        bounce_frame = add_banner_to_frame(bounce_frame, result)

        # Save processed frame
        tools.save_image(bounce_frame, os.path.join(bounce_frames_dir, f"frame{frame}.jpg"))

    # Cleanup and statistics
    detector.clear_coordinates()
    elapsed_time = time.time() - start_time
    print("Time in H/M/S: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)  # Top 20 time-consuming functions
