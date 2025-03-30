import os
import cv2

debug_mode = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes", "on")


def convert_video(input_video: str, output_video: str):
    # Open the video file
    video = cv2.VideoCapture(input_video)

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define the video writer
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), isColor=False)

    threshold_value = 140  # Set the threshold value

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold
        _, thresholded_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

        # Write the frame to the output video
        out.write(thresholded_frame)

    # Release resources
    video.release()
    out.release()
    print("Threshold video created successfully!")


# Function to extract frames from a video and save them to a directory
def extract_frames(video_path, output_dir, frame_interval=1):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    print(f"Extracting frames from {video_path}")
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_number = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        
        # cv2.ROTATE_90_CLOCKWISE
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if not ret:
            break  # Exit if there are no frames left to read

        # Save frame if it matches the interval
        if frame_number % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"{frame_number}.jpg")
            cv2.imwrite(frame_filename, frame)
            if debug_mode:
                print(f"Saved frame {frame_number}")

        frame_number += 1

    video_capture.release()
