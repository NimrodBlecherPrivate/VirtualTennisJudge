from typing import List
import cv2
import os
from process_frames import run as process_frames
from realtime_VideoStreamer import VideoStreamer as Video
import logging
from mp4_to_frames import extract_frames
from tools import printer


def cleanup_folders(folders_paths: List[str]) -> None:
    """
    Clean up all files in the specified folders.

    Args:
        folders_paths: List of folder paths to clean up
    """
    # return
    # Input validation
    if not isinstance(folders_paths, list):
        raise TypeError("folders_paths must be a list of strings")

    if not all(isinstance(path, str) for path in folders_paths):
        raise TypeError("All folder paths must be strings")

    if os.path.exists("centers.txt"):
        os.remove("centers.txt")

    for folder_path in folders_paths:
        # Skip empty or whitespace-only paths
        if not folder_path or folder_path.isspace():
            logging.warning(f"Skipping invalid folder path: '{folder_path}'")
            continue

        # Create directory if it doesn't exist
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                logging.info(f"Created directory: {folder_path}")
            except OSError as e:
                logging.error(f"Error creating directory {folder_path}: {e}")
            continue

        # Clean existing files
        try:
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        try:
                            os.unlink(entry.path)
                            logging.debug(f"Deleted file: {entry.path}")
                        except OSError as e:
                            logging.error(f"Error deleting file {entry.path}: {e}")
        except OSError as e:
            logging.error(f"Error accessing directory {folder_path}: {e}")


class GameStarter:
    def __init__(self):
        self.folder_name = "frames"
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the application."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def capture_frames(self, video: Video) -> list:
        """Capture frames from the video stream."""
        frames = []
        frame_count = 0

        try:
            while True:
                frame = video.read()
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame_path = os.path.join(self.folder_name, f"{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame)
                frame_count += 1

        except KeyboardInterrupt:
            print("Frame capture stopped by user")
            logging.info("Frame capture stopped by user")
        except Exception as e:
            print(f"Error capturing frames: {e}")
            logging.error(f"Error capturing frames: {e}")
        finally:
            print(f"Nimrod... Captured {frame_count} frames")
            video.stop()

        return frames

    def catch_court_frame(self, video: Video):
        """Capture the court frame from the video stream."""
        ## Print a message to the user to capture the court frame, meaning leaving the video runnning and leaving the court empty, without
        ## any players or objects in it. No Ball, no players, no nothing.
        ## The user should press a key to capture the frame.
        ## The frame should be saved in the folder "court_frame.jpg"

        # Print message to user
        print("Capture the court frame. Press any key to capture.")
        print("Leave the court empty, without any players or objects in it.")
        print("No ball, no players, no nothing.\n\n")
        input("Press any key to capture the frame...")

        # Capture one single frame
        court_frame = video.read()
        cv2.imwrite("court_frame.jpg", court_frame)
        print("Court frame captured successfully.")

    def choose_game_mode(self):
        # A player can use a live mode, using a stream from twitch, or a video mode, using a video file.
        # The player will chose it aout of a list of options.
        # if the player chooses the live mode, the player will be asked to enter the twitch link.
        # if the player chooses the video mode, the player will be asked to enter the video file path.

        options = ["Live mode (Twitch stream)", "Video mode (Video file)", "Clean up all folders"]
        printer("Choose a game mode:", bold=True, italian=True, color="blue")
        for i, option in enumerate(options, start=1):
            printer(f"{i}. {option}", bold=False, italian=False, color="blue")
        # The user should enter the number of the chosen option.
        choice = input(
            printer(
                "Enter the number of the chosen option: ", bold=False, italian=False, color="blue", return_as_str=True
            )
        )
        # The user should be asked to enter the twitch link if the live mode is chosen.
        if choice == "1":
            self.run_live_mode()
        elif choice == "2":
            self.run_video_mode()
        elif choice == "3":
            cleanup_folders([self.folder_name, "try", "processed_frames", "bounce_frames"])
            exit()
        else:
            printer("Invalid choice. Please enter a valid number.", bold=False, italian=False, color="red")
            self.choose_game_mode()

    def run_live_mode(self):
        # The player will be asked to enter the twitch link.
        self.twitch_link = input(
            printer("Enter the Twitch link: ", bold=False, italian=False, color="blue", return_as_str=True)
        )
        video = Video(self.twitch_link)

        # catch court frame
        self.catch_court_frame(video)

        # Capture frames
        logging.info("Starting frame capture. Press Ctrl+C to stop.")
        frames = self.capture_frames(video)
        logging.info(f"Captured {len(frames)} frames")

    def run_video_mode(self):
        court_frame_path = input(
            printer("Enter the court frame path: ", bold=False, italian=False, color="blue", return_as_str=True)
        )
        # read and write into "court_frame.jpg"
        court = cv2.imread(court_frame_path)
        cv2.imwrite("court_frame.jpg", court)

        video_file_path = input(
            printer("Enter the video file path: ", bold=False, italian=False, color="blue", return_as_str=True)
        )

        extract_frames(video_file_path, self.folder_name)

        logging.info(f"Captured {len(os.listdir(self.folder_name))} frames")

    def start(self) -> None:
        """Main method to start the game processing."""
        try:
            # Initial cleanup
            printer("Starting game...", bold=True, italian=True, color="blue")
            should_clean = input(
                printer(
                    "Do you want to clean up all folders? (y/n): ",
                    bold=True,
                    italian=False,
                    color="red",
                    return_as_str=True,
                )
            )
            if should_clean and should_clean.lower() == "y":
                cleanup_folders([self.folder_name, "try", "processed_frames", "bounce_frames"])
            # Choose game mode
            self.choose_game_mode()

            # Process frames
            logging.info("Processing frames...")
            process_frames(
                self.folder_name, "bounce_frames", "processed_frames", "output.mp4", "court_frame.jpg", "corners"
            )

            # Restart game
            logging.info("Restarting game...")
            self.start()
        except KeyboardInterrupt:
            logging.info("Game stopped by user")
            # Final cleanup
            # cleanup_folders([self.folder_name, "try", "processed_frames", "bounce_frames"])
        except Exception as e:
            logging.error(f"Error in game execution: {e}")
            # Final cleanup
            # cleanup_folders([self.folder_name, "try", "processed_frames", "bounce_frames"])

            raise


def main():
    game = GameStarter()
    game.start()


if __name__ == "__main__":
    main()
