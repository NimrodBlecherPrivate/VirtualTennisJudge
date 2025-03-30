import cv2
import numpy as np


def load_image(image_path: str, greyscale=False) -> np.ndarray:
    """
    Load an image from the specified file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a NumPy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    if greyscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to the specified file path.

    Args:
        image (np.ndarray): Image to save.
        output_path (str): Path to save the image.
    """
    cv2.imwrite(output_path, image)


def printer(msg: str, bold: bool, italian: bool, color: str, return_as_str: bool = False) -> None:
    """
    Print a message with the specified formatting.

    Args:
        msg (str): Message to print.
        bold (bool): Whether to print the message in bold.
        italian (bool): Whether to print the message in Italian.
        color (str): Color to print the message in.
    """
    colors_map = {
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "purple": 35,
        "cyan": 36,
    }
    color = colors_map.get(color, 37)
    if bold:
        msg = f"\033[1m{msg}\033[0m"
    if italian:
        msg = f"\033[3m{msg}\033[0m"
    msg = f"\033[{color}m{msg}\033[0m"

    if return_as_str:
        return msg
    print(msg)


# def convert_to_hsv(image: np.ndarray) -> np.ndarray:
#     """
#     Convert a BGR image to HSV color space.

#     Args:
#         image (np.ndarray): Input BGR image.

#     Returns:
#         np.ndarray: Image in HSV color space.
#     """
#     return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# def create_color_mask(hsv_image: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
#     """
#     Create a mask for pixels within the specified HSV color range.

#     Args:
#         hsv_image (np.ndarray): Image in HSV color space.
#         lower_bound (np.ndarray): Lower bound for HSV values.
#         upper_bound (np.ndarray): Upper bound for HSV values.

#     Returns:
#         np.ndarray: Binary mask of the specified color range.
#     """
#     return cv2.inRange(hsv_image, lower_bound, upper_bound)


# def calculate_hsv_range(
#     hsv_image: np.ndarray,
#     mask: np.ndarray,
#     hue_tolerance: int = 10,
#     sv_tolerance: int = 20,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Calculate the HSV range of the object using the provided mask, with added tolerance.

#     Args:
#         hsv_image (np.ndarray): HSV image of the object.
#         mask (np.ndarray): Binary mask of the object.
#         hue_tolerance (int, optional): Tolerance to add to the hue range. Defaults to 10.
#         sv_tolerance (int, optional): Tolerance to add to the saturation and value ranges. Defaults to 20.

#     Returns:
#         Tuple[np.ndarray, np.ndarray]: Lower and upper HSV bounds.
#     """
#     object_pixels = hsv_image[mask > 0]
#     lower_hsv = np.maximum(np.min(object_pixels, axis=0) - [hue_tolerance, sv_tolerance, sv_tolerance], 0)
#     upper_hsv = np.minimum(
#         np.max(object_pixels, axis=0) + [hue_tolerance, sv_tolerance, sv_tolerance],
#         [179, 255, 255],
#     )
#     return lower_hsv.astype(np.uint8), upper_hsv.astype(np.uint8)


# def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     """
#     Apply a binary mask to an image.

#     Args:
#         image (np.ndarray): Input image.
#         mask (np.ndarray): Binary mask.

#     Returns:
#         np.ndarray: Resulting image with the mask applied.
#     """
#     return cv2.bitwise_and(image, image, mask=mask)


# def create_ball_mask(
#     ball_image_path: str, hue_tolerance: int = 10, sv_tolerance: int = 20
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Create a mask and HSV range for the tennis ball using the ball-only image.

#     Args:
#         ball_image_path (str): Path to the ball-only image.
#         hue_tolerance (int, optional): Tolerance to add to the hue range. Defaults to 10.
#         sv_tolerance (int, optional): Tolerance to add to the saturation and value ranges. Defaults to 20.

#     Returns:
#         Tuple[np.ndarray, np.ndarray]: Lower and upper HSV bounds for the ball.
#     """
#     ball_image = load_image(ball_image_path)
#     ball_hsv = convert_to_hsv(ball_image)

#     # Define a lenient yellow-green HSV range
#     yellow_green_lower = np.array([18, 90, 90], dtype=np.uint8)
#     yellow_green_upper = np.array([55, 255, 255], dtype=np.uint8)

#     # Create a mask for the ball-only image
#     non_black_mask = create_color_mask(ball_hsv, yellow_green_lower, yellow_green_upper)

#     # Calculate the HSV range of the ball
#     return calculate_hsv_range(ball_hsv, non_black_mask, hue_tolerance, sv_tolerance)


# def get_filtered_frame(match_image: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> np.ndarray:
#     """
#     Filter the match image to isolate the tennis ball using the precomputed HSV range.

#     Args:
#         match_image (np.ndarray): Input match image.
#         lower_hsv (np.ndarray): Lower HSV bound for the ball.
#         upper_hsv (np.ndarray): Upper HSV bound for the ball.

#     Returns:
#         np.ndarray: Resulting image with the ball isolated.
#     """
#     match_hsv = convert_to_hsv(match_image)
#     match_mask = create_color_mask(match_hsv, lower_hsv, upper_hsv)
#     return apply_mask(match_image, match_mask)
