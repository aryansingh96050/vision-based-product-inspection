import cv2
import numpy as np
from typing import Dict, Any, Tuple

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts input image to single-channel 8-bit grayscale.

    Args:
        image: Source image as NumPy ndarray (BGR, RGB, or Grayscale).

    Returns:
        Grayscale 2D image (H, W).
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot convert empty or null image to grayscale.")

    if len(image.shape) == 2:
        return image.copy()
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            # BGRA to BGR first
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

def reduce_noise(
    gray_image: np.ndarray,
    blur_type: str = "gaussian",
    kernel_size: int = 5
) -> np.ndarray:
    """Applies noise filtering to suppress high-frequency sensory noise.

    Args:
        gray_image: Single-channel 8-bit grayscale image.
        blur_type: Filter type ('gaussian', 'median', 'bilateral').
        kernel_size: Kernel dimension (must be positive odd integer).

    Returns:
        Filtered grayscale image.
    """
    if kernel_size <= 1:
        return gray_image.copy()

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    if blur_type.lower() == "gaussian":
        return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigmaX=0)
    elif blur_type.lower() == "median":
        return cv2.medianBlur(gray_image, kernel_size)
    elif blur_type.lower() == "bilateral":
        return cv2.bilateralFilter(gray_image, d=kernel_size, sigmaColor=75, sigmaSpace=75)
    else:
        return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigmaX=0)

def apply_threshold(
    blurred_image: np.ndarray,
    method: str = "otsu",
    threshold_value: int = 127,
    auto_invert: bool = True,
    force_invert: bool = False
) -> Tuple[np.ndarray, float]:
    """Segments workpiece foreground from the background using thresholding.

    OpenCV contour detection expects foreground objects to be white (255)
    and background to be black (0).

    Args:
        blurred_image: Grayscale filtered image.
        method: 'otsu', 'binary', or 'adaptive'.
        threshold_value: Manual threshold cut-off (0-255).
        auto_invert: Automatically invert if the background is lighter than object.
        force_invert: Explicitly flip polarity.

    Returns:
        Tuple of (binary_image, calculated_or_applied_threshold).
    """
    thresh_type = cv2.THRESH_BINARY

    if method.lower() == "otsu":
        actual_thresh, binary = cv2.threshold(
            blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method.lower() == "adaptive":
        actual_thresh = -1.0
        binary = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Standard manual binary threshold
        actual_thresh = float(threshold_value)
        _, binary = cv2.threshold(
            blurred_image, threshold_value, 255, thresh_type
        )

    # Check background polarity: inspect boundary pixels (edges)
    if auto_invert:
        h, w = binary.shape[:2]
        border_pixels = np.concatenate([
            binary[0, :], binary[h - 1, :],
            binary[:, 0], binary[:, w - 1]
        ])
        # If border is mostly white (>50%), the background was white
        if np.mean(border_pixels) > 127:
            binary = cv2.bitwise_not(binary)

    if force_invert:
        binary = cv2.bitwise_not(binary)

    # Clean isolated speckles with gentle morphological open/close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary, actual_thresh

def detect_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    """Computes gradient boundary edges using Canny edge detector.

    Args:
        image: Grayscale or filtered image.
        low_threshold: Lower hysteresis threshold.
        high_threshold: Upper hysteresis threshold.

    Returns:
        Binary edge map (255 for edges, 0 otherwise).
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def run_preprocessing_pipeline(
    image: np.ndarray,
    blur_kernel: int = 5,
    blur_type: str = "gaussian",
    threshold_method: str = "otsu",
    threshold_val: int = 127,
    edge_low: int = 50,
    edge_high: int = 150,
    force_invert: bool = False
) -> Dict[str, Any]:
    """Executes the full sequential preprocessing pipeline, collecting intermediate stages.

    Returns:
        Dict mapping stage names to output ndarrays and metadata.
    """
    if image is None:
        raise ValueError("Received null image in preprocessing pipeline.")

    # 1. Original copy
    original = image.copy()

    # 2. Grayscale
    gray = convert_to_grayscale(original)

    # 3. Noise reduction
    blurred = reduce_noise(gray, blur_type=blur_type, kernel_size=blur_kernel)

    # 4. Thresholding
    thresholded, used_thresh = apply_threshold(
        blurred,
        method=threshold_method,
        threshold_value=threshold_val,
        auto_invert=True,
        force_invert=force_invert
    )

    # 5. Edge detection
    edges = detect_edges(blurred, low_threshold=edge_low, high_threshold=edge_high)

    return {
        "original": original,
        "grayscale": gray,
        "blurred": blurred,
        "thresholded": thresholded,
        "edges": edges,
        "threshold_value_used": used_thresh
    }
