import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

class ObjectNotFoundError(Exception):
    """Raised when no suitable workpiece contour is detected in the image."""
    pass

def find_main_object(
    binary_image: np.ndarray,
    min_contour_area: float = 300.0
) -> Dict[str, Any]:
    """Identifies the dominant workpiece contour from a binary thresholded image.

    Args:
        binary_image: Single-channel 8-bit binary image where object is 255.
        min_contour_area: Minimum pixel area threshold to discard noise specks.

    Returns:
        Dict containing:
            - 'contour': Primary object contour (np.ndarray)
            - 'all_contours': List of all valid contours found
            - 'straight_bbox': (x, y, w, h)
            - 'min_area_rect': ((cx, cy), (w, h), angle)
            - 'box_points': 4 vertices of oriented bounding box (4, 2)
            - 'convex_hull': Convex hull of primary contour
            - 'mask': Binary mask corresponding to the primary object

    Raises:
        ObjectNotFoundError: If no contour meets the minimum area criterion.
    """
    if binary_image is None or binary_image.size == 0:
        raise ObjectNotFoundError("Input image for object detection is empty.")

    # Find external contours
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ObjectNotFoundError("No contours detected. Verify image lighting and threshold parameters.")

    # Filter out spurious noise
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]

    if not valid_contours:
        max_seen = max([cv2.contourArea(c) for c in contours], default=0.0)
        raise ObjectNotFoundError(
            f"No object contour exceeded minimum area of {min_contour_area} px² "
            f"(largest candidate was {round(max_seen, 1)} px²)."
        )

    # Primary object is the largest external contour by area
    main_contour = max(valid_contours, key=cv2.contourArea)

    # 1. Axis-aligned bounding box
    rx, ry, rw, rh = cv2.boundingRect(main_contour)

    # 2. Minimum-area rotated bounding box (eliminates part skew/rotation artifacts)
    min_rect = cv2.minAreaRect(main_contour)
    # Box points coordinates (4 vertices)
    box_pts = cv2.boxPoints(min_rect)
    box_pts = np.int32(box_pts)

    # 3. Convex hull
    hull = cv2.convexHull(main_contour)

    # 4. Generate isolated binary mask for downstream defect extraction
    mask = np.zeros(binary_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

    return {
        "contour": main_contour,
        "all_contours": valid_contours,
        "straight_bbox": (int(rx), int(ry), int(rw), int(rh)),
        "min_area_rect": min_rect,
        "box_points": box_pts,
        "convex_hull": hull,
        "mask": mask
    }

def create_object_mask(
    image_shape: Tuple[int, ...],
    contour: np.ndarray,
    fill_holes: bool = True
) -> np.ndarray:
    """Creates a binary mask for the given contour."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    thickness = cv2.FILLED if fill_holes else 1
    cv2.drawContours(mask, [contour], -1, 255, thickness=thickness)
    return mask
