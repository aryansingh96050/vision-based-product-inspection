import cv2
import numpy as np
import math
from typing import Dict, Any, Tuple, List, Optional

def compute_measurements(
    contour: np.ndarray,
    min_area_rect: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None
) -> Dict[str, float]:
    """Calculates geometric properties of the detected workpiece contour.

    Args:
        contour: Object contour points array.
        min_area_rect: Optional precomputed cv2.minAreaRect tuple.

    Returns:
        Dict containing:
            - 'width': Measured primary width in pixels (oriented)
            - 'height': Measured primary height in pixels (oriented)
            - 'straight_width': Axis-aligned bounding width
            - 'straight_height': Axis-aligned bounding height
            - 'area': Surface area in pixels²
            - 'perimeter': Outer boundary perimeter length in pixels
            - 'circularity': Shape circularity factor (0.0 to 1.0)
            - 'aspect_ratio': Ratio of width to height
            - 'solidity': Ratio of contour area to convex hull area
    """
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))

    # Circularity factor: 4 * pi * Area / (Perimeter^2)
    # Equals 1.0 for a perfect circle; lower for elongated or jagged shapes
    if perimeter > 0:
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        circularity = min(1.0, max(0.0, circularity))
    else:
        circularity = 0.0

    # Axis-aligned bounding box
    rx, ry, rw, rh = cv2.boundingRect(contour)

    # Oriented minimum-area rectangle to prevent orientation bias
    if min_area_rect is None:
        min_rect = cv2.minAreaRect(contour)
    else:
        min_rect = min_area_rect

    (cx, cy), (dim1, dim2), angle = min_rect
    # Normalize dimensions so width >= height by default convention for rectangular parts
    w_oriented = max(dim1, dim2)
    h_oriented = min(dim1, dim2)

    aspect_ratio = (w_oriented / h_oriented) if h_oriented > 0 else 0.0

    # Solidity: Area / Hull Area
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 1.0

    return {
        "width": round(float(w_oriented), 2),
        "height": round(float(h_oriented), 2),
        "straight_width": round(float(rw), 2),
        "straight_height": round(float(rh), 2),
        "area": round(float(area), 2),
        "perimeter": round(float(perimeter), 2),
        "circularity": round(float(circularity), 4),
        "aspect_ratio": round(float(aspect_ratio), 3),
        "solidity": round(float(solidity), 4),
        "angle": round(float(angle), 1)
    }

def evaluate_quality(
    measurements: Dict[str, float],
    expected_width: float,
    width_tolerance: float,
    expected_height: float,
    height_tolerance: float,
    min_circularity: float = 0.0
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Evaluates whether measured dimensions conform to engineering specifications.

    Rules:
        Expected width: W_exp ± tol_w
        Expected height: H_exp ± tol_h

    If measurements fall outside either tolerance range:
        Result = FAIL
    Otherwise:
        Result = PASS

    Returns:
        Tuple of:
            - is_pass (bool)
            - failure_reasons (List[str])
            - evaluation_metrics (Dict)
    """
    failures: List[str] = []

    actual_w = measurements.get("width", 0.0)
    actual_h = measurements.get("height", 0.0)
    actual_circ = measurements.get("circularity", 0.0)

    # Tolerances
    min_w = expected_width - width_tolerance
    max_w = expected_width + width_tolerance

    min_h = expected_height - height_tolerance
    max_h = expected_height + height_tolerance

    # Check Width
    w_delta = actual_w - expected_width
    if actual_w < min_w or actual_w > max_w:
        sign = "+" if w_delta > 0 else ""
        failures.append(
            f"Width deviation: Measured {actual_w} px outside spec [{min_w:.1f}, {max_w:.1f}] px ({sign}{w_delta:.1f} px)"
        )

    # Check Height
    h_delta = actual_h - expected_height
    if actual_h < min_h or actual_h > max_h:
        sign = "+" if h_delta > 0 else ""
        failures.append(
            f"Height deviation: Measured {actual_h} px outside spec [{min_h:.1f}, {max_h:.1f}] px ({sign}{h_delta:.1f} px)"
        )

    # Check Circularity if configured
    if min_circularity > 0.0 and actual_circ < min_circularity:
        failures.append(
            f"Circularity failure: Measured {actual_circ:.3f} below minimum {min_circularity:.3f}"
        )

    is_pass = (len(failures) == 0)

    eval_data = {
        "width_in_spec": min_w <= actual_w <= max_w,
        "height_in_spec": min_h <= actual_h <= max_h,
        "width_delta": round(w_delta, 2),
        "height_delta": round(h_delta, 2),
        "spec_width_range": (round(min_w, 1), round(max_w, 1)),
        "spec_height_range": (round(min_h, 1), round(max_h, 1)),
        "failures": failures
    }

    return is_pass, failures, eval_data
