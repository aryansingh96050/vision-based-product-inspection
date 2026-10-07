import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def detect_defects(
    original_image: np.ndarray,
    gray_image: np.ndarray,
    main_contour: np.ndarray,
    object_mask: np.ndarray,
    min_defect_area: float = 15.0,
    scratch_sensitivity: int = 35,
    detect_edge_chips: bool = True
) -> List[Dict[str, Any]]:
    """Detects surface anomalies including scratches, holes, edge chips, and voids.

    Args:
        original_image: Color image (BGR).
        gray_image: Grayscale image.
        main_contour: Primary workpiece contour.
        object_mask: Binary mask of workpiece.
        min_defect_area: Minimum pixel area to register as a defect.
        scratch_sensitivity: Contrast sensitivity threshold for surface scratches.
        detect_edge_chips: Whether to analyze edge profile for missing regions / chips.

    Returns:
        List of defect dicts:
            [{
                'type': 'Scratch' | 'Small Hole' | 'Edge Chip / Missing Region' | 'Surface Irregularity',
                'bbox': (x, y, w, h),
                'area': float,
                'contour': np.ndarray,
                'confidence': float,
                'center': (cx, cy)
            }]
    """
    defects: List[Dict[str, Any]] = []

    # 1. Surface Inspection Zone (Erode mask to prevent false alarms at outer workpiece boundaries)
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded_surface_mask = cv2.erode(object_mask, erosion_kernel, iterations=1)

    # 2. Scratch & Surface Anomaly Detection using Morphological Black-Hat and Top-Hat
    # Black-hat isolates structures darker than surrounding metal (scratches, pits, cracks)
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, bh_kernel)

    # Top-hat isolates bright gouges / specular scratches
    tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, bh_kernel)

    # Combine surface responses
    surface_anomalies = cv2.add(blackhat, tophat)
    surface_anomalies = cv2.bitwise_and(surface_anomalies, surface_anomalies, mask=eroded_surface_mask)

    # Threshold anomalies using sensitivity cut-off
    _, thresh_anomalies = cv2.threshold(
        surface_anomalies, scratch_sensitivity, 255, cv2.THRESH_BINARY
    )

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_anomalies = cv2.morphologyEx(thresh_anomalies, cv2.MORPH_OPEN, clean_kernel)

    # Find anomaly contours
    anomaly_contours, _ = cv2.findContours(
        thresh_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in anomaly_contours:
        area = float(cv2.contourArea(cnt))
        if area >= min_defect_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            perimeter = float(cv2.arcLength(cnt, closed=True))
            aspect_ratio = max(bw, bh) / max(1, min(bw, bh))

            # Classify anomaly by geometric profile
            if aspect_ratio >= 2.0 or (perimeter > 0 and (perimeter * perimeter / max(1.0, area)) > 26):
                defect_type = "Scratch"
            elif aspect_ratio < 1.8 and perimeter > 0 and (4 * np.pi * area / (perimeter * perimeter)) > 0.40:
                defect_type = "Small Hole"
            else:
                defect_type = "Surface Irregularity"

            defects.append({
                "type": defect_type,
                "bbox": (int(bx), int(by), int(bw), int(bh)),
                "area": round(area, 1),
                "contour": cnt,
                "confidence": min(0.99, round(0.70 + (area / 200.0), 2)),
                "center": (int(bx + bw / 2), int(by + bh / 2))
            })

    # 3. Direct Internal Voids / Holes detection (via intensity deviation)
    masked_pixels = gray_image[eroded_surface_mask == 255]
    if len(masked_pixels) > 0:
        median_val = float(np.median(masked_pixels))
        dark_holes = (gray_image < (median_val - 45)) & (eroded_surface_mask == 255)
        dark_holes_u8 = dark_holes.astype(np.uint8) * 255
        dark_holes_u8 = cv2.morphologyEx(dark_holes_u8, cv2.MORPH_OPEN, clean_kernel)
        hole_cnts, _ = cv2.findContours(dark_holes_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for hcnt in hole_cnts:
            h_area = float(cv2.contourArea(hcnt))
            if h_area >= min_defect_area:
                hbx, hby, hbw, hbh = cv2.boundingRect(hcnt)
                # Check for overlap with already recorded defects
                overlap = False
                for d in defects:
                    dx, dy, dw, dh = d["bbox"]
                    if abs((hbx + hbw/2) - (dx + dw/2)) < 15 and abs((hby + hbh/2) - (dy + dh/2)) < 15:
                        overlap = True
                        break
                if not overlap:
                    defects.append({
                        "type": "Small Hole",
                        "bbox": (int(hbx), int(hby), int(hbw), int(hbh)),
                        "area": round(h_area, 1),
                        "contour": hcnt,
                        "confidence": 0.95,
                        "center": (int(hbx + hbw / 2), int(hby + hbh / 2))
                    })

    # 4. Missing Regions / Edge Chips / Unusual Edges
    if detect_edge_chips and len(main_contour) >= 4:
        # A. Detect concave edge indentations / notches (Difference between Convex Hull and Object Mask)
        hull = cv2.convexHull(main_contour)
        hull_mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

        missing_notches = cv2.subtract(hull_mask, object_mask)
        # Suppress single-pixel edge rasterization discretization
        missing_notches = cv2.morphologyEx(missing_notches, cv2.MORPH_OPEN, clean_kernel)
        notch_cnts, _ = cv2.findContours(missing_notches, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for ncnt in notch_cnts:
            n_area = float(cv2.contourArea(ncnt))
            if n_area >= min_defect_area:
                nx, ny, nw, nh = cv2.boundingRect(ncnt)
                # Ensure not already registered
                overlap = any(
                    abs((nx + nw/2) - (d["center"][0])) < 15 and abs((ny + nh/2) - (d["center"][1])) < 15
                    for d in defects
                )
                if not overlap:
                    defects.append({
                        "type": "Edge Chip / Missing Region",
                        "bbox": (int(nx), int(ny), int(nw), int(nh)),
                        "area": round(n_area, 1),
                        "contour": ncnt,
                        "confidence": 0.94,
                        "center": (int(nx + nw / 2), int(ny + nh / 2))
                    })

        # B. Check for broken corner or missing section against Oriented Bounding Box
        min_rect = cv2.minAreaRect(main_contour)
        box_pts = np.int32(cv2.boxPoints(min_rect))
        box_mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(box_mask, [box_pts], 0, 255, thickness=cv2.FILLED)

        missing_corners = cv2.subtract(box_mask, object_mask)
        missing_corners = cv2.morphologyEx(missing_corners, cv2.MORPH_OPEN, clean_kernel)
        corner_cnts, _ = cv2.findContours(missing_corners, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for ccnt in corner_cnts:
            c_area = float(cv2.contourArea(ccnt))
            # Corner missing area must be notable (e.g. >= 28 px) to avoid rounded corner false positives
            if c_area >= max(28.0, min_defect_area * 1.5):
                cx, cy, cw, ch = cv2.boundingRect(ccnt)
                overlap = any(
                    abs((cx + cw/2) - (d["center"][0])) < 18 and abs((cy + ch/2) - (d["center"][1])) < 18
                    for d in defects
                )
                if not overlap:
                    defects.append({
                        "type": "Edge Chip / Missing Region",
                        "bbox": (int(cx), int(cy), int(cw), int(ch)),
                        "area": round(c_area, 1),
                        "contour": ccnt,
                        "confidence": 0.91,
                        "center": (int(cx + cw / 2), int(cy + ch / 2))
                    })

    return defects

def annotate_inspection_result(
    image: np.ndarray,
    main_contour: np.ndarray,
    box_points: Optional[np.ndarray],
    measurements: Dict[str, float],
    defects: List[Dict[str, Any]],
    is_passed: bool,
    product_id: str = "PROD-SAMPLE",
    failure_reasons: Optional[List[str]] = None
) -> np.ndarray:
    """Draws clear visual annotations highlighting workpiece dimensions and defects.

    Args:
        image: Original input BGR image.
        main_contour: Detected workpiece contour.
        box_points: 4 corner points of oriented bounding box (4, 2).
        measurements: Measured dimension properties.
        defects: Detected defect list.
        is_passed: True if workpiece passed all criteria.
        product_id: Identifier label for workpiece.
        failure_reasons: Optional list of failure explanation strings.

    Returns:
        Annotated BGR image ready for rendering.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Color Palette (BGR)
    COLOR_PASS = (46, 204, 113)      # Emerald Green
    COLOR_FAIL = (41, 57, 231)      # High-visibility Coral Red
    COLOR_DEFECT = (30, 30, 240)    # Crimson Defect Box
    COLOR_DEFECT_FILL = (0, 140, 255) # Amber fill overlay
    COLOR_DIM = (255, 255, 255)     # White for dimension labels
    COLOR_BOX = (50, 160, 255)      # Orange for oriented bounding box

    # Status primary color
    status_color = COLOR_PASS if is_passed else COLOR_FAIL

    # 1. Draw workpiece boundary contour
    cv2.drawContours(annotated, [main_contour], -1, status_color, thickness=2, lineType=cv2.LINE_AA)

    # 2. Draw Oriented Bounding Box
    if box_points is not None:
        cv2.drawContours(annotated, [box_points], 0, COLOR_BOX, thickness=1, lineType=cv2.LINE_AA)

    # 3. Dimension Callout Annotations
    rx, ry, rw, rh = cv2.boundingRect(main_contour)
    w_val = measurements.get("width", float(rw))
    h_val = measurements.get("height", float(rh))

    # Width Dimension Line (along top)
    dim_y = max(18, ry - 10)
    cv2.line(annotated, (rx, dim_y), (rx + rw, dim_y), COLOR_DIM, 1, cv2.LINE_AA)
    cv2.line(annotated, (rx, dim_y - 4), (rx, dim_y + 4), COLOR_DIM, 1)
    cv2.line(annotated, (rx + rw, dim_y - 4), (rx + rw, dim_y + 4), COLOR_DIM, 1)
    w_label = f"W: {w_val:.1f}px"
    cv2.putText(
        annotated, w_label, (rx + max(5, int(rw / 2) - 30), max(14, dim_y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DIM, 1, cv2.LINE_AA
    )

    # Height Dimension Line (along left side)
    dim_x = max(18, rx - 10)
    cv2.line(annotated, (dim_x, ry), (dim_x, ry + rh), COLOR_DIM, 1, cv2.LINE_AA)
    cv2.line(annotated, (dim_x - 4, ry), (dim_x + 4, ry), COLOR_DIM, 1)
    cv2.line(annotated, (dim_x - 4, ry + rh), (dim_x + 4, ry + rh), COLOR_DIM, 1)
    h_label = f"H: {h_val:.1f}px"
    cv2.putText(
        annotated, h_label, (max(2, dim_x - 55), ry + int(rh / 2)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DIM, 1, cv2.LINE_AA
    )

    # 4. Highlight Detected Defects
    overlay = annotated.copy()
    for idx, d in enumerate(defects, start=1):
        dx, dy, dw, dh = d["bbox"]
        dtype = d["type"]
        darea = d["area"]

        # Fill defect contour on alpha overlay
        if "contour" in d and d["contour"] is not None:
            cv2.drawContours(overlay, [d["contour"]], -1, COLOR_DEFECT_FILL, thickness=cv2.FILLED)

        # Draw red bounding box
        cv2.rectangle(
            annotated, (dx - 2, dy - 2), (dx + dw + 2, dy + dh + 2),
            COLOR_DEFECT, 2, cv2.LINE_AA
        )

        # Defect tag badge
        tag = f"#{idx} {dtype} ({darea:.0f}px)"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tag_y = max(th + 4, dy - 6)
        tag_x = min(w - tw - 6, max(4, dx))

        cv2.rectangle(
            annotated,
            (tag_x - 2, tag_y - th - 3),
            (tag_x + tw + 4, tag_y + 3),
            (20, 20, 20),
            thickness=cv2.FILLED
        )
        cv2.rectangle(
            annotated,
            (tag_x - 2, tag_y - th - 3),
            (tag_x + tw + 4, tag_y + 3),
            COLOR_DEFECT,
            thickness=1
        )
        cv2.putText(
            annotated, tag, (tag_x, tag_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Blend overlay for defect regions
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    # 5. Top-Left Primary Status Banner
    banner_w = 230
    banner_h = 56
    cv2.rectangle(annotated, (10, 10), (10 + banner_w, 10 + banner_h), (25, 25, 30), thickness=cv2.FILLED)
    cv2.rectangle(annotated, (10, 10), (10 + banner_w, 10 + banner_h), status_color, thickness=2)

    status_text = "STATUS: PASS" if is_passed else "STATUS: FAIL"
    cv2.putText(
        annotated, status_text, (20, 34),
        cv2.FONT_HERSHEY_DUPLEX, 0.68, status_color, 2, cv2.LINE_AA
    )

    defect_info = f"Defects: {len(defects)} | ID: {product_id}"
    cv2.putText(
        annotated, defect_info, (20, 54),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA
    )

    return annotated
