import cv2
import numpy as np
import os
import math
from typing import Tuple

def create_base_canvas(width=400, height=300, bg_color=(235, 238, 240)):
    """Creates an inspection stage background with subtle lighting gradient."""
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
    gradient = np.tile(np.linspace(0.96, 1.04, width), (height, 1))
    for c in range(3):
        canvas[:, :, c] = np.clip(canvas[:, :, c] * gradient, 0, 255).astype(np.uint8)
    return canvas

def draw_metallic_workpiece(
    canvas: np.ndarray,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    angle_deg: float = 0.0,
    base_color: Tuple[int, int, int] = (90, 95, 100),
    bevel_color: Tuple[int, int, int] = (120, 125, 130)
) -> np.ndarray:
    """Draws a manufactured metallic bracket/part with bevels and texture."""
    h, w = canvas.shape[:2]
    part_layer = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    hw = width // 2
    hh = height // 2
    pts = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh]
    ], dtype=np.float32)

    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    rotated_pts = np.dot(pts, rot_mat.T)
    rotated_pts[:, 0] += center_x
    rotated_pts[:, 1] += center_y
    int_pts = np.int32(rotated_pts)

    cv2.fillPoly(mask, [int_pts], 255)
    cv2.fillPoly(part_layer, [int_pts], base_color)

    # Add brushed metal texture
    streak = np.random.normal(0, 4, (h, w)).astype(np.int16)
    part_layer_bgr = part_layer.astype(np.int16)
    for c in range(3):
        part_layer_bgr[:, :, c] = np.clip(part_layer_bgr[:, :, c] + streak, 0, 255)
    part_layer = part_layer_bgr.astype(np.uint8)

    # Chamfer border
    cv2.polylines(part_layer, [int_pts], isClosed=True, color=bevel_color, thickness=2, lineType=cv2.LINE_AA)

    # Drop shadow
    shadow_mask = np.zeros((h, w), dtype=np.uint8)
    shadow_pts = int_pts + np.array([4, 4])
    cv2.fillPoly(shadow_mask, [shadow_pts], 180)
    shadow_blur = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    for c in range(3):
        canvas[:, :, c] = np.clip(canvas[:, :, c].astype(np.int16) - (shadow_blur // 4), 0, 255).astype(np.uint8)

    # Composite onto canvas
    idx = (mask == 255)
    canvas[idx] = part_layer[idx]

    return int_pts

def generate_sample_suite(output_dir: str):
    """Generates the suite of synthetic test images for the inspection system."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Nominal Compliant Part (Target: W=100px, H=60px -> PASS)
    c1 = create_base_canvas()
    draw_metallic_workpiece(c1, center_x=200, center_y=150, width=100, height=60, angle_deg=0)
    cv2.imwrite(os.path.join(output_dir, "sample_01_perfect_bracket.png"), c1)

    # 2. Oversized Width Defect (W=116px, outside 100±5px -> FAIL)
    c2 = create_base_canvas()
    draw_metallic_workpiece(c2, center_x=200, center_y=150, width=116, height=60, angle_deg=0)
    cv2.imwrite(os.path.join(output_dir, "sample_02_oversized_width.png"), c2)

    # 3. Undersized Height Defect (H=48px, outside 60±5px -> FAIL)
    c3 = create_base_canvas()
    draw_metallic_workpiece(c3, center_x=200, center_y=150, width=100, height=48, angle_deg=0)
    cv2.imwrite(os.path.join(output_dir, "sample_03_undersized_height.png"), c3)

    # 4. Surface Scratch Defect (Dimensions ok, but sharp scratch across face -> FAIL)
    c4 = create_base_canvas()
    draw_metallic_workpiece(c4, center_x=200, center_y=150, width=100, height=60, angle_deg=0)
    scratch_pts = np.array([
        [178, 142], [188, 146], [202, 144], [214, 155], [225, 158]
    ], dtype=np.int32)
    cv2.polylines(c4, [scratch_pts], isClosed=False, color=(25, 25, 25), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "sample_04_surface_scratch.png"), c4)

    # 5. Internal Hole / Void Defect (Unwanted hole on surface -> FAIL)
    c5 = create_base_canvas()
    draw_metallic_workpiece(c5, center_x=200, center_y=150, width=100, height=60, angle_deg=0)
    cv2.circle(c5, (215, 150), radius=5, color=(15, 15, 20), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(c5, (216, 151), radius=5, color=(140, 140, 150), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "sample_05_internal_hole.png"), c5)

    # 6. Edge Chip / Missing Region (A noticeable notched bite out of the edge -> FAIL)
    c6 = create_base_canvas()
    pts = draw_metallic_workpiece(c6, center_x=200, center_y=150, width=100, height=60, angle_deg=0)
    # Cut out an edge notch/chip indentation into the top edge
    chip_poly = np.array([[190, 120], [200, 134], [212, 120]], dtype=np.int32)
    cv2.fillPoly(c6, [chip_poly], (235, 238, 240))
    cv2.polylines(c6, [chip_poly], isClosed=False, color=(170, 175, 180), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "sample_06_edge_chip.png"), c6)

    # 7. Multiple Defects (Scratch + Hole -> FAIL)
    c7 = create_base_canvas()
    draw_metallic_workpiece(c7, center_x=200, center_y=150, width=100, height=60, angle_deg=0)
    cv2.line(c7, (175, 138), (198, 158), color=(20, 20, 20), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(c7, (225, 145), radius=4, color=(10, 10, 15), thickness=-1, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "sample_07_multiple_defects.png"), c7)

    # 8. Nominal Rotated Part (W=100, H=60 rotated by 18 degrees, verifying orientation invariance -> PASS)
    c8 = create_base_canvas()
    draw_metallic_workpiece(c8, center_x=200, center_y=150, width=100, height=60, angle_deg=18.0)
    cv2.imwrite(os.path.join(output_dir, "sample_08_rotated_perfect.png"), c8)

    print(f"Generated 8 sample test images in {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_sample_suite(current_dir)
