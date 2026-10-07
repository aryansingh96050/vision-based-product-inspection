import unittest
import os
import cv2
import numpy as np
from processing.preprocessing import run_preprocessing_pipeline, convert_to_grayscale, reduce_noise
from processing.detection import find_main_object, ObjectNotFoundError
from processing.measurement import compute_measurements, evaluate_quality
from processing.defects import detect_defects, annotate_inspection_result
from database.database import InspectionDatabase

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")

class TestVisionInspectionPipeline(unittest.TestCase):
    """Automated test suite verifying the computer vision product inspection pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.db_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_inspections.db")
        cls.db = InspectionDatabase(cls.db_test_path)
        cls.db.clear_history()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.db_test_path):
            try:
                os.remove(cls.db_test_path)
            except Exception:
                pass

    def _inspect_image(self, filename, exp_w=100.0, tol_w=5.0, exp_h=60.0, tol_h=5.0):
        img_path = os.path.join(SAMPLE_DIR, filename)
        self.assertTrue(os.path.exists(img_path), f"File {filename} not found in sample_data.")
        img = cv2.imread(img_path)
        self.assertIsNotNone(img, f"Failed to load {filename}")

        prep = run_preprocessing_pipeline(img)
        obj = find_main_object(prep["thresholded"])
        meas = compute_measurements(obj["contour"], obj["min_area_rect"])
        is_dim_pass, dim_failures, _ = evaluate_quality(meas, exp_w, tol_w, exp_h, tol_h)
        defects = detect_defects(img, prep["grayscale"], obj["contour"], obj["mask"])
        overall_pass = is_dim_pass and (len(defects) == 0)

        annotated = annotate_inspection_result(
            img, obj["contour"], obj["box_points"], meas, defects, overall_pass, "TEST-ITEM"
        )
        self.assertEqual(annotated.shape, img.shape)

        return {
            "overall_pass": overall_pass,
            "is_dim_pass": is_dim_pass,
            "dim_failures": dim_failures,
            "measurements": meas,
            "defects": defects,
            "annotated": annotated
        }

    def test_tc01_nominal_workpiece(self):
        """TC-01: Compliant part meets all dimension tolerances and contains zero defects."""
        res = self._inspect_image("sample_01_perfect_bracket.png")
        self.assertTrue(res["overall_pass"], "Expected nominal bracket to PASS")
        self.assertEqual(len(res["defects"]), 0, "Nominal bracket should have 0 defects")
        self.assertAlmostEqual(res["measurements"]["width"], 100.0, delta=2.0)
        self.assertAlmostEqual(res["measurements"]["height"], 60.0, delta=2.0)

    def test_tc02_oversized_width_defect(self):
        """TC-02: Workpiece with oversized width triggers FAIL on dimension tolerance."""
        res = self._inspect_image("sample_02_oversized_width.png")
        self.assertFalse(res["overall_pass"])
        self.assertFalse(res["is_dim_pass"])
        self.assertTrue(any("Width deviation" in f for f in res["dim_failures"]))

    def test_tc03_undersized_height_defect(self):
        """TC-03: Workpiece with undersized height triggers FAIL on dimension tolerance."""
        res = self._inspect_image("sample_03_undersized_height.png")
        self.assertFalse(res["overall_pass"])
        self.assertFalse(res["is_dim_pass"])
        self.assertTrue(any("Height deviation" in f for f in res["dim_failures"]))

    def test_tc04_surface_scratch_defect(self):
        """TC-04: Surface scratch is isolated and flagged as a defect."""
        res = self._inspect_image("sample_04_surface_scratch.png")
        self.assertFalse(res["overall_pass"])
        self.assertTrue(len(res["defects"]) >= 1)
        defect_types = [d["type"] for d in res["defects"]]
        self.assertIn("Scratch", defect_types)

    def test_tc05_internal_hole_defect(self):
        """TC-05: Dark void / hole on workpiece is flagged as a defect."""
        res = self._inspect_image("sample_05_internal_hole.png")
        self.assertFalse(res["overall_pass"])
        self.assertTrue(len(res["defects"]) >= 1)
        defect_types = [d["type"] for d in res["defects"]]
        self.assertTrue("Small Hole" in defect_types or "Surface Irregularity" in defect_types)

    def test_tc06_edge_chip_defect(self):
        """TC-06: Edge chip / missing corner is detected as edge defect."""
        res = self._inspect_image("sample_06_edge_chip.png")
        self.assertFalse(res["overall_pass"])
        self.assertTrue(len(res["defects"]) >= 1 or not res["is_dim_pass"])

    def test_tc07_multiple_defects(self):
        """TC-07: Workpiece with multiple surface anomalies records all defects."""
        res = self._inspect_image("sample_07_multiple_defects.png")
        self.assertFalse(res["overall_pass"])
        self.assertTrue(len(res["defects"]) >= 1)

    def test_tc08_rotated_workpiece_orientation_invariance(self):
        """TC-08: Rotated workpiece correctly measures oriented dimensions and PASSES."""
        res = self._inspect_image("sample_08_rotated_perfect.png")
        self.assertTrue(res["overall_pass"])
        self.assertAlmostEqual(res["measurements"]["width"], 100.0, delta=4.0)
        self.assertAlmostEqual(res["measurements"]["height"], 60.0, delta=4.0)

    def test_tc09_sqlite_database_lifecycle(self):
        """TC-09: Verifies recording inspections, querying history, and computing statistics."""
        rec_id = self.db.save_inspection(
            product_id="TEST-999",
            result="PASS",
            measured_width=100.5,
            measured_height=60.1,
            measured_area=6000.0,
            measured_perimeter=320.0,
            circularity=0.74,
            defects_count=0,
            defects_details=[],
            defect_summary="None",
            processing_time_ms=12.4,
            source="Automated Test"
        )
        self.assertGreater(rec_id, 0)

        df = self.db.get_history(limit=10)
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["Product ID"], "TEST-999")

        stats = self.db.get_statistics()
        self.assertGreaterEqual(stats["total_inspected"], 1)
        self.assertGreaterEqual(stats["passed"], 1)

    def test_tc10_error_handling_invalid_inputs(self):
        """TC-10: Pipeline handles empty or invalid inputs gracefully."""
        # Empty image
        blank = np.zeros((200, 200), dtype=np.uint8)
        with self.assertRaises(ObjectNotFoundError):
            find_main_object(blank, min_contour_area=500.0)

        # Null image to preprocessing
        with self.assertRaises(ValueError):
            convert_to_grayscale(None)

if __name__ == "__main__":
    unittest.main()
