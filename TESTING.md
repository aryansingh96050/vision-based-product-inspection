# Testing Specification & Test Plan

This document details the test matrix, test cases, and validation procedures for the **Vision-Based Product Inspection** system.

---

## 1. Automated Test Suite Overview

Automated tests are located in `tests/test_inspection.py` and implemented using the standard Python `unittest` framework.

To execute the test suite:

```powershell
python -m unittest tests/test_inspection.py -v
```

---

## 2. Test Cases Matrix

| Test ID | Scenario | Input Image | Expected Result | Pass Criteria |
|---|---|---|---|---|
| **TC-01** | Nominal Compliant Workpiece | `sample_01_perfect_bracket.png` | **PASS** | Width $\in [95, 105]$ px, Height $\in [55, 65]$ px, 0 defects. |
| **TC-02** | Oversized Dimension Defect | `sample_02_oversized_width.png` | **FAIL** | Measured width $> 105.0$ px; flagged with "Width deviation". |
| **TC-03** | Undersized Dimension Defect | `sample_03_undersized_height.png` | **FAIL** | Measured height $< 55.0$ px; flagged with "Height deviation". |
| **TC-04** | Surface Scratch Defect | `sample_04_surface_scratch.png` | **FAIL** | Scratch detected via Black-Hat morphology; $\ge 1$ defect. |
| **TC-05** | Internal Void / Hole Defect | `sample_05_internal_hole.png` | **FAIL** | Dark void detected; flagged as "Small Hole". |
| **TC-06** | Edge Chip / Missing Corner | `sample_06_edge_chip.png` | **FAIL** | Missing edge geometry detected via Convex Hull difference. |
| **TC-07** | Multiple Surface Defects | `sample_07_multiple_defects.png` | **FAIL** | Detects multiple distinct defects (Scratch + Hole). |
| **TC-08** | Orientation Invariance (Rotated Part) | `sample_08_rotated_perfect.png` | **PASS** | Part rotated $18^\circ$; oriented bounding box yields nominal $100\times 60$ px. |
| **TC-09** | SQLite Persistence & Analytics | Programmatic insertion | **PASS** | Saves record, retrieves history DataFrame, aggregates statistics correctly. |
| **TC-10** | Robust Error Handling | Empty arrays / Null inputs | **PASS** | Gracefully raises `ObjectNotFoundError` or `ValueError` with clear messages. |

---

## 3. Detailed Test Execution Results

```text
test_tc01_nominal_workpiece (tests.test_inspection.TestVisionInspectionPipeline)
TC-01: Compliant part meets all dimension tolerances and contains zero defects. ... ok

test_tc02_oversized_width_defect (tests.test_inspection.TestVisionInspectionPipeline)
TC-02: Workpiece with oversized width triggers FAIL on dimension tolerance. ... ok

test_tc03_undersized_height_defect (tests.test_inspection.TestVisionInspectionPipeline)
TC-03: Workpiece with undersized height triggers FAIL on dimension tolerance. ... ok

test_tc04_surface_scratch_defect (tests.test_inspection.TestVisionInspectionPipeline)
TC-04: Surface scratch is isolated and flagged as a defect. ... ok

test_tc05_internal_hole_defect (tests.test_inspection.TestVisionInspectionPipeline)
TC-05: Dark void / hole on workpiece is flagged as a defect. ... ok

test_tc06_edge_chip_defect (tests.test_inspection.TestVisionInspectionPipeline)
TC-06: Edge chip / missing corner is detected as edge defect. ... ok

test_tc07_multiple_defects (tests.test_inspection.TestVisionInspectionPipeline)
TC-07: Workpiece with multiple surface anomalies records all defects. ... ok

test_tc08_rotated_workpiece_orientation_invariance (tests.test_inspection.TestVisionInspectionPipeline)
TC-08: Rotated workpiece correctly measures oriented dimensions and PASSES. ... ok

test_tc09_sqlite_database_lifecycle (tests.test_inspection.TestVisionInspectionPipeline)
TC-09: Verifies recording inspections, querying history, and computing statistics. ... ok

test_tc10_error_handling_invalid_inputs (tests.test_inspection.TestVisionInspectionPipeline)
TC-10: Pipeline handles empty or invalid inputs gracefully. ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.259s

OK
```

---

## 4. Manual UI Verification Procedures

1. **Single Part Inspection Mode**:
   - Select "Nominal Compliant Part" $\rightarrow$ Verify green **PASS** banner displays with dimensions close to $100.0 \times 60.0$ px.
   - Select "Surface Scratch Defect" $\rightarrow$ Verify red **FAIL** banner displays, with red bounding box around the scratch labeled `[SCRATCH]`.
   - Inspect intermediate tabs ("1. Grayscale" to "5. Contour & Mask") $\rightarrow$ Confirm images render correctly.
2. **Batch Inspection Mode**:
   - Click "📁 Load All 8 Sample Test Images for Batch Run".
   - Click "🚀 Execute Batch Inspection".
   - Confirm batch metrics show 2 Passes, 6 Fails, and 25.0% Pass Rate.
   - Click "📥 Download Batch Results (CSV)" and verify the generated CSV file.
3. **Historical Analytics Mode**:
   - Verify donut chart reflects Pass/Fail proportions.
   - Verify horizontal bar chart ranks defect frequencies.
   - Verify historical inspection records table matches SQLite records.
4. **Parameter Interactivity**:
   - On an oversized workpiece, expand the width tolerance slider from $\pm 5$ px to $\pm 20$ px $\rightarrow$ Verify the verdict changes interactively from **FAIL** to **PASS**.
