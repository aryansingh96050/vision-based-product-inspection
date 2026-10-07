# Vision-Based Product Inspection

An automated, local computer vision quality inspection system engineered with **Python**, **OpenCV**, and **Streamlit**. 

This system demonstrates how classical computer vision algorithms inspect manufactured mechanical workpieces, verify precision dimensional tolerances, and detect micro-surface anomalies (scratches, unwanted holes/voids, edge chips, and missing geometry) in real-time without requiring expensive industrial hardware or cloud dependencies.

---

## Key Features

- **Multi-Source Image Acquisition**:
  - Load pre-bundled synthetic sample workpieces (nominal, dimensional fails, scratches, voids, chips, rotated).
  - Upload user-provided workpiece imagery (`.png`, `.jpg`, `.jpeg`, `.bmp`).
  - Real-time webcam frame acquisition for desktop or benchtop testing.
- **Visualized 6-Stage Preprocessing Pipeline**:
  - Original Image $\rightarrow$ Grayscale conversion $\rightarrow$ Noise reduction (Gaussian filtering) $\rightarrow$ Thresholding (Otsu / Adaptive / Binary) $\rightarrow$ Edge detection (Canny) $\rightarrow$ Suzuki contour segmentation $\rightarrow$ Annotated quality verdict.
  - Interactive stage viewer allowing engineers to inspect intermediate transformation arrays.
- **Rotation-Invariant Geometric Measurement**:
  - Calculates Oriented Width and Height using Minimum Area Bounding Rectangles (`cv2.minAreaRect`), avoiding skew distortions.
  - Calculates contour Area ($\text{px}^2$), Perimeter ($\text{px}$), Circularity ($4\pi A / P^2$), Aspect Ratio, and Solidity.
- **Configurable Engineering Quality Limits**:
  - Real-time specification checks for Target Width ($\pm$ tolerance) and Target Height ($\pm$ tolerance).
  - Instantaneous binary **PASS** / **FAIL** classification with human-readable failure rationales.
- **Multi-Defect Surface Anomaly Engine**:
  - **Scratches**: Isolated via morphological Black-Hat & Top-Hat differential filters on internal surface pixels.
  - **Holes / Pores**: Detected via localized intensity deficit segmentation.
  - **Edge Chips / Missing Regions**: Discovered via morphological difference against the Convex Hull and oriented bounding envelope.
- **Batch Processing**:
  - Upload multiple parts simultaneously or batch-process all sample test suites.
  - Progress tracking, aggregate pass rate metrics, and one-click CSV export.
- **Persistent SQLite Audit Trail & Analytics**:
  - Automatically records every inspection: Product ID, timestamp, verdict, dimensional measurements, defect details, and execution latency.
  - Interactive dashboards featuring Pass/Fail donut charts, defect frequency histograms, and query filters.

---

## Technology Stack

| Component | Library / Framework | Purpose |
|---|---|---|
| **Core Vision** | `OpenCV (opencv-python)` | Preprocessing, morphology, contour tracing, defect detection, and drawing |
| **Numerical Processing** | `NumPy` | High-performance array manipulation and mask operations |
| **User Interface** | `Streamlit` | Interactive dashboard, real-time parameter tuning, and stage visualization |
| **Data Persistence** | `SQLite3` | Local relational database for complete audit history logging |
| **Data Analysis** | `Pandas` | Tabular reporting, data cleaning, and CSV exports |
| **Data Visualization** | `Matplotlib` | Quality distribution charts and defect frequency analytics |

---

## Project Structure

```text
vision_inspection/
├── app.py                      # Main Streamlit web dashboard & inspection interface
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & setup guide
├── ALGORITHM.md                # In-depth algorithmic breakdown & CV mathematics
├── TESTING.md                  # Test specifications & test cases (TC-01 through TC-10)
├── inspections.db              # SQLite historical inspection records (auto-generated)
├── processing/
│   ├── __init__.py             # Pipeline module exports
│   ├── preprocessing.py        # Grayscale, Gaussian blur, Otsu/Binary threshold, Canny
│   ├── detection.py            # Dominant workpiece extraction & bounding geometry
│   ├── measurement.py          # Geometric metrics (W, H, Area, Perimeter, Circularity)
│   └── defects.py              # Scratches, holes, edge chips & visual overlays
├── database/
│   ├── __init__.py             # Database exports
│   └── database.py             # SQLite CRUD, aggregate statistics & audit logging
├── sample_data/
│   ├── generate_samples.py     # Script to generate realistic test images
│   ├── sample_01_perfect_bracket.png
│   ├── sample_02_oversized_width.png
│   ├── sample_03_undersized_height.png
│   ├── sample_04_surface_scratch.png
│   ├── sample_05_internal_hole.png
│   ├── sample_06_edge_chip.png
│   ├── sample_07_multiple_defects.png
│   └── sample_08_rotated_perfect.png
├── utils/
│   ├── __init__.py
│   └── logger.py               # Structured application logger
└── tests/
    └── test_inspection.py     # Automated unit test suite (10 test cases)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed.
- Standard webcam (optional, required only for live camera feed mode).

### 2. Clone or Navigate to Directory
```powershell
cd C:\Users\aryan\.gemini\antigravity\scratch\vision_inspection
```

### 3. Create & Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Generate Sample Images (Optional - Pre-bundled)
```powershell
python .\sample_data\generate_samples.py
```

---

## Running the Application

Launch the Streamlit inspection dashboard with:

```powershell
streamlit run app.py
```

Once started, the application opens locally at `http://localhost:8501`.

---

## Running Automated Tests

Execute the unit test suite covering all defect scenarios, dimensional checks, rotation invariance, and database persistence:

```powershell
python -m unittest tests/test_inspection.py -v
```

Expected result:
```text
Ran 10 tests in 0.26s
OK
```

---

## Adjustable Parameters Guide

| Control | Recommended Value | Impact |
|---|---|---|
| **Expected Width** | `100.0 px` | Target nominal width of the workpiece. |
| **Width Tolerance** | `±5.0 px` | Permissible variation range ($95.0$ to $105.0$ px). |
| **Expected Height** | `60.0 px` | Target nominal height of the workpiece. |
| **Height Tolerance** | `±5.0 px` | Permissible variation range ($55.0$ to $65.0$ px). |
| **Gaussian Blur Kernel** | `5` | Suppresses high-frequency sensor noise. Higher values smooth rough edges. |
| **Threshold Algorithm** | `otsu` | Automatically computes bimodal separation threshold. |
| **Binary Threshold Cut-off** | `127` | Used when manual thresholding is selected. |
| **Canny Low / High** | `50 / 150` | Hysteresis thresholds for edge contour detection. |
| **Min Defect Area** | `15.0 px²` | Filters out single-pixel sensor noise from registering as defects. |
| **Scratch Sensitivity** | `35` | Lower values increase sensitivity to faint scratches. |
