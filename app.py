import streamlit as st
import cv2
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# Import local modules
from processing.preprocessing import (
    run_preprocessing_pipeline,
    convert_to_grayscale,
    reduce_noise,
    apply_threshold,
    detect_edges
)
from processing.detection import find_main_object, ObjectNotFoundError
from processing.measurement import compute_measurements, evaluate_quality
from processing.defects import detect_defects, annotate_inspection_result
from database.database import InspectionDatabase
from utils.logger import setup_logger

logger = setup_logger("app")

# Page configuration
st.set_page_config(
    page_title="Vision-Based Product Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean dashboard look
st.markdown("""
<style>
    .reportview-container {
        background: #fdfdfd;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-pass {
        background-color: #d1fae5;
        color: #065f46;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 6px 16px;
        border-radius: 6px;
        border: 1px solid #10b981;
        display: inline-block;
    }
    .badge-fail {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 6px 16px;
        border-radius: 6px;
        border: 1px solid #ef4444;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 8px;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database singleton in session state
@st.cache_resource
def get_database():
    return InspectionDatabase()

db = get_database()

# Sample data directory
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
SAMPLE_FILES = {
    "Nominal Compliant Part (Pass)": "sample_01_perfect_bracket.png",
    "Oversized Width Part (Fail)": "sample_02_oversized_width.png",
    "Undersized Height Part (Fail)": "sample_03_undersized_height.png",
    "Surface Scratch Defect (Fail)": "sample_04_surface_scratch.png",
    "Internal Void / Hole Defect (Fail)": "sample_05_internal_hole.png",
    "Edge Chip / Missing Region (Fail)": "sample_06_edge_chip.png",
    "Multiple Surface Defects (Fail)": "sample_07_multiple_defects.png",
    "Rotated Compliant Part (Pass)": "sample_08_rotated_perfect.png",
}

# ==========================================
# SIDEBAR CONTROLS & ADJUSTABLE PARAMETERS
# ==========================================
st.sidebar.title("🔍 Vision Inspection")
st.sidebar.caption("Automated quality verification with OpenCV")

mode = st.sidebar.radio(
    "Navigation",
    ["Single Part Inspection", "Batch Inspection", "Historical Analytics", "System Guide"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Quality Rules")
col_exp_w, col_tol_w = st.sidebar.columns(2)
with col_exp_w:
    expected_width = st.number_input("Expected Width (px)", value=100.0, step=1.0)
with col_tol_w:
    width_tolerance = st.number_input("Width Tol (±px)", value=5.0, step=0.5)

col_exp_h, col_tol_h = st.sidebar.columns(2)
with col_exp_h:
    expected_height = st.number_input("Expected Height (px)", value=60.0, step=1.0)
with col_tol_h:
    height_tolerance = st.number_input("Height Tol (±px)", value=5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Vision Parameters")

with st.sidebar.expander("Image Preprocessing", expanded=False):
    blur_kernel = st.slider("Gaussian Blur Kernel", min_value=1, max_value=15, value=5, step=2)
    threshold_method = st.selectbox("Threshold Algorithm", ["otsu", "binary", "adaptive"], index=0)
    manual_threshold = st.slider("Binary Threshold Cut-off", min_value=10, max_value=245, value=127)
    force_invert = st.checkbox("Invert Binary Polarity", value=False)
    edge_low = st.slider("Canny Low Threshold", 10, 200, 50)
    edge_high = st.slider("Canny High Threshold", 50, 300, 150)

with st.sidebar.expander("Defect Detection Sensitivity", expanded=False):
    min_defect_area = st.slider("Minimum Defect Area (px²)", min_value=5.0, max_value=100.0, value=15.0, step=1.0)
    scratch_sensitivity = st.slider("Scratch Contrast Threshold", min_value=15, max_value=90, value=35)
    detect_edge_chips = st.checkbox("Detect Edge Chips & Indents", value=True)


def execute_inspection(image_bgr: np.ndarray, product_id: str):
    """Executes the complete vision pipeline on an image."""
    start_time = time.perf_counter()

    # Preprocessing
    prep_results = run_preprocessing_pipeline(
        image=image_bgr,
        blur_kernel=blur_kernel,
        threshold_method=threshold_method,
        threshold_val=manual_threshold,
        edge_low=edge_low,
        edge_high=edge_high,
        force_invert=force_invert
    )

    # Object Detection
    obj_data = find_main_object(prep_results["thresholded"], min_contour_area=250.0)

    # Dimensional Measurement
    measurements = compute_measurements(obj_data["contour"], obj_data["min_area_rect"])

    # Quality Spec Evaluation
    is_dim_pass, dim_failures, eval_metrics = evaluate_quality(
        measurements=measurements,
        expected_width=expected_width,
        width_tolerance=width_tolerance,
        expected_height=expected_height,
        height_tolerance=height_tolerance
    )

    # Defect Detection
    defects = detect_defects(
        original_image=image_bgr,
        gray_image=prep_results["grayscale"],
        main_contour=obj_data["contour"],
        object_mask=obj_data["mask"],
        min_defect_area=min_defect_area,
        scratch_sensitivity=scratch_sensitivity,
        detect_edge_chips=detect_edge_chips
    )

    # Combined pass/fail decision
    overall_pass = is_dim_pass and (len(defects) == 0)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Defect summary text
    defect_summaries = []
    if not is_dim_pass:
        defect_summaries.extend(dim_failures)
    for d in defects:
        defect_summaries.append(f"{d['type']} ({d['area']} px²)")
    summary_text = "; ".join(defect_summaries) if defect_summaries else "None"

    # Visual Annotation
    annotated = annotate_inspection_result(
        image=image_bgr,
        main_contour=obj_data["contour"],
        box_points=obj_data["box_points"],
        measurements=measurements,
        defects=defects,
        is_passed=overall_pass,
        product_id=product_id,
        failure_reasons=dim_failures
    )

    return {
        "overall_pass": overall_pass,
        "is_dim_pass": is_dim_pass,
        "dim_failures": dim_failures,
        "measurements": measurements,
        "defects": defects,
        "defect_summary": summary_text,
        "processing_time_ms": elapsed_ms,
        "prep_stages": prep_results,
        "obj_data": obj_data,
        "annotated_bgr": annotated
    }


# ==========================================
# VIEW 1: SINGLE PART INSPECTION
# ==========================================
if mode == "Single Part Inspection":
    st.title("Automated Product Inspection")
    st.markdown("Inspect workpiece geometry, detect surface irregularities, and verify dimension tolerance.")

    # Input source selection
    input_source = st.radio(
        "Select Image Input Source:",
        ["Sample Images", "Upload Image", "Webcam Capture"],
        horizontal=True
    )

    input_image_bgr = None
    input_source_label = "Sample"

    if input_source == "Sample Images":
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            selected_sample_label = st.selectbox("Choose a test workpiece:", list(SAMPLE_FILES.keys()))
            sample_filename = SAMPLE_FILES[selected_sample_label]
            sample_path = os.path.join(SAMPLE_DIR, sample_filename)
            if os.path.exists(sample_path):
                input_image_bgr = cv2.imread(sample_path)
                input_source_label = f"Sample ({sample_filename})"
            else:
                st.error(f"Sample image not found: {sample_path}")
        with col_s2:
            st.info(f"**Loaded:** `{sample_filename}`\n\nNominal spec: 100×60 px ±5 px")

    elif input_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload workpiece image (PNG, JPG, BMP)",
            type=["png", "jpg", "jpeg", "bmp"]
        )
        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                input_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if input_image_bgr is None:
                    st.error("Error: Could not decode uploaded file as an image.")
                input_source_label = f"Upload ({uploaded_file.name})"
            except Exception as e:
                st.error(f"Failed to read image: {e}")

    elif input_source == "Webcam Capture":
        st.write("Align the manufactured part within the camera frame under consistent lighting.")
        camera_img = st.camera_input("Capture inspection snapshot")
        if camera_img is not None:
            try:
                bytes_data = camera_img.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                input_image_bgr = cv2_img
                input_source_label = "Webcam"
            except Exception as e:
                st.error(f"Camera frame capture error: {e}")

    # Processing section
    if input_image_bgr is not None:
        # Default Product ID
        prod_id_col1, prod_id_col2 = st.columns([2, 1])
        with prod_id_col1:
            product_id = st.text_input("Product Identifier", value=f"PROD-{int(time.time()) % 10000:04d}")
        with prod_id_col2:
            auto_save = st.checkbox("Auto-save to Database", value=True)

        try:
            results = execute_inspection(input_image_bgr, product_id)

            # Save to Database if enabled
            if auto_save:
                # Use session state to avoid duplicate saves on rerun
                save_key = f"saved_{product_id}_{results['overall_pass']}"
                if save_key not in st.session_state:
                    db.save_inspection(
                        product_id=product_id,
                        result="PASS" if results["overall_pass"] else "FAIL",
                        measured_width=results["measurements"]["width"],
                        measured_height=results["measurements"]["height"],
                        measured_area=results["measurements"]["area"],
                        measured_perimeter=results["measurements"]["perimeter"],
                        circularity=results["measurements"]["circularity"],
                        defects_count=len(results["defects"]),
                        defects_details=results["defects"],
                        defect_summary=results["defect_summary"],
                        processing_time_ms=results["processing_time_ms"],
                        source=input_source_label,
                        config_summary={
                            "exp_w": expected_width,
                            "tol_w": width_tolerance,
                            "exp_h": expected_height,
                            "tol_h": height_tolerance
                        }
                    )
                    st.session_state[save_key] = True

            # RESULT BANNER & SUMMARY
            st.markdown("---")
            res_col1, res_col2, res_col3, res_col4 = st.columns([2, 2, 2, 2])

            with res_col1:
                st.markdown("**Inspection Result**")
                if results["overall_pass"]:
                    st.markdown("<div class='badge-pass'>✅ PASS</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='badge-fail'>❌ FAIL</div>", unsafe_allow_html=True)

            with res_col2:
                st.markdown("**Dimensions (W × H)**")
                w_curr = results["measurements"]["width"]
                h_curr = results["measurements"]["height"]
                st.write(f"**{w_curr:.1f} px** × **{h_curr:.1f} px**")
                st.caption(f"Spec: {expected_width}±{width_tolerance} × {expected_height}±{height_tolerance}")

            with res_col3:
                st.markdown("**Defects Detected**")
                num_defects = len(results["defects"])
                if num_defects == 0:
                    st.write("🟢 None")
                else:
                    st.write(f"🔴 **{num_defects} defect(s)**")
                    st.caption(", ".join([d["type"] for d in results["defects"]]))

            with res_col4:
                st.markdown("**Processing Time**")
                st.write(f"⚡ **{results['processing_time_ms']:.1f} ms**")

            # VISUAL COMPARISON
            st.markdown("### 🖼️ Visual Comparison")
            vis_col1, vis_col2 = st.columns(2)
            with vis_col1:
                st.subheader("Original Workpiece")
                st.image(cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            with vis_col2:
                st.subheader("Inspection Overlay")
                st.image(cv2.cvtColor(results["annotated_bgr"], cv2.COLOR_BGR2RGB), use_container_width=True)

            # INTERMEDIATE PIPELINE STAGES
            st.markdown("### 🔬 Step-by-Step Processing Pipeline")
            st.caption("Inspect each computer vision stage from raw input through segmentation to edge features:")

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "1. Grayscale",
                "2. Noise Reduction",
                "3. Thresholding",
                "4. Edge Detection",
                "5. Contour & Mask",
                "6. Geometric Metrics"
            ])

            with tab1:
                st.write("**Grayscale Conversion:** Strips color channels to extract luminance intensity.")
                st.image(results["prep_stages"]["grayscale"], clamp=True, use_container_width=True)

            with tab2:
                st.write(f"**Gaussian Filtering (kernel {blur_kernel}×{blur_kernel}):** Suppresses high-frequency sensor noise.")
                st.image(results["prep_stages"]["blurred"], clamp=True, use_container_width=True)

            with tab3:
                st.write(f"**Thresholding ({threshold_method.upper()}):** Binarizes the part from background.")
                st.image(results["prep_stages"]["thresholded"], clamp=True, use_container_width=True)

            with tab4:
                st.write(f"**Canny Edge Boundaries:** Gradient thresholds [{edge_low}, {edge_high}].")
                st.image(results["prep_stages"]["edges"], clamp=True, use_container_width=True)

            with tab5:
                st.write("**Object Mask & Primary Contour:** Isolated workpiece region.")
                st.image(results["obj_data"]["mask"], clamp=True, use_container_width=True)

            with tab6:
                st.write("**Detailed Geometric Measurements:**")
                meas = results["measurements"]
                m_df = pd.DataFrame([
                    {"Parameter": "Oriented Width", "Measured": f"{meas['width']} px", "Spec / Tolerance": f"{expected_width} ± {width_tolerance} px", "Status": "PASS" if abs(meas['width'] - expected_width) <= width_tolerance else "FAIL"},
                    {"Parameter": "Oriented Height", "Measured": f"{meas['height']} px", "Spec / Tolerance": f"{expected_height} ± {height_tolerance} px", "Status": "PASS" if abs(meas['height'] - expected_height) <= height_tolerance else "FAIL"},
                    {"Parameter": "Straight Width", "Measured": f"{meas['straight_width']} px", "Spec / Tolerance": "Reference only", "Status": "—"},
                    {"Parameter": "Straight Height", "Measured": f"{meas['straight_height']} px", "Spec / Tolerance": "Reference only", "Status": "—"},
                    {"Parameter": "Contour Area", "Measured": f"{meas['area']} px²", "Spec / Tolerance": "—", "Status": "—"},
                    {"Parameter": "Perimeter", "Measured": f"{meas['perimeter']} px", "Spec / Tolerance": "—", "Status": "—"},
                    {"Parameter": "Circularity Factor", "Measured": f"{meas['circularity']:.4f}", "Spec / Tolerance": "0.0 - 1.0 (Circle=1.0)", "Status": "—"},
                    {"Parameter": "Aspect Ratio", "Measured": f"{meas['aspect_ratio']:.3f}", "Spec / Tolerance": "—", "Status": "—"},
                    {"Parameter": "Solidity", "Measured": f"{meas['solidity']:.4f}", "Spec / Tolerance": "Contour / Convex Hull", "Status": "—"},
                ])
                st.dataframe(m_df, use_container_width=True, hide_index=True)

            # DEFECTS BREAKDOWN
            if results["defects"]:
                st.markdown("### ⚠️ Surface Irregularities Detected")
                defect_records = []
                for idx, d in enumerate(results["defects"], start=1):
                    defect_records.append({
                        "ID": f"#{idx}",
                        "Anomaly Type": d["type"],
                        "Pixel Area": f"{d['area']} px²",
                        "Location (X, Y)": f"{d['center'][0]}, {d['center'][1]}",
                        "Bounding Box (W×H)": f"{d['bbox'][2]} × {d['bbox'][3]} px",
                        "Confidence": f"{int(d['confidence'] * 100)}%"
                    })
                st.table(pd.DataFrame(defect_records))

        except ObjectNotFoundError as onf:
            st.error(f"❌ Inspection Error: {onf}")
            st.info("Tip: Try adjusting the Threshold value or selecting 'Invert Binary Polarity' in the sidebar.")
        except Exception as e:
            st.error(f"Unexpected inspection error: {str(e)}")
            logger.exception("Error executing inspection")

# ==========================================
# VIEW 2: BATCH INSPECTION
# ==========================================
elif mode == "Batch Inspection":
    st.title("Batch Workpiece Inspection")
    st.markdown("Upload multiple images to run automated multi-part inspection and generate batch compliance reports.")

    batch_files = st.file_uploader(
        "Upload images for batch analysis (PNG, JPG, BMP)",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True
    )

    col_btn, col_demo = st.columns([1, 2])
    with col_demo:
        load_samples_btn = st.button("📁 Load All 8 Sample Test Images for Batch Run")

    images_to_process = []
    if batch_files:
        for f in batch_files:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                images_to_process.append((f.name, img))
    elif load_samples_btn:
        for lbl, fn in SAMPLE_FILES.items():
            fpath = os.path.join(SAMPLE_DIR, fn)
            if os.path.exists(fpath):
                img = cv2.imread(fpath)
                images_to_process.append((fn, img))

    if images_to_process:
        st.write(f"**Found {len(images_to_process)} workpiece image(s) to process.**")
        start_batch = st.button("🚀 Execute Batch Inspection", type="primary")

        if start_batch:
            progress_bar = st.progress(0)
            status_text = st.empty()

            batch_results = []
            pass_count = 0
            fail_count = 0

            for i, (fname, img) in enumerate(images_to_process):
                status_text.text(f"Processing ({i+1}/{len(images_to_process)}): {fname}...")
                pid = f"BATCH-{i+1:03d}_{fname.split('.')[0][:12]}"

                try:
                    res = execute_inspection(img, pid)
                    is_pass = res["overall_pass"]
                    if is_pass:
                        pass_count += 1
                    else:
                        fail_count += 1

                    # Save to DB
                    db.save_inspection(
                        product_id=pid,
                        result="PASS" if is_pass else "FAIL",
                        measured_width=res["measurements"]["width"],
                        measured_height=res["measurements"]["height"],
                        measured_area=res["measurements"]["area"],
                        measured_perimeter=res["measurements"]["perimeter"],
                        circularity=res["measurements"]["circularity"],
                        defects_count=len(res["defects"]),
                        defects_details=res["defects"],
                        defect_summary=res["defect_summary"],
                        processing_time_ms=res["processing_time_ms"],
                        source="Batch"
                    )

                    batch_results.append({
                        "File Name": fname,
                        "Product ID": pid,
                        "Result": "PASS" if is_pass else "FAIL",
                        "Width (px)": res["measurements"]["width"],
                        "Height (px)": res["measurements"]["height"],
                        "Defect Count": len(res["defects"]),
                        "Defects Found": res["defect_summary"],
                        "Time (ms)": f"{res['processing_time_ms']:.1f}"
                    })
                except Exception as e:
                    batch_results.append({
                        "File Name": fname,
                        "Product ID": pid,
                        "Result": "ERROR",
                        "Width (px)": 0,
                        "Height (px)": 0,
                        "Defect Count": 0,
                        "Defects Found": f"Inspection failed: {str(e)}",
                        "Time (ms)": "0.0"
                    })

                progress_bar.progress((i + 1) / len(images_to_process))

            status_text.success("Batch inspection complete!")

            # Batch Summary Metrics
            st.markdown("### 📊 Batch Quality Summary")
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Total Processed", len(images_to_process))
            b_col2.metric("Passed", pass_count)
            b_col3.metric("Failed", fail_count)
            pass_rate = (pass_count / len(images_to_process) * 100) if images_to_process else 0.0
            b_col4.metric("Pass Rate", f"{pass_rate:.1f}%")

            # Batch Results Table
            batch_df = pd.DataFrame(batch_results)
            st.dataframe(batch_df, use_container_width=True)

            # CSV Download
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Results (CSV)",
                data=csv_data,
                file_name=f"batch_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# ==========================================
# VIEW 3: HISTORICAL ANALYTICS
# ==========================================
elif mode == "Historical Analytics":
    st.title("📈 Inspection Analytics & Database")
    st.markdown("Track quality trends, defect distribution, and review records stored in SQLite.")

    stats = db.get_statistics()

    # Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Inspected", stats["total_inspected"])
    m2.metric("Total Passed", stats["passed"])
    m3.metric("Total Failed", stats["failed"])
    m4.metric("Pass Rate", f"{stats['pass_percentage']}%")
    m5.metric("Avg Speed", f"{stats['avg_processing_time_ms']} ms")

    if stats["total_inspected"] > 0:
        st.markdown("### 📊 Quality Distribution & Defect Trends")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Pass vs Fail Donut Chart
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            labels = ["Passed", "Failed"]
            sizes = [stats["passed"], stats["failed"]]
            colors = ["#2ecc71", "#e74c3c"]
            if sum(sizes) > 0:
                ax1.pie(
                    sizes, labels=labels, autopct="%1.1f%%",
                    colors=colors, startangle=140,
                    wedgeprops=dict(width=0.4, edgecolor='w')
                )
                ax1.set_title("Overall Pass / Fail Ratio", fontsize=11, fontweight="bold")
                st.pyplot(fig1)
            plt.close(fig1)

        with chart_col2:
            # Defect Type Frequency Bar Chart
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            df_counts = stats["defect_type_counts"]
            if df_counts:
                names = list(df_counts.keys())
                vals = list(df_counts.values())
                y_pos = np.arange(len(names))
                ax2.barh(y_pos, vals, color="#3498db")
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(names, fontsize=9)
                ax2.invert_yaxis()
                ax2.set_xlabel("Frequency Count", fontsize=9)
                ax2.set_title("Defect Type Distribution", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("No defects recorded yet.")
            plt.close(fig2)

        st.markdown("### 🗄️ Inspection History Log")
        history_df = db.get_history(limit=500)
        st.dataframe(history_df, use_container_width=True)

        col_export, col_clear = st.columns([1, 1])
        with col_export:
            csv_history = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export History to CSV",
                data=csv_history,
                file_name="inspection_history.csv",
                mime="text/csv"
            )
        with col_clear:
            if st.button("🗑️ Clear Inspection Database History"):
                db.clear_history()
                st.warning("Database records cleared.")
                st.rerun()

    else:
        st.info("No inspection records in database yet. Run single or batch inspections to populate analytics.")

# ==========================================
# VIEW 4: SYSTEM GUIDE
# ==========================================
elif mode == "System Guide":
    st.title("System Guide & Vision Architecture")
    st.markdown("""
    ### How Vision-Based Product Inspection Works
    
    1. **Preprocessing**:
       - Converts color inputs to 8-bit grayscale for illumination processing.
       - Gaussian convolution kernel suppresses sensor noise and speckles.
       - Otsu/Adaptive binarization separates the manufactured object from the conveyor/table background.
       
    2. **Object Detection**:
       - Suzuki's border following algorithm identifies external geometric contours.
       - The dominant contour is isolated using contour area filtering.
       - Minimum Area Bounding Box computes oriented width and height without tilt error.

    3. **Quality Rules**:
       - Evaluates oriented dimensions against expected nominal size and tolerances.
       - If either dimension deviates beyond specification range, the part triggers a **FAIL**.

    4. **Surface Irregularity Detection**:
       - **Scratches**: Morphological Black-Hat and Top-Hat filtering detect elongated high-contrast grooves.
       - **Holes / Pores**: Local intensity deficit segmentation detects dark internal voids.
       - **Edge Chips / Missing Regions**: Contour difference with Convex Hull and Oriented Box identifies missing material along boundaries.

    5. **Database Storage**:
       - Every inspection record is logged to SQLite with timestamps, measurements, defect details, and processing duration.
    """)
