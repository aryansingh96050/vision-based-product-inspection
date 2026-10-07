"""Computer Vision Processing Pipeline for Product Inspection."""
from processing.preprocessing import (
    convert_to_grayscale,
    reduce_noise,
    apply_threshold,
    detect_edges,
    run_preprocessing_pipeline
)
from processing.detection import (
    find_main_object,
    create_object_mask
)
from processing.measurement import (
    compute_measurements,
    evaluate_quality
)
from processing.defects import (
    detect_defects,
    annotate_inspection_result
)

__all__ = [
    "convert_to_grayscale",
    "reduce_noise",
    "apply_threshold",
    "detect_edges",
    "run_preprocessing_pipeline",
    "find_main_object",
    "create_object_mask",
    "compute_measurements",
    "evaluate_quality",
    "detect_defects",
    "annotate_inspection_result",
]
