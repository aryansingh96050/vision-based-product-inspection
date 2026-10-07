# Computer Vision Algorithms & Methodology

This document details the mathematical and algorithmic foundations of the **Vision-Based Product Inspection** system.

---

## 1. Why Preprocessing is Required

Raw sensor data from CMOS/CCD cameras or image files cannot be directly fed into measurement algorithms due to noise, uneven illumination, and unnecessary color dimensionality:

### 1.1 Color to Grayscale Conversion
A standard color image contains three color channels (Red, Green, Blue) representing $H \times W \times 3$ values. In dimensional inspection, color chromaticity is largely redundant; the geometric boundary of the object is governed by luminance contrast. Converting to single-channel 8-bit grayscale ($H \times W \times 1$) reduces compute latency by $\sim 66\%$ and standardizes intensity:

$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

### 1.2 Noise Reduction via Gaussian Filtering
Industrial inspection environments are subject to sensor thermal noise, electrical fluctuations, and photon shot noise. High-frequency pixel variations cause false edges during gradient detection. 

We convolve the grayscale image with a 2D Gaussian smoothing kernel:

$$G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

This attenuates high-frequency noise while preserving macro geometric boundaries.

### 1.3 Segmentation via Otsu's Thresholding
To isolate the workpiece from the conveyor or background, we separate the image into foreground ($255$) and background ($0$). Otsu's algorithm calculates the optimal threshold $t^*$ by maximizing the between-class variance $\sigma_b^2(t)$ (equivalent to minimizing intra-class variance):

$$\sigma_b^2(t) = \omega_0(t) \omega_1(t) \left(\mu_0(t) - \mu_1(t)\right)^2$$

Where:
- $\omega_0(t), \omega_1(t)$ are cumulative probabilities of background and foreground.
- $\mu_0(t), \mu_1(t)$ are mean intensity levels of the respective classes.

### 1.4 Boundary Edge Detection via Canny
The Canny operator detects crisp structural edges through:
1. **Gaussian smoothing**.
2. **Gradient magnitude & direction** calculation using Sobel kernels ($G = \sqrt{G_x^2 + G_y^2}$, $\theta = \arctan(G_y / G_x)$).
3. **Non-Maximum Suppression (NMS)**: Thins gradient peaks to single-pixel width along the gradient normal.
4. **Hysteresis Thresholding**: Edges above $T_{\text{high}}$ are retained; pixels between $T_{\text{low}}$ and $T_{\text{high}}$ are preserved only if connected to strong edges.

---

## 2. How Contours Work

Contours represent continuous curves connecting all boundary points having the same color or intensity.

### 2.1 Suzuki's Border Following Algorithm
OpenCV’s `cv2.findContours` implements Satoshi Suzuki and Keiichi Abe's (1985) topological structural analysis algorithm:
- Raster scans the binary image row by row until encountering a transition from background ($0$) to foreground ($1$).
- Distinguishes between outer borders and hole borders using direction vectors and state tags.
- Constructs a 2-level or hierarchical tree representing enclosing contours and internal voids.

### 2.2 Discrete Green's Theorem for Area
Contour area is calculated analytically via the Shoelace Formula (derived from Green's Theorem):

$$A = \frac{1}{2} \left| \sum_{i=0}^{N-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$$

Where $(x_N, y_N) = (x_0, y_0)$.

### 2.3 Polygonal Arc Length for Perimeter
The perimeter $P$ is computed by accumulating Euclidean distances between successive contour vertices:

$$P = \sum_{i=0}^{N-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$$

---

## 3. How Measurements are Calculated

### 3.1 Orientation-Invariant Dimensions
In real manufacturing setups, parts are rarely aligned perfectly parallel to the camera axes. An axis-aligned bounding box (`cv2.boundingRect`) will measure inflated width and height when a part is rotated even by $5^\circ$.

To eliminate orientation error, our system applies **Minimum Area Bounding Box** (`cv2.minAreaRect`):
- Computes the convex hull of the workpiece.
- Utilizes the **Rotating Calipers** algorithm (Toussaint, 1983) to find the minimum enclosing bounding rectangle of arbitrary orientation in $O(N)$ time.
- Returns oriented dimensions $(\text{dim}_1, \text{dim}_2)$ independent of part rotation angle $\theta$.
- Normalizes dimensions:
  $$\text{Width} = \max(\text{dim}_1, \text{dim}_2), \quad \text{Height} = \min(\text{dim}_1, \text{dim}_2)$$

### 3.2 Circularity Metric
Circularity (isoperimetric quotient) assesses roundness:

$$C = \frac{4 \pi \cdot A}{P^2}$$

- For a perfect circle, $C = 1.0$.
- For a square, $C = \frac{\pi}{4} \approx 0.785$.
- For elongated or chipped parts, $C$ decreases significantly.

### 3.3 Solidity
Solidity quantifies geometric completeness:

$$S = \frac{\text{Area}(C)}{\text{Area}(\text{ConvexHull}(C))}$$

A part missing material or with edge notches exhibits reduced solidity ($S < 1.0$).

---

## 4. How Defects are Detected

Defects are divided into surface irregularities and boundary irregularities:

### 4.1 Internal Boundary Masking
To prevent the outer edge of the workpiece from triggering false alarms, the workpiece binary mask $\mathcal{M}_{\text{object}}$ is eroded by an elliptical structuring element $K_e$:

$$\mathcal{M}_{\text{surface}} = \mathcal{M}_{\text{object}} \ominus K_e$$

All surface anomaly operations are restricted strictly within $\mathcal{M}_{\text{surface}}$.

### 4.2 Scratch Detection (Morphological Black-Hat & Top-Hat)
Scratches appear as sharp local intensity deficits (grooves) or specular highlights. We compute:

$$\text{Black-Hat}(I) = (I \bullet K) - I$$
$$\text{Top-Hat}(I) = I - (I \circ K)$$

Where $\bullet$ is morphological closing and $\circ$ is opening with structuring element $K$ ($9 \times 9$). The combined anomaly response is masked by $\mathcal{M}_{\text{surface}}$ and thresholded:

$$A_{\text{surface}} = \left( \text{Black-Hat}(I) + \text{Top-Hat}(I) \right) \cap \mathcal{M}_{\text{surface}}$$

Connected components with aspect ratio $\ge 2.0$ or high perimeter-to-area ratio are classified as **Scratches**.

### 4.3 Holes and Internal Voids
Compact dark regions inside $\mathcal{M}_{\text{surface}}$ with aspect ratio $< 1.8$ and circularity $> 0.40$ are categorized as **Small Holes / Voids**.

### 4.4 Edge Chips & Missing Material
Edge chips represent missing material along the perimeter. We detect them via boolean set difference between the workpiece mask and its convex envelope:

$$\mathcal{M}_{\text{chips}} = \text{ConvexHull}(\mathcal{M}_{\text{object}}) \setminus \mathcal{M}_{\text{object}}$$

Any connected component of $\mathcal{M}_{\text{chips}}$ with area $\ge \text{min\_defect\_area}$ is categorized as an **Edge Chip / Missing Region**.

---

## 5. Limitations of the System

While effective for 2D planar inspection, classical computer vision has inherent physical limitations:

1. **Planar Perspective Assumption**:
   - Assumes orthogonal (telecentric) camera projection. If the camera is tilted relative to the part plane, perspective distortion alters measured width and height.
2. **Lighting Sensitivity**:
   - Extreme shadows or specular reflections (glare) can disrupt Otsu thresholding or trigger false scratch detections. Consistent diffuse lighting (e.g., dome light or ring light) is essential in production.
3. **Absence of Absolute World Calibration**:
   - Measurements are reported in pixels ($\text{px}$). Converting to engineering units (e.g., millimeters) requires calibration against an optical target (e.g., Checkerboard or Dot Target) at a fixed working distance.
4. **Z-Axis Blindness**:
   - 2D cameras cannot measure workpiece depth, thickness, or 3D warpage. A 3D laser profiler or structured light system is required for depth inspection.
5. **Overlapping Workpieces**:
   - The system assumes isolated parts. Overlapping or touching workpieces merge into a single contour unless separated via Watershed segmentation.
