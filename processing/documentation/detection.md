# Complete Guide to Understanding the Fiber Optic Anomaly Detection Code

## Table of Contents
1. [Introduction and Overview](#introduction-and-overview)
2. [Understanding the Imports](#understanding-the-imports)
3. [The Configuration System](#the-configuration-system)
4. [The JSON Encoder](#the-json-encoder)
5. [The Main Analyzer Class](#the-main-analyzer-class)
6. [Image Loading and Processing](#image-loading-and-processing)
7. [Statistical Functions](#statistical-functions)
8. [Feature Extraction](#feature-extraction)
9. [Comparison Methods](#comparison-methods)
10. [Reference Model Building](#reference-model-building)
11. [Anomaly Detection](#anomaly-detection)
12. [Visualization](#visualization)
13. [Report Generation](#report-generation)
14. [Main Execution](#main-execution)
15. [Line-by-Line Deep Dive Examples](#line-by-line-deep-dive-examples)

---

## Introduction and Overview

This Python script is a sophisticated system for detecting anomalies (defects) in fiber optic cable end faces. Think of it like a quality control inspector that uses computer vision and statistics to find problems in fiber optic connectors.

**For beginners**: Imagine you're inspecting the end of a fiber optic cable (like the cables that carry internet data) with a microscope. This program does that automatically, finding scratches, dirt, and other problems.

**For developers**: This is a comprehensive anomaly detection system using statistical learning, computer vision techniques, and multiple feature extraction methods to identify defects in fiber optic end face images.

### What the Script Does:
1. **Learns** what "normal" fiber optic ends look like from reference images
2. **Analyzes** new images to find problems
3. **Reports** specific defects like scratches, digs, and contamination
4. **Visualizes** the results with color-coded overlays

---

## Understanding the Imports

```python
#!/usr/bin/env python3
```
**For beginners**: This line tells the computer to use Python 3 to run this script. The `#!` is called a "shebang" and helps Unix/Linux systems know how to execute the file.

**For developers**: Standard shebang for Python 3 scripts, enabling direct execution on Unix-like systems.

```python
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging
import time
```

Let's break down each import:

### `import json`
**For beginners**: JSON (JavaScript Object Notation) is a way to store data in a format that's easy for both humans and computers to read. Like a structured shopping list.

**For developers**: Used for serializing/deserializing data structures, particularly for saving the reference model and reading pixel data from JSON files.

### `import os`
**For beginners**: Provides ways to interact with your computer's file system - checking if files exist, creating folders, etc.

**For developers**: Operating system interface for file system operations.

### `import cv2`
**For beginners**: This is OpenCV, a powerful library for working with images. It can read images, apply filters, find edges, and much more. Think of it as Photoshop for programmers.

**For developers**: OpenCV (Open Computer Vision) library - the primary computer vision toolkit used for image processing operations throughout the script.

### `import matplotlib.pyplot as plt`
**For beginners**: Creates graphs and visualizations. Like Excel charts but more powerful.

**For developers**: Used for creating the comprehensive multi-panel visualization of analysis results.

### `import numpy as np`
**For beginners**: NumPy handles mathematical operations on large arrays of numbers efficiently. If Python lists are like shopping lists, NumPy arrays are like spreadsheets.

**For developers**: Fundamental library for numerical computing, providing n-dimensional arrays and mathematical functions.

### `from dataclasses import dataclass`
**For beginners**: A decorator that automatically creates special methods for classes. It's like a template that saves you from writing repetitive code.

**For developers**: Python 3.7+ feature that reduces boilerplate for data-holding classes by auto-generating `__init__`, `__repr__`, etc.

### `from typing import Optional, List, Dict, Any, Tuple`
**For beginners**: These help specify what types of data functions expect and return. Like labeling boxes to show what goes inside.

**For developers**: Type hints for static type checking and improved code documentation. Enhances IDE support and helps catch type-related bugs.

### `from pathlib import Path`
**For beginners**: A modern way to work with file paths that works on Windows, Mac, and Linux.

**For developers**: Object-oriented filesystem paths, more intuitive than string manipulation for path operations.

### `import logging`
**For beginners**: Creates a diary of what the program is doing - useful for debugging when things go wrong.

**For developers**: Python's built-in logging framework for structured application logging.

### `import time`
**For beginners**: Works with dates and times, like checking what time it is or measuring how long something takes.

**For developers**: Used for timestamps and timing operations.

---

## Logging Configuration

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
```

**For beginners**: This sets up the logging system to:
- Show INFO level messages and above (INFO, WARNING, ERROR)
- Display messages with timestamp, level (like INFO or ERROR), and the actual message
- Example output: `2024-01-15 10:30:45 - [INFO] - Analyzing image...`

**For developers**: Configures the root logger with:
- Minimum level of INFO (won't show DEBUG messages)
- Format string using % formatting with timestamp, log level, and message
- Applies globally to all loggers in the application

---

## The Configuration System

```python
@dataclass
class OmniConfig:
    """Configuration for OmniFiberAnalyzer - matches expected structure from app.py"""
```

**For beginners**: The `@dataclass` decorator is like a recipe card that automatically creates a class with less code. The class holds all the settings for our analyzer.

**For developers**: Dataclass decorator generates boilerplate methods. The docstring indicates this config structure must match what the parent application (app.py) expects.

### Configuration Fields

```python
knowledge_base_path: Optional[str] = None
```
**For beginners**: Where to save/load the learned knowledge about good fiber optics. `Optional[str]` means it can be text (string) or `None` (nothing). Default is `None`.

**For developers**: Path to persistence file for the reference model. Optional type hint indicates nullable string.

```python
min_defect_size: int = 10
max_defect_size: int = 5000
```
**For beginners**: Defects smaller than 10 pixels or larger than 5000 pixels are ignored. This filters out tiny specs of dust and huge areas that are probably not real defects.

**For developers**: Size thresholds in pixels for connected component filtering. Helps eliminate noise (too small) and image artifacts (too large).

```python
severity_thresholds: Optional[Dict[str, float]] = None
```
**For beginners**: A dictionary (like a phone book) mapping severity levels (CRITICAL, HIGH, etc.) to confidence scores (0.0 to 1.0).

**For developers**: Maps severity classifications to confidence thresholds. Type is optional dictionary with string keys and float values.

```python
confidence_threshold: float = 0.3
anomaly_threshold_multiplier: float = 2.5
enable_visualization: bool = True
```
**For beginners**: 
- `confidence_threshold`: Only report anomalies we're at least 30% sure about
- `anomaly_threshold_multiplier`: Used in statistics (we'll explain later)
- `enable_visualization`: Whether to create visual output images

**For developers**:
- Minimum confidence for anomaly reporting
- Multiplier for standard deviation in statistical threshold calculation
- Boolean flag for visualization generation

### The `__post_init__` Method

```python
def __post_init__(self):
    if self.severity_thresholds is None:
        self.severity_thresholds = {
            'CRITICAL': 0.9,
            'HIGH': 0.7,
            'MEDIUM': 0.5,
            'LOW': 0.3,
            'NEGLIGIBLE': 0.1
        }
```

**For beginners**: This runs after the class is created. If no severity thresholds were provided, it sets up default ones. Like "90% confident = CRITICAL problem".

**For developers**: Post-initialization hook provided by dataclass. Establishes default severity mappings if none provided. Thresholds create bands for classification.

---

## The JSON Encoder

```python
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
```

**For beginners**: JSON can't directly save NumPy arrays or special number types. This class teaches JSON how to convert them to regular Python types it can save.

**For developers**: Extends JSONEncoder to handle NumPy types which aren't natively JSON serializable.

```python
def default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return super(NumpyEncoder, self).default(obj)
```

**For beginners**: 
- Checks if `obj` is a NumPy integer → converts to Python int
- Checks if `obj` is a NumPy float → converts to Python float  
- Checks if `obj` is a NumPy array → converts to Python list
- Otherwise, uses the parent class's method

**For developers**: Override of default serialization method. Uses isinstance for type checking. Converts NumPy types to JSON-compatible Python primitives. Falls back to parent implementation for unhandled types.

---

## The Main Analyzer Class

```python
class OmniFiberAnalyzer:
    """The ultimate fiber optic anomaly detection system - pipeline compatible version."""
```

**For beginners**: This is the main "brain" of our program. Classes are like blueprints - this one creates analyzer objects that can learn and detect anomalies.

**For developers**: Primary class implementing the anomaly detection pipeline. Designed for integration with a larger system (indicated by "pipeline compatible").

### Initialization

```python
def __init__(self, config: OmniConfig):
    self.config = config
    self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
```

**For beginners**: 
- `__init__` is the constructor - it runs when creating a new analyzer
- Stores the configuration
- Sets knowledge base path (uses default if none provided)

**For developers**: Constructor with dependency injection of configuration. Uses short-circuit evaluation for default path.

```python
self.reference_model = {
    'features': [],
    'statistical_model': None,
    'archetype_image': None,
    'feature_names': [],
    'comparison_results': {},
    'learned_thresholds': {},
    'timestamp': None
}
```

**For beginners**: Creates an empty reference model structure. Like preparing empty folders for different types of information we'll collect about "normal" fiber optics.

**For developers**: Initializes reference model dictionary with expected keys. This structure persists learned patterns from reference images.

Key components:
- `features`: List of feature dictionaries from each reference image
- `statistical_model`: Statistical parameters (mean, covariance, etc.)
- `archetype_image`: Median/representative image
- `feature_names`: Consistent ordering of features
- `comparison_results`: Cached comparison data
- `learned_thresholds`: Dynamically learned anomaly thresholds
- `timestamp`: Model creation/update time

```python
self.current_metadata = None
self.logger = logging.getLogger(__name__)
self.load_knowledge_base()
```

**For beginners**:
- `current_metadata`: Stores information about the image being processed
- `logger`: Creates a logger specifically for this class
- Tries to load existing knowledge from disk

**For developers**: 
- Metadata storage for current analysis context
- Logger instance using module name for hierarchical logging
- Attempts to restore persisted model on initialization

---

## The Main Analysis Method

```python
def analyze_end_face(self, image_path: str, output_dir: str):
    """Main analysis method - compatible with pipeline expectations"""
```

**For beginners**: This is the main entry point - give it an image path and output directory, and it analyzes the fiber optic end face.

**For developers**: Primary public interface method. Signature matches pipeline integration requirements.

Let's trace through the method:

```python
self.logger.info(f"Analyzing fiber end face: {image_path}")
```

**For beginners**: Logs a message saying which image we're analyzing. The `f` before the string allows us to insert variables using `{}`.

**For developers**: F-string formatting for dynamic log message. INFO level for operational tracking.

```python
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
```

**For beginners**: 
- Converts the output directory string to a Path object
- Creates the directory (and any parent directories) if they don't exist
- `exist_ok=True` means don't error if directory already exists

**For developers**: Path object for cross-platform compatibility. Creates directory hierarchy with parents=True. Idempotent operation with exist_ok.

```python
if not self.reference_model.get('statistical_model'):
    self.logger.warning("No reference model available. Building from single image...")
    self._build_minimal_reference(image_path)
```

**For beginners**: If we don't have a reference model (haven't learned what "normal" looks like), create a minimal one using the current image.

**For developers**: Defensive programming - ensures reference model exists before analysis. Falls back to single-image reference if needed.

```python
results = self.detect_anomalies_comprehensive(image_path)
```

**For beginners**: Runs the comprehensive anomaly detection and stores results.

**For developers**: Delegates to core detection method. Returns comprehensive analysis dictionary.

```python
if results:
    pipeline_report = self._convert_to_pipeline_format(results, image_path)
```

**For beginners**: If analysis succeeded, convert results to the format expected by the pipeline system.

**For developers**: Transforms internal result structure to match pipeline API contract.

### Saving Results

```python
report_path = output_path / f"{Path(image_path).stem}_report.json"
with open(report_path, 'w') as f:
    json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
```

**For beginners**:
- Creates output filename using the input filename (without extension) + "_report.json"
- Opens file for writing ('w')
- Saves the report as JSON with nice formatting (indent=2)
- Uses our custom encoder for NumPy types

**For developers**: 
- Path division operator for path joining
- Context manager ensures file closure
- Custom encoder handles NumPy serialization

### Visualization Generation

```python
if self.config.enable_visualization:
    viz_path = output_path / f"{Path(image_path).stem}_analysis.png"
    self.visualize_comprehensive_results(results, str(viz_path))
```

**For beginners**: If visualization is enabled, create analysis images showing detected problems.

**For developers**: Conditional visualization based on config flag. Converts Path back to string for visualization method.

### Defect Mask Creation

```python
mask_path = output_path / f"{Path(image_path).stem}_defect_mask.npy"
defect_mask = self._create_defect_mask(results)
np.save(mask_path, defect_mask)
```

**For beginners**: 
- Creates a black and white image where white shows defects
- Saves as NumPy array file (.npy) for later processing

**For developers**: Binary mask generation for downstream processing. NPY format preserves array structure efficiently.

---

## Converting Results to Pipeline Format

```python
def _convert_to_pipeline_format(self, results: Dict, image_path: str) -> Dict:
    """Convert internal results format to pipeline-expected format"""
```

**For beginners**: Takes our internal analysis results and reformats them to match what the larger system expects.

**For developers**: Adapter pattern - transforms internal representation to external API contract.

### Processing Anomaly Regions

```python
defects = []
defect_id = 1

for region in results['local_analysis']['anomaly_regions']:
    x, y, w, h = region['bbox']
    cx, cy = region['centroid']
```

**For beginners**:
- Creates empty list for defects
- For each anomaly region found:
  - Unpacks bounding box (x,y position and width,height)
  - Gets center point

**For developers**: Iterates anomaly regions, extracting bounding box via tuple unpacking. Centroid provides center of mass.

```python
confidence = region['confidence']
severity = self._confidence_to_severity(confidence)
```

**For beginners**: Gets how confident we are about this anomaly and converts it to a severity level (CRITICAL, HIGH, etc.).

**For developers**: Maps continuous confidence scores to discrete severity classifications.

### Creating Defect Dictionary

```python
defect = {
    'defect_id': f"ANOM_{defect_id:04d}",
    'defect_type': 'ANOMALY',
    'location_xy': [cx, cy],
    'bbox': [x, y, w, h],
    'area_px': region['area'],
    'confidence': float(confidence),
    'severity': severity,
    'orientation': None,
    'contributing_algorithms': ['ultra_comprehensive_matrix_analyzer'],
    'detection_metadata': {
        'max_intensity': region.get('max_intensity', 0),
        'anomaly_score': float(confidence)
    }
}
```

**For beginners**: Creates a detailed record for each defect:
- ID like "ANOM_0001" (the `:04d` means 4 digits with leading zeros)
- Type, location, size, confidence, severity
- Which algorithm found it
- Additional details

**For developers**: Structured defect representation following pipeline schema. Format specifiers ensure consistent ID formatting.

---

## Feature Extraction - The Heart of Analysis

Feature extraction is where we measure hundreds of characteristics about an image. Think of it like a doctor checking vital signs - we measure many different things to understand the image's "health".

### Statistical Features

```python
def _extract_statistical_features(self, gray):
    """Extract comprehensive statistical features."""
    flat = gray.flatten()
    percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
```

**For beginners**:
- `flatten()` converts the 2D image to a 1D list of pixel values
- `percentile` finds values where X% of pixels are darker

**For developers**: Flattening enables 1D statistical operations. Percentiles provide distribution insights.

#### Understanding Each Statistical Measure:

```python
'stat_mean': float(np.mean(gray))
```
**For beginners**: Average brightness of all pixels. If pixels range 0-255, a mean of 128 would be medium gray.

**For developers**: First moment of the pixel intensity distribution.

```python
'stat_std': float(np.std(gray))
```
**For beginners**: Standard deviation - how much pixel values vary from the average. Low = uniform image, High = lots of contrast.

**For developers**: Second moment, measures distribution spread.

```python
'stat_skew': float(self._compute_skewness(flat))
```
**For beginners**: Skewness tells if the image is generally darker (negative skew) or lighter (positive skew) than average.

**For developers**: Third standardized moment, measures distribution asymmetry.

```python
'stat_kurtosis': float(self._compute_kurtosis(flat))
```
**For beginners**: Kurtosis measures if the image has lots of extreme values (very dark or very bright pixels).

**For developers**: Fourth standardized moment minus 3, measures tail weight of distribution.

### Understanding Entropy

```python
def _compute_entropy(self, data, bins=256):
    hist, _ = np.histogram(data, bins=bins, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist + 1e-10))
```

**For beginners**: Entropy measures randomness or information content:
1. Count how many pixels have each brightness (histogram)
2. Convert counts to probabilities
3. Apply Shannon's entropy formula

High entropy = complex image with many different values
Low entropy = simple image with few different values

**For developers**: Shannon entropy implementation:
- Histogram with 256 bins for 8-bit images
- Normalize to probability distribution
- Remove zero bins to avoid log(0)
- Small epsilon (1e-10) prevents numerical errors

### Local Binary Patterns (LBP)

```python
def _extract_lbp_features(self, gray):
    for radius in [1, 2, 3, 5]:
        lbp = np.zeros_like(gray, dtype=np.float32)
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                
                shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                lbp += (shifted >= gray).astype(np.float32)
```

**For beginners**: LBP looks at each pixel and its neighbors:
1. For each neighbor, check if it's brighter than the center
2. Count how many neighbors are brighter
3. This creates a pattern that describes local texture

**For developers**: Custom LBP implementation:
- Multiple scales (radius 1,2,3,5)
- Uses numpy roll for efficient neighbor access
- Binary comparison creates rotation-invariant texture descriptor

### Gray-Level Co-occurrence Matrix (GLCM)

```python
def _extract_glcm_features(self, gray):
    img_q = (gray // 32).astype(np.uint8)
    levels = 8
```

**For beginners**: GLCM measures texture by looking at pixel pairs:
- Quantize image to 8 levels (0-7) for faster computation
- Count how often each gray level appears next to each other

**For developers**: GLCM texture analysis:
- Quantization reduces computational complexity
- Integer division creates 8 gray levels
- Analyzes second-order statistics

### Fourier Transform Features

```python
def _extract_fourier_features(self, gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    power = magnitude**2
    phase = np.angle(fshift)
```

**For beginners**: Fourier transform reveals patterns in the image:
- Converts image to frequency domain (like finding musical notes in a song)
- Magnitude = strength of each frequency
- Phase = timing of each frequency
- Power = magnitude squared

**For developers**: 2D FFT analysis:
- FFT2 for 2D discrete Fourier transform
- fftshift moves DC component to center
- Complex output split into magnitude and phase
- Power spectrum for energy distribution

---

## Anomaly Detection Process

### Building the Reference Model

```python
def build_comprehensive_reference_model(self, ref_dir):
    """Build an exhaustive reference model from a directory of JSON/image files."""
```

**For beginners**: This learns what "normal" looks like by analyzing many good fiber optic images.

**For developers**: Constructs statistical model from reference samples for anomaly detection baseline.

The process:
1. Load all reference images
2. Extract features from each
3. Compute statistics (mean, covariance)
4. Learn anomaly thresholds

### Robust Statistics

```python
def _compute_robust_statistics(self, data):
    robust_mean = np.median(data, axis=0)
    deviations = data - robust_mean
    mad = np.median(np.abs(deviations), axis=0)
    mad_scaled = mad * 1.4826
```

**For beginners**: 
- Uses median instead of mean (less affected by outliers)
- MAD = Median Absolute Deviation (robust measure of spread)
- 1.4826 converts MAD to be comparable with standard deviation

**For developers**: Robust estimators resistant to outliers:
- Median for location parameter
- MAD for scale parameter
- Consistency factor 1.4826 for normal distribution equivalence

### Mahalanobis Distance

```python
diff = test_vector - stat_model['robust_mean']
mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
```

**For beginners**: Mahalanobis distance is like regular distance but accounts for correlations:
- Regular distance: How far apart are two points?
- Mahalanobis: How far apart considering the data's shape?

**For developers**: Multivariate distance metric:
- Accounts for feature correlations via covariance matrix
- Scale-invariant distance measure
- Core metric for multivariate anomaly detection

### Local Anomaly Detection

```python
def _compute_local_anomaly_map(self, test_img, reference_img):
    for win_size in window_sizes:
        stride = win_size // 2
        
        for y in range(0, h - win_size + 1, stride):
            for x in range(0, w - win_size + 1, stride):
                test_win = test_img[y:y+win_size, x:x+win_size]
                ref_win = reference_img[y:y+win_size, x:x+win_size]
```

**For beginners**: Slides windows of different sizes across the image:
- Compares small patches between test and reference
- Overlapping windows (stride = half window size)
- Multiple scales catch defects of different sizes

**For developers**: Multi-scale sliding window approach:
- 50% overlap for dense coverage
- Scale pyramid (16, 32, 64 pixels)
- Local SSIM approximation for structural comparison

---

## Specific Defect Detection

### Scratch Detection

```python
edges = cv2.Canny(gray, 30, 100)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                       minLineLength=20, maxLineGap=5)
```

**For beginners**:
- Canny finds edges (sharp brightness changes)
- HoughLinesP finds straight lines in the edges
- Parameters control sensitivity and minimum line length

**For developers**: 
- Canny edge detection with hysteresis thresholds
- Probabilistic Hough transform for line segments
- Parameters tuned for scratch-like features

### Dig Detection

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
```

**For beginners**:
- Black-hat finds dark spots smaller than the kernel
- Like finding valleys in a landscape

**For developers**:
- Morphological black-hat: (closed - original)
- Elliptical structuring element for isotropic response
- Isolates dark features smaller than kernel

---

## Visualization

### Creating Comprehensive Visualization

```python
fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
```

**For beginners**: Creates a large figure (24x16 inches) with a 3x4 grid for multiple panels.

**For developers**: GridSpec for complex subplot layouts with controlled spacing.

Each panel shows different aspects:
1. **Original Test Image**: What we're analyzing
2. **Reference Archetype**: What "normal" looks like
3. **SSIM Map**: Structural similarity at each pixel
4. **Anomaly Heatmap**: Where anomalies are likely
5. **Detected Anomalies**: Blue boxes around problems
6. **Specific Defects**: Color-coded by type
7. **Feature Deviations**: Bar chart of abnormal measurements
8. **Analysis Summary**: Text summary of findings

---

## Report Generation

```python
def generate_detailed_report(self, results, output_path):
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ULTRA-COMPREHENSIVE ANOMALY DETECTION REPORT\n")
```

**For beginners**: Creates a text file with detailed findings, formatted nicely with headers and sections.

**For developers**: Structured text report generation with consistent formatting.

---

## Error Handling and Robustness

Throughout the code, you'll see patterns like:

```python
if std == 0:
    return 0.0
```

**For beginners**: Prevents division by zero errors.

**For developers**: Defensive programming against edge cases.

```python
try:
    # risky operation
except Exception as e:
    self.logger.warning(f"Operation failed: {e}")
    # fallback behavior
```

**For beginners**: Try to do something, but if it fails, log the error and do something else.

**For developers**: Exception handling with logging and graceful degradation.

---

## Main Execution

```python
if __name__ == "__main__":
    main()
```

**For beginners**: This only runs if the script is executed directly, not when imported by another script.

**For developers**: Standard Python idiom for script/module duality.

---

## Deep Dive: Mathematical Concepts Explained

### Understanding Standard Deviation and Z-Scores

```python
z_scores = np.abs(diff) / (stat_model['std'] + 1e-10)
```

**For beginners**: 
- Standard deviation measures spread: If heights average 5'10" with std of 2", then 68% of people are between 5'8" and 6'0"
- Z-score tells how many standard deviations from average: Z=2 means you're 2 standard deviations away
- The `+ 1e-10` prevents division by zero (1e-10 = 0.0000000001)

**Mathematical explanation**:
```
Z = (X - μ) / σ
Where:
- X = observed value
- μ (mu) = population mean
- σ (sigma) = population standard deviation
```

### Understanding Covariance and Correlation

```python
robust_cov = np.dot(weighted_data.T, weighted_data)
```

**For beginners**: 
- Covariance measures how two things change together
- If temperature goes up when ice cream sales go up, they have positive covariance
- The covariance matrix shows this for all feature pairs

**Mathematical explanation**:
```
Cov(X,Y) = E[(X - μx)(Y - μy)]
```
For a matrix, each element (i,j) is the covariance between feature i and feature j.

### Understanding Eigenvalues and Eigenvectors

```python
eigenvalues, eigenvectors = np.linalg.eigh(robust_cov)
eigenvalues = np.maximum(eigenvalues, 1e-6)
```

**For beginners**:
- Imagine stretching a rubber sheet - eigenvectors are the directions of stretch
- Eigenvalues are how much stretch in each direction
- Ensuring positive eigenvalues keeps our math valid

**Mathematical explanation**:
For a matrix A and vector v: Av = λv
- v is an eigenvector (direction)
- λ (lambda) is an eigenvalue (magnitude)

### Understanding Convolution and Kernels

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
```

**For beginners**:
- A kernel is like a stamp or cookie cutter
- We slide it over the image and perform operations
- Ellipse shape looks at circular neighborhoods

**Visual representation**:
```
Elliptical kernel (7x7):
  0 0 1 1 1 0 0
  0 1 1 1 1 1 0
  1 1 1 1 1 1 1
  1 1 1 1 1 1 1
  1 1 1 1 1 1 1
  0 1 1 1 1 1 0
  0 0 1 1 1 0 0
```

### Understanding SSIM (Structural Similarity)

```python
ssim_map = luminance * contrast * structure
```

**For beginners**: SSIM measures image similarity in three ways:
1. **Luminance**: Are they similarly bright?
2. **Contrast**: Do they have similar light/dark variation?
3. **Structure**: Do the patterns match?

**Mathematical formula**:
```
SSIM(x,y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ

Where:
- l(x,y) = (2μxμy + C1)/(μx² + μy² + C1) [luminance]
- c(x,y) = (2σxσy + C2)/(σx² + σy² + C2) [contrast]
- s(x,y) = (σxy + C3)/(σxσy + C3) [structure]
```

### Understanding the Hough Transform

```python
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                       minLineLength=20, maxLineGap=5)
```

**For beginners**: 
- Imagine drawing all possible lines through edge points
- Where many lines intersect in "line space" indicates a real line in the image
- It's like a voting system for lines

**Mathematical concept**:
A line y = mx + b becomes a point (m,b) in Hough space
Or using polar coordinates: ρ = x·cos(θ) + y·sin(θ)

## Deep Dive: Algorithm Workflows

### The Complete Anomaly Detection Workflow

1. **Image Loading**:
   ```
   File → OpenCV/JSON → NumPy Array → Grayscale Conversion
   ```

2. **Feature Extraction Pipeline**:
   ```
   Image → Statistical Features → Texture Features → 
   Frequency Features → Shape Features → Feature Vector
   ```

3. **Comparison Pipeline**:
   ```
   Test Features → Compare with Each Reference → 
   Statistical Distances → Aggregate Score → Threshold
   ```

4. **Local Analysis Pipeline**:
   ```
   Image → Sliding Windows → Local Comparisons → 
   Anomaly Map → Region Detection → Defect Classification
   ```

### The Learning Process

When building the reference model:

```python
for i in range(len(all_features)):
    for j in range(i + 1, len(all_features)):
        comp = self.compute_exhaustive_comparison(all_features[i], all_features[j])
```

**For beginners**: 
- Compares every reference image with every other reference image
- Learns how much "normal" images can differ from each other
- Sets thresholds based on these normal variations

**For developers**: 
- O(n²) pairwise comparisons for threshold learning
- Statistical modeling of intra-class variation
- Percentile-based threshold determination

## Practical Applications and Use Cases

### When Each Feature Type Helps

1. **Statistical Features** detect:
   - Overall brightness problems
   - Contrast issues
   - Unusual distributions

2. **Texture Features** (LBP, GLCM) detect:
   - Surface roughness changes
   - Pattern irregularities
   - Micro-scratches

3. **Frequency Features** detect:
   - Periodic patterns
   - High-frequency noise
   - Blur or focus issues

4. **Shape Features** detect:
   - Geometric distortions
   - Asymmetries
   - Overall shape changes

5. **Morphological Features** detect:
   - Small bright/dark spots
   - Thin scratches
   - Contamination particles

### Performance Considerations

**For large-scale deployment**:
- Feature extraction is parallelizable
- Sliding window can use GPU acceleration
- Reference model can be pre-computed
- Results can be cached

**Memory usage**:
- Scales with image size: O(width × height)
- Feature vector fixed size: ~100-200 floats
- Reference model: O(n_references × n_features)

## Summary

This script implements a sophisticated anomaly detection system using:

1. **Statistical Learning**: Builds a model of "normal" from reference images
2. **Feature Engineering**: Extracts 100+ measurements from each image
3. **Multi-method Detection**: 
   - Global statistical comparison
   - Local patch-based analysis
   - Specific defect detection
4. **Comprehensive Reporting**: JSON, visualizations, and text reports

The system is designed to be:
- **Robust**: Handles various image formats and edge cases
- **Configurable**: Extensive configuration options
- **Integrable**: Designed for pipeline integration
- **Comprehensive**: Multiple detection methods for high accuracy

For beginners: This is like having a very thorough quality inspector that never gets tired and always checks everything the same way.

For developers: A production-ready anomaly detection system combining classical computer vision with statistical learning, designed for integration into larger quality control pipelines.

---

## Line-by-Line Deep Dive Examples

Let's examine some complete methods line by line to understand every detail:

### Example 1: The `_compute_entropy` Method

```python
def _compute_entropy(self, data, bins=256):
```
- `def`: Keyword to define a function
- `_compute_entropy`: Method name (underscore indicates internal/private method)
- `self`: Reference to the current instance (required for class methods)
- `data`: Input parameter - the image data to analyze
- `bins=256`: Optional parameter with default value 256 (for 8-bit images)

```python
    hist, _ = np.histogram(data, bins=bins, range=(0, 256))
```
- `hist, _`: Unpacking tuple - we want first value, ignore second (underscore is convention for unused)
- `np.histogram`: NumPy function that counts values in bins
- `data`: Our input data
- `bins=bins`: Number of bins (using parameter value)
- `range=(0, 256)`: Count values from 0 to 255 (pixel brightness range)
- Result: `hist` contains counts of pixels at each brightness level

```python
    hist = hist / (hist.sum() + 1e-10)
```
- `hist.sum()`: Total count of all pixels
- `+ 1e-10`: Add tiny value (0.0000000001) to prevent division by zero
- `/`: Division converts counts to probabilities
- Result: `hist` now contains probability of each brightness level

```python
    hist = hist[hist > 0]
```
- `hist > 0`: Creates boolean array (True where hist is positive)
- `hist[...]`: Array indexing - keeps only positive values
- Purpose: Remove zero bins (can't take log of zero)

```python
    return -np.sum(hist * np.log2(hist + 1e-10))
```
- `np.log2`: Logarithm base 2 (information theory standard)
- `hist + 1e-10`: Prevent log(0) which is undefined
- `hist * np.log2(...)`: Element-wise multiplication
- `np.sum`: Add all values together
- `-`: Negative sign (entropy formula convention)
- `return`: Send result back to caller

**Complete formula being implemented**: H(X) = -Σ p(x) log₂(p(x))

### Example 2: The `_find_anomaly_regions` Method (First Part)

```python
def _find_anomaly_regions(self, anomaly_map, original_shape):
    """Find distinct anomaly regions from the anomaly map."""
```
- Method takes anomaly heatmap and original image dimensions
- Returns list of detected anomaly regions

```python
    positive_values = anomaly_map[anomaly_map > 0]
```
- `anomaly_map > 0`: Boolean mask where anomaly scores are positive
- `anomaly_map[...]`: Fancy indexing - extracts only positive values
- Result: 1D array of all positive anomaly scores

```python
    if positive_values.size == 0:
        return []
```
- `.size`: Number of elements in array
- `== 0`: Check if empty
- `return []`: Return empty list if no anomalies

```python
    threshold = np.percentile(positive_values, 80)
```
- `np.percentile`: Find value where 80% of data is below
- `80`: Keep top 20% most anomalous regions
- Purpose: Adaptive thresholding based on data

```python
    binary_map = (anomaly_map > threshold).astype(np.uint8)
```
- `anomaly_map > threshold`: Boolean array (True/False)
- `.astype(np.uint8)`: Convert True→1, False→0
- Result: Binary image where 1 = anomaly, 0 = normal

```python
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
```
- `cv2.connectedComponentsWithStats`: Find connected regions
- `connectivity=8`: Pixels connected by edges or corners (8-way)
- Returns:
  - `num_labels`: Number of regions found (including background)
  - `labels`: Image where each pixel has region ID
  - `stats`: Bounding box and area for each region
  - `centroids`: Center point of each region

### Example 3: Complete Analysis of `_confidence_to_severity`

```python
def _confidence_to_severity(self, confidence: float) -> str:
    """Convert confidence score to severity level"""
```
- Type hints: `confidence: float` (input is decimal), `-> str` (returns text)

```python
    for severity, threshold in sorted(self.config.severity_thresholds.items(), 
                                    key=lambda x: x[1], reverse=True):
```
Breaking this complex line down:
- `self.config.severity_thresholds.items()`: Get (name, value) pairs from dictionary
- `sorted(...)`: Sort the pairs
- `key=lambda x: x[1]`: Sort by second element (threshold value)
- `reverse=True`: Highest thresholds first
- `for severity, threshold in`: Unpack each pair into two variables

Example iteration order:
1. ('CRITICAL', 0.9)
2. ('HIGH', 0.7)
3. ('MEDIUM', 0.5)
4. ('LOW', 0.3)
5. ('NEGLIGIBLE', 0.1)

```python
        if confidence >= threshold:
            return severity
```
- Check if confidence meets this severity level
- Return immediately when found (stops checking lower levels)
- Example: confidence=0.75 returns 'HIGH' (first threshold ≤ 0.75)

```python
    return 'NEGLIGIBLE'
```
- Fallback if confidence below all thresholds
- Guarantees method always returns a value

### Example 4: Understanding Numpy Operations

Let's break down this line from the Mahalanobis distance calculation:

```python
mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
```

Step by step:
1. `diff`: Vector of differences from mean (shape: [n_features])
2. `diff.T`: Transpose (shape: [1, n_features]) - row vector
3. `@`: Matrix multiplication operator (Python 3.5+)
4. `stat_model['robust_inv_cov']`: Inverse covariance matrix
5. `diff.T @ stat_model['robust_inv_cov']`: First multiplication
6. `... @ diff`: Second multiplication (results in scalar)
7. `np.abs(...)`: Absolute value (handle numerical errors)
8. `np.sqrt(...)`: Square root for final distance

**Mathematical notation**: d = √[(x-μ)ᵀ Σ⁻¹ (x-μ)]

### Example 5: Understanding List Comprehensions

```python
deviant_features = [(feature_names[i], z_scores[i], test_vector[i], stat_model['mean'][i]) 
                   for i in top_indices]
```

Equivalent to:
```python
deviant_features = []
for i in top_indices:
    feature_tuple = (
        feature_names[i],      # Feature name
        z_scores[i],          # How deviant
        test_vector[i],       # Test value
        stat_model['mean'][i] # Reference value
    )
    deviant_features.append(feature_tuple)
```

### Example 6: Understanding Context Managers

```python
with open(report_path, 'w') as f:
    json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
```

Equivalent to:
```python
f = open(report_path, 'w')  # Open file for writing
try:
    json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
finally:
    f.close()  # Always close, even if error occurs
```

### Example 7: Understanding Numpy Broadcasting

```python
local_score = max(local_score, 1 - ssim_approx)
```

When arrays have different shapes, NumPy "broadcasts" to make them compatible:
- If `local_score` is a single value and `1 - ssim_approx` is an array
- Operation applies element-wise
- Result has shape of larger array

### Example 8: Understanding Morphological Operations

```python
wth = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
```

What happens internally:
1. `opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)`
   - Erosion followed by dilation
   - Removes small bright spots
2. `wth = gray - opening`
   - Subtracts opened image from original
   - Leaves only the small bright spots that were removed

### Common Patterns Explained

#### Pattern 1: Safe Division
```python
value = numerator / (denominator + 1e-10)
```
- Always add small epsilon to prevent division by zero
- Common in scientific computing

#### Pattern 2: Dictionary Get with Default
```python
region.get('max_intensity', 0)
```
- Try to get 'max_intensity' from dictionary
- Return 0 if key doesn't exist
- Prevents KeyError exceptions

#### Pattern 3: Array Slicing
```python
mask[y:y+h, x:x+w] = 255
```
- `y:y+h`: Rows from y to y+h (exclusive)
- `x:x+w`: Columns from x to x+w (exclusive)
- `= 255`: Set all pixels in region to white

#### Pattern 4: Conditional Expression
```python
severity = 'MEDIUM' if blob['area'] > 500 else 'LOW'
```
- Equivalent to:
```python
if blob['area'] > 500:
    severity = 'MEDIUM'
else:
    severity = 'LOW'
```

## Final Notes

This code demonstrates several advanced programming concepts:

1. **Object-Oriented Design**: Classes encapsulate functionality
2. **Defensive Programming**: Extensive error checking
3. **Scientific Computing**: NumPy for efficient array operations
4. **Computer Vision**: OpenCV for image processing
5. **Statistical Analysis**: Multiple statistical measures
6. **Machine Learning**: Learns from examples (reference images)
7. **Software Engineering**: Logging, configuration, modularity

The beauty of this code is how it combines multiple disciplines to solve a real-world problem - automated quality control for fiber optic manufacturing.