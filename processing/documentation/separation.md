# Complete Guide to the Fiber Optic Segmentation System

## Table of Contents
1. [Overview and Purpose](#overview)
2. [Understanding the Problem Domain](#problem-domain)
3. [System Architecture](#architecture)
4. [Line-by-Line Code Explanation](#code-explanation)
5. [Mathematical Concepts](#mathematical-concepts)
6. [Key Algorithms](#algorithms)
7. [Practical Examples](#examples)

---

## 1. Overview and Purpose {#overview}

This code implements a sophisticated system for analyzing images of fiber optic cables. The goal is to automatically identify and separate three distinct regions in these images:

- **Core**: The innermost part of the fiber where light travels
- **Cladding**: The layer surrounding the core that keeps light confined
- **Ferrule**: The outer protective/connector material

### Why This Matters

In fiber optic manufacturing and quality control, precise measurement of these regions is crucial. Manual inspection is time-consuming and error-prone. This system automates the process using multiple computer vision techniques and combines their results for accuracy.

### The Key Innovation

Instead of relying on a single segmentation method, this system:
1. Runs 11 different segmentation algorithms in parallel
2. Combines their results using a sophisticated consensus algorithm
3. Learns from past performance to weight more accurate methods higher
4. Handles image defects through pre-processing

---

## 2. Understanding the Problem Domain {#problem-domain}

### What is Image Segmentation?

Image segmentation is the process of dividing an image into meaningful regions. Think of it like:
- Coloring different parts of a coloring book
- Separating foreground from background in a photo
- Identifying different organs in a medical scan

### The Fiber Optic Challenge

Fiber optic images present unique challenges:
- The boundaries between regions can be subtle
- Images may have defects or artifacts
- Lighting conditions vary
- Different fiber types have different characteristics

---

## 3. System Architecture {#architecture}

The system consists of several key components:

```
UnifiedSegmentationSystem (Main Orchestrator)
    ├── Multiple Segmentation Methods (11 different approaches)
    ├── EnhancedConsensusSystem (Combines results)
    ├── Performance Tracking (Learning from results)
    └── Pre-processing Pipeline (Defect removal)
```

---

## 4. Line-by-Line Code Explanation {#code-explanation}

### Import Section (Lines 1-17)

```python
import os  # Provides functions for interacting with the operating system (file paths, environment variables)
```

**What this does**: The `os` module lets Python interact with your computer's operating system. It's like giving Python the ability to:
- Navigate folders (like Windows Explorer or Mac Finder)
- Read environment variables (system settings)
- Check if files exist

**For beginners**: Think of it as Python's way to talk to your computer's file system.

**For developers**: Provides OS-agnostic file system operations and process environment access.

```python
import sys  # Provides access to system-specific parameters and functions (command line args, python executable path)
```

**What this does**: The `sys` module provides access to Python interpreter settings and command-line arguments.

**For beginners**: If you run the program with extra information (like `python program.py folder_name`), `sys` helps read that "folder_name" part.

**For developers**: Enables access to argv for CLI parameters and sys.executable for subprocess Python invocations.

```python
import json  # Enables encoding/decoding of JSON data for saving/loading results and configuration
```

**What this does**: JSON (JavaScript Object Notation) is a way to store data in a human-readable format.

**For beginners**: It's like saving your data in a structured text file that both humans and computers can read easily. Like:
```json
{
    "name": "fiber_image_001",
    "core_radius": 25.5,
    "confidence": 0.95
}
```

**For developers**: Provides serialization/deserialization for data persistence and inter-process communication.

```python
import time  # Provides time-related functions for measuring execution duration and timestamps
```

**What this does**: Lets the program measure how long things take.

**For beginners**: Like using a stopwatch to time how long each analysis method takes.

**For developers**: Used for performance profiling and benchmarking individual method execution times.

```python
import numpy as np  # Numerical computing library for efficient array operations and mathematical computations
```

**What this does**: NumPy is Python's powerhouse for mathematical operations on large arrays of numbers.

**For beginners**: Instead of processing pixels one by one (slow), NumPy processes entire images at once (fast). It's like the difference between counting coins one by one vs. using a coin counting machine.

**For developers**: Provides vectorized operations on n-dimensional arrays with C-level performance.

```python
import cv2  # OpenCV library for computer vision operations (image loading, processing, morphological operations)
```

**What this does**: OpenCV (Open Computer Vision) is a library specialized in image processing.

**For beginners**: It's like Photoshop for Python - it can load images, apply filters, detect shapes, etc.

**For developers**: Industry-standard computer vision library providing optimized implementations of image processing algorithms.

```python
from pathlib import Path  # Object-oriented filesystem path handling for cross-platform compatibility
```

**What this does**: Path makes working with file paths easier and works the same on Windows, Mac, and Linux.

**For beginners**: Instead of worrying about forward slashes (/) vs backslashes (\), Path handles it automatically.

**For developers**: Provides a high-level, platform-agnostic API for path manipulation with rich comparison and manipulation methods.

```python
import subprocess  # Enables running external Python scripts in isolated processes for method execution
```

**What this does**: Allows the program to run other Python scripts as separate processes.

**For beginners**: It's like having the main program be a manager that delegates tasks to workers (other scripts), protecting itself if a worker crashes.

**For developers**: Enables process isolation for fault tolerance - if a segmentation method crashes, it doesn't take down the main system.

```python
import tempfile  # Creates temporary directories and files for intermediate processing results
```

**What this does**: Creates temporary files that are automatically cleaned up.

**For beginners**: Like using scratch paper that gets thrown away when you're done.

**For developers**: Provides secure temporary file creation with automatic cleanup on context exit.

```python
from typing import Dict, List, Tuple, Optional, Any  # Type hints for better code documentation and IDE support
```

**What this does**: Type hints tell other programmers (and tools) what kind of data functions expect and return.

**For beginners**: It's like labeling boxes - "This function takes a list of numbers and returns a dictionary."

**For developers**: Enables static type checking, improves IDE intellisense, and serves as inline documentation.

```python
import warnings  # Controls warning messages from libraries
warnings.filterwarnings('ignore')  # Suppresses all warning messages to keep console output clean
```

**What this does**: Hides warning messages that might clutter the output.

**For beginners**: Like putting your phone on silent during a meeting.

**For developers**: Suppresses non-critical warnings from dependencies to maintain clean console output.

### Matplotlib Import Section (Lines 19-27)

```python
try:  # Attempt to import matplotlib for creating summary visualizations
    import matplotlib  # Main matplotlib package for plotting
    matplotlib.use('Agg')  # Sets backend to non-interactive mode for server/headless environments
    import matplotlib.pyplot as plt  # Pyplot interface for creating figures and plots
    HAS_MATPLOTLIB = True  # Flag indicating matplotlib is available for visualization features
except ImportError:  # Handle case where matplotlib is not installed
    HAS_MATPLOTLIB = False  # Disable visualization features that require matplotlib
    print("Warning: matplotlib not available, some visualizations will be skipped")
```

**What this does**: Tries to import matplotlib (plotting library) but continues if it's not installed.

**For beginners**: The program checks if it has access to a drawing tool. If not, it continues without making pictures but warns you.

**For developers**: Implements graceful degradation - core functionality works without optional dependencies.

**The Backend Setting**: `matplotlib.use('Agg')` tells matplotlib to create images without trying to display them on screen, important for server environments.

### SciPy Import Section (Lines 29-36)

```python
try:  # Attempt to import advanced image processing functions from scipy
    from scipy.ndimage import median_filter, gaussian_filter  # Image filtering functions for noise reduction
    from scipy.ndimage import binary_opening, binary_closing  # Morphological operations for cleaning binary masks
    HAS_SCIPY_FULL = True  # Flag indicating full scipy functionality is available
except ImportError:  # Handle case where scipy is not installed or partially installed
    HAS_SCIPY_FULL = False  # Disable advanced post-processing features
    print("Warning: Some scipy components not available, using basic post-processing")
```

**What this does**: Tries to import advanced image processing functions.

**For beginners**: 
- `median_filter`: Removes "salt and pepper" noise (random black/white dots)
- `gaussian_filter`: Blurs images to smooth out rough edges
- `binary_opening/closing`: Cleans up shapes by removing small bumps or filling small holes

**For developers**: Morphological operations for noise reduction and mask refinement. Falls back to basic processing if unavailable.

### NumpyEncoder Class (Lines 38-47)

```python
class NumpyEncoder(json.JSONEncoder):  # Custom JSON encoder to handle NumPy data types
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):  # Override default method to handle NumPy types
        if isinstance(obj, (np.integer, np.int_)):  # Check if object is NumPy integer type
            return int(obj)  # Convert NumPy integer to Python int for JSON compatibility
        if isinstance(obj, (np.floating, np.float_)):  # Check if object is NumPy float type
            return float(obj)  # Convert NumPy float to Python float for JSON compatibility
        if isinstance(obj, np.ndarray):  # Check if object is NumPy array
            return obj.tolist()  # Convert NumPy array to Python list for JSON compatibility
        return super(NumpyEncoder, self).default(obj)  # Fall back to default encoder for other types
```

**What this does**: Solves a technical problem - NumPy numbers can't be directly saved to JSON files.

**For beginners**: JSON files can only store regular Python numbers, not NumPy's special number types. This class acts as a translator.

**Example**:
- NumPy integer 42 → Regular Python integer 42
- NumPy array [1,2,3] → Regular Python list [1,2,3]

**For developers**: Extends JSONEncoder to handle NumPy types through type checking and conversion.

### EnhancedConsensusSystem Class (Lines 49-146)

This is the heart of the consensus algorithm - it combines results from multiple methods.

```python
class EnhancedConsensusSystem:  # Core consensus algorithm for combining multiple segmentation results
    """
    model aware voting system
    """
    def __init__(self, min_agreement_ratio=0.3):  # Initialize consensus system with configurable threshold
        self.min_agreement_ratio = min_agreement_ratio  # Minimum ratio of methods that must agree for consensus (0.3 = 30%)
```

**What this does**: Creates the consensus system with a configurable agreement threshold.

**For beginners**: If you have 10 different methods analyzing an image, at least 3 (30%) must agree on where the boundaries are.

**For developers**: Configurable consensus threshold allows tuning between strict agreement (high threshold) and permissive consensus (low threshold).

#### IoU Calculation Method (Lines 57-65)

```python
def _calculate_iou(self, mask1, mask2):  # Private method to compute Intersection over Union metric
    """Calculates Intersection over Union for two binary masks."""
    if mask1 is None or mask2 is None:  # Check if either mask is missing
        return 0.0  # Return zero overlap if either mask doesn't exist
    intersection = np.logical_and(mask1, mask2)  # Compute pixel-wise AND to find overlapping regions
    union = np.logical_or(mask1, mask2)  # Compute pixel-wise OR to find total coverage
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-6)  # Ratio of intersection to union area
    return iou_score  # Return IoU score between 0 (no overlap) and 1 (perfect overlap)
```

**What this does**: Measures how similar two masks are using the "Intersection over Union" (IoU) metric.

**For beginners**: Imagine two circles drawn on paper:
- Intersection = The area where both circles overlap
- Union = The total area covered by both circles
- IoU = Overlap area ÷ Total area

**Mathematical Example**:
- If two circles overlap perfectly: IoU = 1.0 (100% match)
- If they don't overlap at all: IoU = 0.0 (0% match)
- If they half overlap: IoU ≈ 0.33 (33% match)

**For developers**: Standard computer vision metric for segmentation quality. The `1e-6` prevents division by zero.

#### Generate Consensus Method (Lines 67-123)

This is the most complex method - it implements a 4-stage consensus algorithm:

```python
def generate_consensus(self,
                       results: List['SegmentationResult'],  # List of segmentation results from different methods
                       method_scores: Dict[str, float],  # Historical performance scores for each method
                       image_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:  # Image dimensions
```

**Stage 1: Preliminary Weighted Pixel Vote**

```python
# --- Stage 1: Preliminary Weighted Pixel Vote ---
weighted_votes = np.zeros((h, w, 3), dtype=np.float32)  # 3-channel array for core/cladding/ferrule votes
for r in valid_results:  # Iterate through each valid segmentation result
    weight = method_scores.get(r.method_name, 1.0) * r.confidence  # Combine historical score with result confidence
    if r.masks.get('core') is not None:  # Check if core mask exists
        weighted_votes[:, :, 0] += (r.masks['core'] > 0).astype(np.float32) * weight
```

**What this does**: Each pixel gets "votes" from different methods about what it should be (core, cladding, or ferrule).

**For beginners**: Imagine 10 people looking at a pixel and voting:
- 6 say "it's core" (each with different confidence)
- 3 say "it's cladding"
- 1 says "it's ferrule"
The system weighs these votes by how accurate each voter has been in the past.

**The Math**: For each pixel position (x,y):
- weighted_votes[x,y,0] = Sum of (core votes × method weights)
- weighted_votes[x,y,1] = Sum of (cladding votes × method weights)
- weighted_votes[x,y,2] = Sum of (ferrule votes × method weights)

**Stage 2: Identify High-Agreement Methods**

```python
preliminary_classification = np.argmax(weighted_votes, axis=2)  # Find winning class for each pixel
```

**What this does**: Determines the winning vote for each pixel.

**For beginners**: `argmax` finds which option (core=0, cladding=1, ferrule=2) got the most votes.

**The Math**: If at pixel (10,20):
- Core votes = 5.2
- Cladding votes = 3.1
- Ferrule votes = 1.7
Then `argmax` returns 0 (core wins)

**Stage 3: Parameter-Space Consensus**

```python
consensus_params = {'cx': [], 'cy': [], 'core_r': [], 'clad_r': []}  # Lists to collect geometric parameters
```

**What this does**: Instead of just voting on pixels, this stage averages the geometric parameters (center point, radii).

**For beginners**: If 5 methods say:
- Method 1: "Center is at (100,100), core radius is 25"
- Method 2: "Center is at (102,99), core radius is 24"
- etc.
The system averages these to get the best estimate.

**Stage 4: Generate Final Ideal Masks**

```python
final_masks = self.create_masks_from_params(  # Convert averaged parameters to binary masks
    final_center, final_core_radius, final_cladding_radius, image_shape
)
```

**What this does**: Creates perfect circular masks using the averaged parameters.

**For beginners**: Like using a compass to draw perfect circles at the calculated center with the calculated radii.

### SegmentationResult Class (Lines 148-166)

```python
class SegmentationResult:  # Container class for storing results from individual segmentation methods
    """Standardized result format for all segmentation methods"""
    def __init__(self, method_name: str, image_path: str):
        self.method_name = method_name  # Name of the segmentation method
        self.image_path = image_path  # Path to the processed image
        self.center = None  # Tuple of (x, y) center coordinates
        self.core_radius = None  # Radius of the fiber core in pixels
        self.cladding_radius = None  # Radius of the fiber cladding in pixels
        self.masks = None  # Dictionary containing core, cladding, and ferrule masks
        self.confidence = 0.5  # Confidence score for this segmentation (0-1)
        self.execution_time = 0.0  # Time taken to run this method in seconds
        self.error = None  # Error message if method failed
```

**What this does**: Defines a standard container for storing results from each segmentation method.

**For beginners**: Like a form that each method fills out with its findings:
- Where is the center? 
- How big is the core?
- How confident are you?
- Did anything go wrong?

**For developers**: Data class pattern for type safety and consistent result handling across heterogeneous methods.

### UnifiedSegmentationSystem Class (Lines 168-477)

This is the main orchestrator that manages everything.

#### Initialization (Lines 172-186)

```python
def __init__(self, methods_dir: str = "zones_methods"):
    self.methods_dir = Path(methods_dir)  # Convert to Path object for cross-platform compatibility
    self.output_dir = Path("output")  # Directory for saving results
    self.output_dir.mkdir(exist_ok=True)  # Create output directory if it doesn't exist
```

**What this does**: Sets up the system with necessary directories.

**For beginners**: 
- `methods_dir`: Where to find all the different analysis methods
- `output_dir`: Where to save the results
- `mkdir(exist_ok=True)`: Create the folder if it doesn't exist, don't complain if it does

**The Vulnerable Methods List**:
```python
self.vulnerable_methods = [  # List of methods that benefit from pre-processed (inpainted) images
    'adaptive_intensity', 'gradient_approach', 'guess_approach', 'threshold_separation', 'intelligent_segmenter'
]
```

**What this does**: Some methods work poorly with image defects, so they get cleaned images.

**For beginners**: Like giving some students a calculator for a math test because they struggle with mental math.

#### Knowledge Persistence (Lines 188-199)

```python
def load_knowledge(self):  # Load historical performance data from previous runs
    if self.knowledge_file.exists():  # Check if knowledge file exists
        try:
            with open(self.knowledge_file, 'r') as f:
                self.dataset_stats.update(json.load(f))  # Update stats with loaded data
```

**What this does**: The system remembers which methods performed well in the past.

**For beginners**: Like a teacher remembering which students are good at which subjects.

**For developers**: Implements persistent performance tracking for method weighting in consensus.

#### Method Loading (Lines 201-220)

```python
def load_methods(self):  # Discover and register available segmentation methods
    method_files = [  # List of expected method script filenames
        'adaptive_intensity.py', 'bright_core_extractor.py', 'computational_separation.py',
        # ... more methods
    ]
```

**What this does**: Finds and registers all available segmentation methods.

**For beginners**: Like taking attendance - checking which analysis tools are available.

**For developers**: Dynamic method discovery and registration with score initialization.

#### Defect Detection and Inpainting (Lines 222-230)

```python
def detect_and_inpaint_anomalies(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Create elliptical structuring element
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)  # Black-hat transform
```

**What this does**: Finds and fixes defects in images before analysis.

**For beginners**: 
1. Convert to grayscale (like a black & white photo)
2. Black-hat transform finds dark spots that shouldn't be there
3. Inpainting fills in these defects by guessing what should be there

**The Math**: 
- Black-hat = (Closed image) - (Original image)
- This highlights dark features smaller than the kernel

**For developers**: Morphological black-hat operation for anomaly detection followed by Telea inpainting.

#### Isolated Method Execution (Lines 232-289)

This is one of the most innovative parts - running each method in isolation:

```python
def run_method_isolated(self, method_name: str, image_path: Path, temp_output: Path) -> dict:
    """Generates a wrapper script to run a method in isolation and captures its JSON output."""
```

**What this does**: Runs each segmentation method in its own process.

**For beginners**: Like having each chef work in their own kitchen - if one burns their dish, it doesn't affect the others.

**The Process**:
1. Creates a temporary Python script
2. The script imports and runs the specific method
3. Results are saved to a JSON file
4. Main program reads the results

**Why Isolation Matters**:
- If a method crashes, it doesn't crash the whole system
- Methods can't interfere with each other
- Memory is cleaned up after each method

#### Performance Learning (Lines 337-356)

```python
def update_learning(self, consensus: Dict, all_results: List[SegmentationResult]):
    for result in all_results:
        if result.error or not result.masks: continue
        core_iou = self.consensus_system._calculate_iou(result.masks.get('core'), consensus_masks.get('core'))
        # ... calculate accuracy
        
        learning_rate = 0.1  # Learning rate for exponential moving average
        target_score = 0.1 + (1.9 * avg_iou)  # Map IoU [0,1] to score [0.1,2.0]
        new_score = current_score * (1 - learning_rate) + target_score * learning_rate
```

**What this does**: Updates each method's performance score based on how well it agreed with consensus.

**For beginners**: Like updating a student's grade - if they got the answer close to the class consensus, their credibility goes up.

**The Math (Exponential Moving Average)**:
- New Score = 90% of old score + 10% of current performance
- This smooths out random variations

**For developers**: Implements online learning with exponential smoothing for method weight adaptation.

### Visualization and Output (Lines 393-477)

#### Saving Results (Lines 393-425)

```python
def save_results(self, image_path: Path, consensus: Dict, image: np.ndarray, output_dir: str, defect_mask: np.ndarray):
    # Save JSON report
    report = {k: v for k, v in consensus.items() if k not in ['masks', 'all_results']}
    
    # Save mask images
    cv2.imwrite(str(result_dir / "mask_core.png"), masks['core'] * 255)
    
    # Apply masks to original image
    region_core = cv2.bitwise_and(image, image, mask=masks['core'])
```

**What this does**: Saves all results in multiple formats.

**For beginners**: 
- JSON report: A text file with all the measurements
- Mask images: Black and white images showing each region
- Region images: The original image with only one region visible

**The Bitwise AND Operation**:
- Where mask = 1 (white): Keep original pixel
- Where mask = 0 (black): Make pixel black
- Result: Only the masked region is visible

#### Summary Visualization (Lines 427-477)

```python
def create_summary_visualization(self, result_dir: Path, original_image: np.ndarray, ...):
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)  # Create 2x2 subplot grid
```

**What this does**: Creates a 4-panel summary image showing:
1. Original image with detected boundaries
2. Color-coded segmentation masks
3. Method performance table
4. Separated regions combined

**For beginners**: Like a medical report with multiple views of the same scan.

---

## 5. Mathematical Concepts {#mathematical-concepts}

### Intersection over Union (IoU)

The fundamental metric for comparing segmentations:

```
IoU = Area of Overlap / Area of Union
    = |A ∩ B| / |A ∪ B|
```

**Example with Numbers**:
- Region A has 100 pixels
- Region B has 120 pixels  
- They overlap by 80 pixels
- Union = 100 + 120 - 80 = 140 pixels
- IoU = 80/140 = 0.57 (57% similarity)

### Weighted Voting

Each pixel gets votes weighted by method confidence and historical accuracy:

```
Vote(pixel, class) = Σ (method_vote × method_score × method_confidence)
```

**Example**:
- Method A (score=0.9, confidence=0.8) votes "core"
- Method B (score=0.7, confidence=0.9) votes "cladding"
- Core vote = 0.9 × 0.8 = 0.72
- Cladding vote = 0.7 × 0.9 = 0.63
- Result: Pixel classified as "core"

### Exponential Moving Average

For updating method scores:

```
new_score = α × old_score + (1-α) × current_performance
```

Where α = 0.9 (learning rate = 0.1)

This gives 90% weight to historical performance and 10% to current result.

### Morphological Operations

**Black-hat Transform**:
```
black_hat(image) = closing(image) - image
```

This highlights dark features smaller than the structuring element.

**Binary Opening**:
```
opening = erosion followed by dilation
```
Removes small objects while preserving the shape of larger objects.

**Binary Closing**:
```
closing = dilation followed by erosion  
```
Fills small holes while preserving the shape of larger objects.

---

## 6. Key Algorithms {#algorithms}

### The 4-Stage Consensus Algorithm

1. **Pixel Voting Stage**
   - Each method votes on each pixel
   - Votes are weighted by confidence and historical accuracy
   - Preliminary masks are created

2. **Agreement Filtering Stage**
   - Methods are evaluated against preliminary masks
   - Only methods with >60% IoU are kept
   - This filters out outliers

3. **Parameter Averaging Stage**
   - Geometric parameters from agreeing methods are collected
   - Weighted average creates final parameters
   - This ensures smooth, circular regions

4. **Mask Generation Stage**
   - Perfect circular masks are generated
   - Morphological cleanup ensures consistency
   - Final masks are mutually exclusive

### The Isolation Strategy

Each method runs in a subprocess because:
- **Fault Tolerance**: Crashes don't affect other methods
- **Resource Management**: Memory is freed after each run
- **Environment Control**: Each method gets a clean environment

### The Learning Algorithm

Performance tracking uses:
- **Immediate Feedback**: Compare to consensus after each image
- **Smooth Updates**: Exponential averaging prevents overreaction
- **Persistent Storage**: Learning carries across sessions

---

## 7. Practical Examples {#examples}

### Example 1: Processing a Single Image

Let's trace through processing one fiber optic image:

1. **Image Loading**
   ```python
   original_img = cv2.imread("fiber_001.png")
   # Loads a 512×512 pixel image
   ```

2. **Defect Detection**
   - Black-hat transform finds 15 dark spots
   - Inpainting fills these with interpolated values
   - Clean image saved to temporary file

3. **Method Execution**
   - 11 methods run in parallel
   - Method "hough_separation" finds:
     - Center: (256, 258)
     - Core radius: 25 pixels
     - Cladding radius: 62.5 pixels
     - Confidence: 0.85
   - Takes 1.2 seconds

4. **Consensus Building**
   - 8 methods agree (IoU > 0.6)
   - Weighted average:
     - Center: (256.3, 257.8)
     - Core radius: 25.2 pixels
     - Cladding radius: 62.3 pixels

5. **Output Generation**
   - Masks saved as PNG files
   - JSON report with measurements
   - Summary visualization created

### Example 2: Learning Over Time

Initial scores (all methods start at 1.0):
```
Session 1: adaptive_intensity performs poorly
- IoU with consensus: 0.4
- New score: 0.9 × 1.0 + 0.1 × 0.85 = 0.985

Session 5: adaptive_intensity continues struggling  
- Average IoU: 0.45
- Score now: 0.92

Session 10: bright_core_extractor excels
- Average IoU: 0.95
- Score now: 1.85
```

The system now trusts bright_core_extractor nearly twice as much as adaptive_intensity.

### Example 3: Handling Edge Cases

**Missing Core Detection**:
```python
if all([result.center, result.core_radius, result.cladding_radius]):
    # Generate masks only if all parameters exist
```

**Non-Circular Shapes**:
```python
circularity = (4 * np.pi * area) / (perimeter**2)
if circularity < 0.85:
    result.confidence *= 0.5  # Penalize non-circular detections
```

**Complete Failure**:
- If < 2 methods succeed: No consensus possible
- System reports failure and continues to next image

---

## Summary

This system represents a sophisticated approach to a complex computer vision problem. Its key innovations are:

1. **Multi-method Ensemble**: Using 11 different approaches provides robustness
2. **Intelligent Consensus**: Not just voting, but smart parameter averaging
3. **Continuous Learning**: Performance improves over time
4. **Fault Tolerance**: Isolated execution prevents cascade failures
5. **Comprehensive Output**: Multiple formats serve different use cases

The code demonstrates advanced software engineering principles while solving a real-world problem in fiber optic quality control.
