# Fiber Optic Defect Detection Pipeline - Comprehensive README

## Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Pipeline Architecture](#pipeline-architecture)
- [Detailed Usage Instructions](#detailed-usage-instructions)
- [Component Details](#component-details)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

This is a comprehensive multi-stage pipeline for detecting and analyzing defects in fiber optic cable end-face images. The system uses advanced computer vision techniques to:

- Generate multiple image variations for robust analysis
- Segment fiber images into core, cladding, and ferrule regions
- Detect anomalies and specific defects in each region
- Aggregate results and provide quality assessment

### Key Features
- Multi-method consensus segmentation using 11 different algorithms
- Comprehensive defect detection including scratches, digs, contamination
- Intelligent data aggregation with clustering and deduplication
- Quality scoring and pass/fail determination
- Detailed visualization and reporting

## System Requirements

### Hardware Requirements
- **Minimum RAM:** 8GB (16GB recommended for batch processing)
- **Storage:** At least 1GB free space per image analyzed
- **Processor:** Multi-core processor recommended for faster processing

### Software Requirements
- **Python:** 3.8 or higher
- **Operating System:** Windows, Linux, or macOS

### Python Dependencies
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0
- scikit-learn >= 0.24.0
- pathlib (built-in for Python 3.4+)

## Installation

### Step 1: Clone or Download the Repository
```bash
# If using git
git clone <repository-url>
cd fiber-optic-defect-detection

# Or download and extract the ZIP file
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install opencv-python numpy matplotlib scipy scikit-learn
```

### Step 4: Verify Installation
```bash
python -c "import cv2, numpy, matplotlib, scipy, sklearn; print('All dependencies installed successfully')"
```

## File Structure
```
fiber-optic-defect-detection/
│
├── app.py                    # Main orchestrator script
├── config.json              # Configuration file
├── process.py               # Image transformation module
├── separation.py            # Region segmentation module
├── detection.py             # Defect detection module
├── data_acquisition.py      # Results aggregation module
│
├── zones_methods/           # Segmentation algorithms directory
│   ├── adaptive_intensity.py
│   ├── bright_core_extractor.py
│   ├── computational_separation.py
│   ├── geometric_approach.py
│   ├── gradient_approach.py
│   ├── guess_approach.py
│   ├── hough_separation.py
│   ├── segmentation.py
│   ├── threshold_separation.py
│   ├── unified_core_cladding_detector.py
│   └── intelligent_segmenter.py
│
└── processing/              # Results directory (created automatically)
    └── results/
```

## Configuration

The `config.json` file controls all pipeline parameters:

```json
{
  "paths": {
    "results_dir": "./processing/results",
    "zones_methods_dir": "./zones_methods",
    "detection_knowledge_base": "./processing/detection_kb.json"
  },
  
  "process_settings": {
    "output_folder_name": "1_reimagined",
    "num_variations": 5
  },
  
  "separation_settings": {
    "output_folder_name": "2_separated",
    "min_agreement_ratio": 0.3,
    "consensus_method": "model_driven",
    "vulnerable_methods": [...]
  },
  
  "detection_settings": {
    "output_folder_name": "3_detected",
    "config": {
      "min_defect_area_px": 5,
      "max_defect_area_px": 10000,
      "confidence_threshold": 0.3,
      ...
    }
  },
  
  "data_acquisition_settings": {
    "clustering_eps": 30.0,
    "min_cluster_size": 1,
    ...
  }
}
```

### Key Configuration Parameters
- **min_defect_area_px:** Minimum defect size in pixels (filters noise)
- **confidence_threshold:** Minimum confidence to report a defect (0-1)
- **clustering_eps:** Distance threshold for merging nearby defects
- **min_agreement_ratio:** Minimum consensus required among segmentation methods

## Pipeline Architecture

The pipeline consists of 4 sequential stages:

```
Input Image → [Stage 1: Processing] → Multiple Image Variations
                                           ↓
                                    [Stage 2: Separation] → Segmented Regions
                                           ↓
                                    [Stage 3: Detection] → Defect Reports
                                           ↓
                                    [Stage 4: Data Acquisition] → Final Analysis
```

## Detailed Usage Instructions

### Method 1: Interactive Mode (Recommended for Beginners)

1. Open a terminal/command prompt in the project directory

2. Run the main application:
```bash
python app.py
```

3. When prompted for config path, press Enter to use default or provide custom path:
```
Enter path to config.json (or press Enter for default 'config.json'):
```

4. Select processing mode:
```
--- MAIN MENU ---
1. Process a list of specific images
2. Process all images in a folder
3. Exit
Please select an option (1-3):
```

5. **For Option 1 (Specific Images):**
   - Enter full paths separated by spaces
   - Use quotes for paths with spaces
```
Enter one or more full image paths. Separate paths with spaces.
Example: C:\Users\Test\img1.png "C:\My Images\test.png"
> C:\FiberImages\sample1.png C:\FiberImages\sample2.png
```

6. **For Option 2 (Folder Processing):**
```
Enter the full path to the folder containing images: C:\FiberImages
```

### Method 2: Direct Script Execution

For processing a single image programmatically:

```python
from pathlib import Path
from app import PipelineOrchestrator

# Initialize pipeline
orchestrator = PipelineOrchestrator("config.json")

# Process single image
image_path = Path("C:/FiberImages/sample.png")
final_report = orchestrator.run_full_pipeline(image_path)

# Check results
if final_report:
    summary = final_report['analysis_summary']
    print(f"Status: {summary['pass_fail_status']}")
    print(f"Quality Score: {summary['quality_score']}/100")
    print(f"Total Defects: {summary['total_merged_defects']}")
```

## Component Details

### Stage 1: Image Processing (process.py)

Generates multiple variations of the input image using various transformations:

- **Thresholding:** Binary, Otsu, adaptive
- **Color transformations:** HSV, LAB, various colormaps
- **Preprocessing:** Blur, denoise, morphological operations
- **Edge detection:** Canny, Sobel, Laplacian
- **Intensity adjustments:** Brightness, contrast modifications

**Output:** `processing/results/<image_name>/1_reimagined/`

### Stage 2: Region Separation (separation.py)

Uses 11 different segmentation methods with consensus voting:

#### Vulnerable Methods (use pre-processed images):
- adaptive_intensity
- gradient_approach
- guess_approach
- threshold_separation
- intelligent_segmenter

#### Robust Methods (use original images):
- bright_core_extractor
- computational_separation
- geometric_approach
- hough_separation
- unified_core_cladding_detector

#### Consensus Algorithm:
1. Preliminary pixel voting
2. High-agreement method identification
3. Parameter-space averaging
4. Final mask generation

**Output:** `processing/results/<image_name>/2_separated/`

### Stage 3: Defect Detection (detection.py)

Performs comprehensive anomaly detection:

- **Feature Extraction:** 100+ statistical, morphological, and textural features
- **Reference Model:** Learns from clean fiber images
- **Anomaly Detection:**
  - Global statistical analysis (Mahalanobis distance)
  - Local anomaly mapping
  - Specific defect detection (scratches, digs, contamination)
- **Confidence Scoring:** Multi-criteria evaluation

**Output:** `processing/results/<image_name>/3_detected/`

### Stage 4: Data Acquisition (data_acquisition.py)

Aggregates and analyzes all detection results:

- **Defect Clustering:** DBSCAN algorithm to merge nearby detections
- **Intelligent Merging:** Considers defect type and confidence
- **Quality Assessment:**
  - Overall quality score (0-100)
  - Pass/fail determination
  - Severity-based deductions
- **Comprehensive Visualization:** Multi-panel analysis summary

**Output:** `processing/results/<image_name>/4_final_analysis/`

## Output Structure

After processing, each image generates the following structure:

```
processing/results/<image_name>/
│
├── 1_reimagined/               # Image variations
│   ├── threshold_binary.jpg
│   ├── preprocessing_blur.jpg
│   └── ... (30+ variations)
│
├── 2_separated/                # Segmented regions
│   ├── <variation_name>/
│   │   ├── region_core.png
│   │   ├── region_cladding.png
│   │   ├── region_ferrule.png
│   │   ├── mask_*.png
│   │   └── consensus_report.json
│   └── ...
│
├── 3_detected/                 # Defect detection results
│   ├── <region_name>/
│   │   ├── *_report.json
│   │   ├── *_analysis.png
│   │   └── *_detailed.txt
│   └── ...
│
├── 4_final_analysis/          # Aggregated results
│   ├── <image_name>_comprehensive_analysis.png
│   ├── <image_name>_final_report.json
│   ├── <image_name>_summary.txt
│   └── integrity_log.json
│
└── FINAL_SUMMARY.txt          # Quick overview
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "No module named 'cv2'" Error
```bash
pip install opencv-python
```

#### 2. "matplotlib not available" Warning
- Some visualizations will be skipped
- Install with: `pip install matplotlib`

#### 3. Memory Issues with Large Images
- Reduce image size before processing
- Process images individually instead of batch
- Increase system swap space

#### 4. No Methods Found in zones_methods
- Ensure all method scripts are in the correct directory
- Check file permissions
- Verify Python path includes the zones_methods directory

#### 5. Detection Stage Fails
- Check if detection knowledge base exists
- May need to build reference model first
- Verify image format compatibility (PNG, JPG, BMP)

### Debug Mode

Enable detailed logging by modifying `app.py`:

```python
logging.basicConfig(level=logging.DEBUG)  # Change from INFO to DEBUG
```

## Advanced Usage

### Customizing Detection Sensitivity

Modify `config.json`:

```json
"detection_settings": {
  "config": {
    "confidence_threshold": 0.2,  // Lower = more sensitive
    "min_defect_area_px": 3,      // Smaller = detect tiny defects
    "anomaly_threshold_multiplier": 2.0  // Lower = more anomalies
  }
}
```

### Building Custom Reference Models

```python
from detection import OmniFiberAnalyzer, OmniConfig

config = OmniConfig(knowledge_base_path="custom_kb.json")
analyzer = OmniFiberAnalyzer(config)

# Build from directory of clean fiber images
analyzer.build_comprehensive_reference_model("path/to/clean/images")
```

### Batch Processing with Custom Parameters

```python
import json
from pathlib import Path
from app import PipelineOrchestrator

# Modify config
with open("config.json", "r") as f:
    config = json.load(f)

config["detection_settings"]["config"]["confidence_threshold"] = 0.1
config["data_acquisition_settings"]["clustering_eps"] = 50.0

# Save temporary config
with open("temp_config.json", "w") as f:
    json.dump(config, f)

# Process with custom config
orchestrator = PipelineOrchestrator("temp_config.json")
for image_file in Path("images").glob("*.png"):
    orchestrator.run_full_pipeline(image_file)
```

### Accessing Individual Stages

```python
# Run only detection on a pre-segmented image
from detection import OmniFiberAnalyzer, OmniConfig

config = OmniConfig()
analyzer = OmniFiberAnalyzer(config)
analyzer.analyze_end_face("segmented_core.png", "output_dir")

# Run only data acquisition on existing results
from data_acquisition import integrate_with_pipeline

report = integrate_with_pipeline(
    results_base_dir="processing/results/sample_image",
    image_name="sample_image",
    clustering_eps=30.0
)
```

### Performance Optimization Tips

#### For faster processing:
- Reduce number of image variations in process.py
- Limit segmentation methods to most reliable ones
- Increase confidence thresholds to reduce false positives

#### For higher accuracy:
- Build reference model with multiple clean samples
- Use lower confidence thresholds
- Enable all segmentation methods

#### For batch processing:
- Process images in parallel using multiprocessing
- Pre-resize large images
- Use SSD storage for temporary files

## Support and Contributions

For issues, questions, or contributions:
- Check the troubleshooting section
- Review the debug logs
- Ensure all dependencies are correctly installed
- Verify input image quality and format