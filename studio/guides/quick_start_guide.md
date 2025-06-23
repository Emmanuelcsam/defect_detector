# Image Processing Pipeline GUI - Quick Start Guide

## 🚀 Getting Started

### Step 1: Setup
```bash
# 1. Install dependencies
pip install opencv-python numpy PyQt5

# 2. Run the setup script to create directories and examples
python setup_gui.py

# 3. Fix your existing scripts with Unicode errors
python enhanced_script_cleaner.py --source "all_modules - Copy" --output scripts
```

### Step 2: Launch the GUI
```bash
python image_processor_gui.py
```

## 🎯 Key Features

### 1. **Automatic Script Cleaning**
- The GUI automatically handles scripts with Unicode errors
- Hardcoded paths are removed
- Scripts are wrapped in a standard `process_image()` function

### 2. **Visual Pipeline Builder**
- Drag and drop to reorder processing steps
- Double-click items to edit parameters
- Save/load pipeline configurations

### 3. **Real-time Script Display**
- See exactly which script file is executing
- Red highlighted label shows current script
- Progress bar shows pipeline progress

### 4. **Advanced Image Viewer**
- **Zoom**: Mouse wheel (centered on cursor)
- **Pan**: Middle-click and drag
- **Zoom controls**: +/- buttons, Fit, 100% reset

### 5. **Search and Filter**
- Search functions by keyword
- Filter by category (Edge Detection, Filtering, etc.)
- See function details before adding

## 📁 Directory Structure

```
your_project/
├── scripts/              # Your image processing functions
│   └── cleaned/         # Auto-cleaned versions
├── images/              # Test images
├── output/              # Saved results
├── pipelines/           # Saved pipeline configurations
├── image_processor_gui.py    # Main GUI application
├── enhanced_script_cleaner.py # Script cleaner utility
└── setup_gui.py         # Setup script
```

## 🔧 Making Your Scripts Compatible

### Option 1: Automatic (Recommended)
Place your scripts in a directory and run:
```bash
python enhanced_script_cleaner.py --source your_scripts_dir --output scripts
```

### Option 2: Manual Template
Create scripts following this template:

```python
"""Description of what your function does"""
import cv2
import numpy as np

def process_image(image: np.ndarray, param1: int = 10, param2: float = 1.0) -> np.ndarray:
    """
    Process the image.
    
    Args:
        image: Input image (numpy array)
        param1: First parameter description
        param2: Second parameter description
        
    Returns:
        Processed image (numpy array)
    """
    # Your processing code here
    result = image.copy()
    
    # Example: Apply Gaussian blur
    result = cv2.GaussianBlur(result, (param1, param1), param2)
    
    return result
```

## 🛠️ Troubleshooting

### Unicode Errors
- Use `enhanced_script_cleaner.py` to fix these automatically
- The GUI has built-in Unicode error handling

### Script Not Loading
1. Check console for error messages
2. Ensure script has `process_image` function
3. Try the script cleaner
4. Verify all imports are available

### Processing Fails
- Check the exact script name shown in the execution display
- Look at console output for detailed error messages
- Ensure input/output image formats match

## 💡 Pro Tips

1. **Pipeline Order Matters**: Grayscale conversion should usually come first
2. **Save Pipelines**: Save complex workflows for reuse
3. **Parameter Types**: Use type hints for automatic UI generation
4. **Test Individual Functions**: Add one function at a time to debug

## 📝 Example Workflow

1. Load an image: Click "Load Image"
2. Search for "blur" in the function library
3. Double-click "gaussian_blur.py" to add it
4. Double-click the pipeline item to adjust blur strength
5. Add more functions (edge detection, threshold, etc.)
6. Click "PROCESS IMAGE" to see results
7. Save the result or the pipeline for later use

## 🎨 Built-in Categories

- **Filtering**: Blur, noise reduction
- **Edge Detection**: Canny, Sobel, Laplacian
- **Thresholding**: Binary, Otsu, adaptive
- **Morphology**: Erosion, dilation, opening, closing
- **Enhancement**: Histogram equalization, CLAHE
- **Detection**: Circles, contours
- **Color**: Grayscale conversion, colormaps

## 🔥 Keyboard Shortcuts

- **Ctrl+O**: Load image
- **Ctrl+S**: Save result
- **Mouse Wheel**: Zoom in/out
- **Middle Click**: Pan
- **Double Click**: Edit parameters

Happy image processing! 🖼️✨