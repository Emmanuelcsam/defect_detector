#!/usr/bin/env python3
"""
Setup Script for Image Processing GUI
Creates directory structure and example functions
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'scripts',
        'scripts/cleaned',
        'images',
        'output',
        'pipelines'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_example_functions():
    """Create example processing functions to get started"""
    
    example_functions = {
        'gaussian_blur.py': '''"""Apply Gaussian blur to smooth the image"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to the image.
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
''',

        'edge_detection.py': '''"""Detect edges using Canny edge detection"""
import cv2
import numpy as np

def process_image(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges using Canny edge detection.
    
    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
    
    Returns:
        Edge map
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur first to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Canny edge detection
    return cv2.Canny(blurred, low_threshold, high_threshold)
''',

        'threshold_binary.py': '''"""Apply binary threshold to the image"""
import cv2
import numpy as np

def process_image(image: np.ndarray, threshold: int = 127, max_value: int = 255) -> np.ndarray:
    """
    Apply binary threshold to the image.
    
    Args:
        image: Input image
        threshold: Threshold value
        max_value: Maximum value to use with THRESH_BINARY
    
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
    return binary
''',

        'grayscale_convert.py': '''"""Convert image to grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale using OpenCV's color conversion.
    
    Args:
        image: Input image
    
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
''',

        'histogram_equalization.py': '''"""Apply histogram equalization for contrast enhancement"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance contrast.
    
    Args:
        image: Input image
    
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        # For color images, convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        # For grayscale images
        return cv2.equalizeHist(image)
''',

        'median_filter.py': '''"""Apply median filter for noise reduction"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter to reduce noise while preserving edges.
    
    Args:
        image: Input image
        kernel_size: Size of the median filter kernel
    
    Returns:
        Filtered image
    """
    return cv2.medianBlur(image, kernel_size)
''',

        'morphology_close.py': '''"""Apply morphological closing to fill gaps"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological closing to fill small gaps.
    
    Args:
        image: Input image
        kernel_size: Size of the structuring element
    
    Returns:
        Processed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
''',

        'circle_detection.py': '''"""Detect and highlight circles in the image"""
import cv2
import numpy as np

def process_image(image: np.ndarray, min_radius: int = 10, max_radius: int = 0) -> np.ndarray:
    """
    Detect circles using Hough Circle Transform.
    
    Args:
        image: Input image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius (0 for no limit)
    
    Returns:
        Image with detected circles highlighted
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = image.copy()
    else:
        gray = image
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
    )
    
    # Draw circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw circle outline
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw center point
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    return output
''',

        'sobel_edges.py': '''"""Apply Sobel edge detection"""
import cv2
import numpy as np

def process_image(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply Sobel edge detection to find gradients.
    
    Args:
        image: Input image
        ksize: Size of the Sobel kernel
    
    Returns:
        Edge magnitude image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Calculate magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude
''',

        'clahe_enhancement.py': '''"""Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
import cv2
import numpy as np

def process_image(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE for local contrast enhancement.
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        # Convert to LAB color space and apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # For grayscale images
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)
'''
    }
    
    scripts_dir = Path('scripts')
    for filename, content in example_functions.items():
        filepath = scripts_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created example function: {filename}")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """# Requirements for Image Processing GUI
opencv-python>=4.5.0
numpy>=1.19.0
PyQt5>=5.15.0
Pillow>=8.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✓ Created requirements.txt")

def create_readme():
    """Create README file with instructions"""
    readme = """# Image Processing Pipeline GUI

A powerful and flexible GUI for building custom image processing pipelines.

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the setup script (already done):
   ```bash
   python setup_gui.py
   ```

3. Start the GUI:
   ```bash
   python image_processor_gui.py
   ```

## Adding Your Scripts

### Method 1: Direct Copy (if scripts are already compatible)
1. Copy your .py files to the `scripts` directory
2. Make sure each script has a `process_image(image)` function
3. The GUI will automatically detect and load them

### Method 2: Using Script Cleaner (for scripts with hardcoded paths)
1. Place your original scripts in a separate directory
2. Run the script cleaner:
   ```bash
   python script_cleaner.py --source your_scripts_dir --output scripts
   ```

### Method 3: Manual Conversion
Create a wrapper for your existing functions:

```python
'''Description of what your script does'''
import cv2
import numpy as np

def process_image(image: np.ndarray, param1: int = 10) -> np.ndarray:
    '''
    Process the image.
    
    Args:
        image: Input image
        param1: Description of parameter
        
    Returns:
        Processed image
    '''
    # Your processing code here
    result = your_existing_function(image, param1)
    return result
```

## Directory Structure

- `scripts/` - Place your image processing functions here
- `scripts/cleaned/` - Automatically cleaned versions of scripts
- `images/` - Store your test images here
- `output/` - Save processed images here
- `pipelines/` - Save/load pipeline configurations here

## Features

- **Dynamic Function Loading**: Automatically detects all scripts in the scripts directory
- **Visual Pipeline Builder**: Drag and drop to reorder processing steps
- **Parameter Editing**: Double-click pipeline items to edit parameters
- **Real-time Feedback**: See which script is currently executing
- **Zoom & Pan**: Mouse wheel to zoom, middle-click to pan
- **Search & Filter**: Find functions by name or category
- **Save/Load Pipelines**: Save your processing workflows for reuse

## Tips

1. Scripts with Unicode errors will be automatically cleaned
2. The GUI shows the exact filename being executed for debugging
3. Use Ctrl+Mouse Wheel for zooming
4. Middle-click and drag to pan around the image
5. Pipeline configurations are saved as JSON files

## Troubleshooting

If a script fails to load:
1. Check the console for error messages
2. Ensure the script has a `process_image` function
3. Try running the script cleaner on it
4. Check that all imports are available

Happy image processing!
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    print("✓ Created README.md")

def check_dependencies():
    """Check if required packages are installed"""
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PyQt5': 'PyQt5'
    }
    
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("\n⚠️  Missing dependencies:")
        for package in missing:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("✓ All dependencies installed")
    return True

def main():
    print("="*60)
    print("Image Processing GUI - Setup Script")
    print("="*60)
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    # Create example functions
    print("\n2. Creating example functions...")
    create_example_functions()
    
    # Create requirements file
    print("\n3. Creating requirements file...")
    create_requirements_file()
    
    # Create README
    print("\n4. Creating README...")
    create_readme()
    
    # Check dependencies
    print("\n5. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    
    print("\nNext steps:")
    if not deps_ok:
        print("1. Install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("\n2. Add your scripts to the 'scripts' directory")
        print("   - Each script needs a process_image(image) function")
        print("   - Scripts with Unicode errors will be auto-cleaned")
        print("\n3. Run the GUI:")
        print("   python image_processor_gui.py")
    else:
        print("1. Add your scripts to the 'scripts' directory")
        print("   - Each script needs a process_image(image) function")
        print("   - Scripts with Unicode errors will be auto-cleaned")
        print("\n2. Run the GUI:")
        print("   python image_processor_gui.py")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
