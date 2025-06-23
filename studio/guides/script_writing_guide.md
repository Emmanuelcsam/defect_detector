# Script Writing Guide for Image Processing GUI

## 🎯 Core Requirements

Every script must have a `process_image` function with this signature:

```python
def process_image(image: np.ndarray, **optional_params) -> np.ndarray:
    """Process and return the image"""
    # Your code here
    return processed_image
```

## 📝 Basic Script Template

```python
"""Brief description of what this script does"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Detailed description of the processing.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Processed image as numpy array
    """
    # Create a copy to avoid modifying the original
    result = image.copy()
    
    # Your processing code here
    
    return result
```

## 🎨 Script Templates by Category

### 1. **Simple Filter (No Parameters)**

```python
"""Apply sharpening filter to enhance image details"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Apply sharpening filter using a custom kernel.
    
    Args:
        image: Input image
        
    Returns:
        Sharpened image
    """
    # Define sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Apply the kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened
```

### 2. **Filter with Parameters**

```python
"""Apply box blur with adjustable kernel size"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply box blur filter.
    
    Args:
        image: Input image
        kernel_size: Size of the blur kernel (must be positive)
        
    Returns:
        Blurred image
    """
    # Ensure kernel size is positive and odd
    kernel_size = max(1, abs(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply box filter
    blurred = cv2.boxFilter(image, -1, (kernel_size, kernel_size))
    
    return blurred
```

### 3. **Grayscale/Color Handling**

```python
"""Detect edges with proper grayscale handling"""
import cv2
import numpy as np

def process_image(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """
    Detect edges using Canny algorithm.
    
    Args:
        image: Input image (color or grayscale)
        low: Lower threshold
        high: Upper threshold
        
    Returns:
        Edge map (grayscale)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Detect edges
    edges = cv2.Canny(blurred, low, high)
    
    return edges
```

### 4. **Color Output from Grayscale Input**

```python
"""Apply colormap visualization"""
import cv2
import numpy as np

def process_image(image: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """
    Apply a colormap to visualize grayscale data.
    
    Args:
        image: Input image
        colormap: Colormap name (jet, hot, cool, viridis)
        
    Returns:
        Colorized image (BGR)
    """
    # Convert to grayscale first if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Map colormap names to OpenCV constants
    colormap_dict = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "parula": cv2.COLORMAP_PARULA
    }
    
    # Get the colormap constant
    cmap = colormap_dict.get(colormap.lower(), cv2.COLORMAP_JET)
    
    # Apply colormap
    colored = cv2.applyColorMap(gray, cmap)
    
    return colored
```

### 5. **Complex Processing with Multiple Steps**

```python
"""Detect and highlight defects in fiber optic images"""
import cv2
import numpy as np

def process_image(image: np.ndarray, 
                  sensitivity: float = 1.0,
                  min_defect_size: int = 10,
                  highlight_color: str = "red") -> np.ndarray:
    """
    Detect and highlight potential defects.
    
    Args:
        image: Input image
        sensitivity: Detection sensitivity (0.5-2.0)
        min_defect_size: Minimum defect area in pixels
        highlight_color: Color for highlighting (red, green, blue)
        
    Returns:
        Image with highlighted defects
    """
    # Ensure we have a color output
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image
    else:
        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0 * sensitivity, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Threshold to find dark regions (potential defects)
    mean_val = np.mean(enhanced)
    thresh_val = mean_val * (0.7 / sensitivity)
    _, thresh = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Color mapping
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "magenta": (255, 0, 255),
        "cyan": (255, 255, 0)
    }
    color = color_map.get(highlight_color.lower(), (0, 0, 255))
    
    # Draw contours for defects larger than minimum size
    defect_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_defect_size:
            cv2.drawContours(result, [contour], -1, color, 2)
            # Add bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 1)
            defect_count += 1
    
    # Add text overlay
    cv2.putText(result, f"Defects: {defect_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return result
```

### 6. **Working with ROIs (Regions of Interest)**

```python
"""Apply processing only to circular center region"""
import cv2
import numpy as np

def process_image(image: np.ndarray, 
                  radius_percent: float = 50.0,
                  blur_strength: int = 15) -> np.ndarray:
    """
    Apply blur only to the center circular region.
    
    Args:
        image: Input image
        radius_percent: Radius as percentage of image size (0-100)
        blur_strength: Blur kernel size
        
    Returns:
        Image with blurred center
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    # Calculate center and radius
    center = (w // 2, h // 2)
    radius = int(min(h, w) * radius_percent / 200)
    
    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply blur to entire image
    blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    # Combine using mask
    if len(image.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) // 255
        result = result * (1 - mask) + blurred * mask
        result = result.astype(np.uint8)
    else:
        result = cv2.bitwise_and(blurred, blurred, mask=mask)
        result += cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    
    return result
```

## 🛡️ Best Practices

### 1. **Always Handle Both Color and Grayscale**
```python
# Check image dimensions
if len(image.shape) == 2:
    # Grayscale image
    channels = 1
elif len(image.shape) == 3:
    # Color image
    channels = image.shape[2]
```

### 2. **Parameter Validation**
```python
def process_image(image: np.ndarray, param: int = 5) -> np.ndarray:
    # Validate parameters
    param = max(1, min(param, 100))  # Clamp to valid range
    
    if param % 2 == 0:  # Ensure odd for kernel sizes
        param += 1
```

### 3. **Type Hints for Automatic UI**
```python
# The GUI will create appropriate widgets based on type hints
def process_image(image: np.ndarray,
                  threshold: int = 127,        # Creates a SpinBox
                  sigma: float = 1.0,          # Creates a DoubleSpinBox  
                  enable_blur: bool = True,    # Creates a CheckBox
                  mode: str = "normal") -> np.ndarray:  # Creates a LineEdit
```

### 4. **Error Handling**
```python
def process_image(image: np.ndarray, divisions: int = 4) -> np.ndarray:
    try:
        if divisions == 0:
            raise ValueError("Divisions cannot be zero")
            
        # Your processing code
        result = image // divisions
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        # Return original image on error
        return image
```

### 5. **Memory Efficiency**
```python
def process_image(image: np.ndarray) -> np.ndarray:
    # DON'T modify the input directly
    # image[:] = 255  # Bad!
    
    # DO create a copy first
    result = image.copy()
    result[:] = 255  # Good!
    
    return result
```

## 🧪 Testing Your Script

### 1. **Standalone Test**
```python
# Add this at the bottom of your script for testing
if __name__ == "__main__":
    # Test the function standalone
    test_image = cv2.imread("test.jpg")
    if test_image is not None:
        result = process_image(test_image, param1=10)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not load test image")
```

### 2. **Quick Test Script**
```python
# test_my_function.py
import cv2
import numpy as np
from my_new_script import process_image

# Create test images
test_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
test_color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Test with both
result_gray = process_image(test_gray)
result_color = process_image(test_color)

print(f"Gray input: {test_gray.shape} -> {result_gray.shape}")
print(f"Color input: {test_color.shape} -> {result_color.shape}")
```

## 📋 Checklist for New Scripts

- [ ] Has `process_image` function
- [ ] First parameter is `image: np.ndarray`
- [ ] Returns `np.ndarray`
- [ ] Has docstring describing what it does
- [ ] Handles both grayscale and color images
- [ ] Parameters have type hints
- [ ] Parameters have default values
- [ ] No hardcoded file paths
- [ ] No `cv2.imshow()` or `cv2.waitKey()` in main code
- [ ] Works on a copy of the image, not the original
- [ ] Handles errors gracefully

## 🚀 Advanced Tips

### 1. **Chain-Friendly Operations**
Make sure your output can be input to other functions:
```python
# Good: Maintains image format
def process_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        # Process and keep as grayscale
        return processed_gray
    else:
        # Process and keep as color
        return processed_color
```

### 2. **Debugging in the GUI**
```python
def process_image(image: np.ndarray, debug: bool = False) -> np.ndarray:
    if debug:
        print(f"Input shape: {image.shape}")
        print(f"Input type: {image.dtype}")
        print(f"Input range: [{image.min()}, {image.max()}]")
    
    # Processing...
    
    if debug:
        print(f"Output shape: {result.shape}")
    
    return result
```

### 3. **Performance Considerations**
```python
def process_image(image: np.ndarray, quality: str = "fast") -> np.ndarray:
    if quality == "fast":
        # Use faster but lower quality method
        return cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    else:
        # Use slower but higher quality method
        return cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
```

## 💾 Save Your Scripts

Place your new scripts in the `scripts/` directory. The GUI will automatically detect them when you:
1. Click "Refresh Functions" 
2. Restart the GUI

Script naming suggestions:
- `edge_detection_custom.py`
- `blur_advanced.py`
- `threshold_adaptive_custom.py`
- `fiber_defect_detector.py`

The filename will be displayed when the script executes, so use descriptive names!