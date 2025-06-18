# 🎯 Image Processing GUI - Script Quick Reference

## Minimum Required Structure
```python
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """What this does"""
    result = image.copy()
    # Your code here
    return result
```

## Parameter Types → UI Elements
| Type Hint | GUI Widget | Example |
|-----------|------------|---------|
| `int` | SpinBox | `kernel_size: int = 5` |
| `float` | DoubleSpinBox | `sigma: float = 1.0` |
| `bool` | CheckBox | `enable: bool = True` |
| `str` | LineEdit | `mode: str = "fast"` |

## Common Patterns

### Handle Color/Grayscale
```python
if len(image.shape) == 3:
    # Color image (BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    # Already grayscale
    gray = image
```

### Ensure Output Format
```python
# If your operation produces grayscale but input was color:
if len(image.shape) == 3 and len(result.shape) == 2:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
```

### Validate Parameters
```python
# Clamp to range
value = max(min_val, min(value, max_val))

# Ensure odd kernel size
if kernel_size % 2 == 0:
    kernel_size += 1
```

## DO's ✅
- Always work on a copy: `result = image.copy()`
- Add type hints for all parameters
- Provide default values for parameters
- Include a docstring
- Handle errors gracefully
- Return the same type (np.ndarray)

## DON'Ts ❌
- Don't use `cv2.imshow()` or `plt.show()`
- Don't include hardcoded file paths
- Don't use `input()` or `print()` excessively
- Don't modify the input image directly
- Don't use `cv2.waitKey()` or `cv2.destroyAllWindows()`

## Testing Your Script

### 1. Validate It
```bash
python validate_script.py your_script.py
```

### 2. Quick Test
```python
# At the bottom of your script:
if __name__ == "__main__":
    test = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = process_image(test)
    print(f"Success! {test.shape} -> {result.shape}")
```

### 3. Load in GUI
- Place in `scripts/` folder
- Click "Refresh Functions" or restart GUI

## Example: Complete Script
```python
"""Apply bilateral filter for edge-preserving smoothing"""
import cv2
import numpy as np

def process_image(image: np.ndarray, 
                  d: int = 9,
                  sigma_color: float = 75.0,
                  sigma_space: float = 75.0) -> np.ndarray:
    """
    Apply bilateral filter to preserve edges while smoothing.
    
    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
```

## File Naming
Use descriptive names - they appear in the GUI:
- ✅ `adaptive_threshold_custom.py`
- ✅ `fiber_core_detector.py`
- ❌ `script1.py`
- ❌ `test.py`