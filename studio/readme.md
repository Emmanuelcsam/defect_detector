# Image Processing Pipeline Studio - OpenCV Practice


### Image Processing Pipeline GUI 
A powerful visual interface that allows you to:
- **Build Custom Pipelines**: Drag-and-drop interface for creating processing workflows
- **Real-time Processing**: See which script is executing with live feedback
- **Parameter Editing**: Double-click to adjust function parameters with auto-generated UI
- **Advanced Viewer**: Zoom (mouse wheel) and pan (middle-click) functionality
- **Search & Filter**: Find functions by name or category
- **Save/Load Pipelines**: Store and reuse your processing workflows

## Prerequisites

### Installing UV (Python Environment Manager)
Choose one of the following installation methods:

- **Windows PowerShell:**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS/Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Quick Setup
```bash
# Navigate to the App directory
cd App

# Create environment and install dependencies
uv venv && uv pip install -r requirements.txt

# Run setup script
uv run python setup_gui.py

# Launch the GUI
uv run python image_processor_gui.py
```

## Project Structure

```
OpenCV-Practice/
├── app/                        # Main application directory (NEW)
│   ├── scripts/               # Image processing functions
│   │   └── cleaned/          # Auto-cleaned script versions
│   ├── images/               # Input images
│   ├── output/               # Processing results
│   ├── pipelines/            # Saved pipeline configurations
│   ├── image_processor_gui.py    # Main GUI application
│   ├── enhanced_script_cleaner.py # Script cleaning utility
│   ├── setup_gui.py              # Initial setup script
│   └── requirements.txt          # Python dependencies
│
├── guides/                    # Comprehensive documentation (NEW)
│   ├── script_writing_guide.md   # How to write compatible scripts
│   ├── quick_reference.md        # Quick reference card
│   ├── quick_start_guide.md      # Getting started guide
│   ├── example_custom_script.py  # Example processing script
│   ├── minimal_script_template.py # Basic script template
│   └── validate_script.py        # Script validation tool
│
├── all_modules_unchanged/     # Original unmodified scripts
├── Additional_programs/       # Supplementary tools
├── scratchdataset/           # Sample scratch detection images
├── results/                  # Analysis outputs
└── old_process/             # Legacy processing methods
```

## Key Features

### 1. **Automatic Script Cleaning**
- Handles Unicode errors automatically
- Removes hardcoded paths
- Creates standardized `process_image()` wrappers

### 2. **Visual Pipeline Builder**
- Drag-and-drop to reorder processing steps
- Real-time display of executing script names
- Save/load pipeline configurations as JSON

### 3. **Dynamic Function Loading**
- Automatically detects scripts in the `scripts/` directory
- Categorizes functions (Filtering, Edge Detection, etc.)
- Search by keyword functionality

### 4. **Advanced Parameter Control**
- Type-based UI generation:
  - `int` → SpinBox
  - `float` → DoubleSpinBox
  - `bool` → CheckBox
  - `str` → LineEdit

## Usage

### Running the GUI
```bash
cd App
uv run python image_processor_gui.py
```

### Cleaning Legacy Scripts
If you have scripts with Unicode errors or hardcoded paths:
```bash
cd App
uv run python enhanced_script_cleaner.py --source ../all_modules_unchanged --output scripts
```

### Writing New Scripts
Scripts must follow this structure:
```python
"""Description of what this script does"""
import cv2
import numpy as np

def process_image(image: np.ndarray, param1: int = 10) -> np.ndarray:
    """
    Process the image.
    
    Args:
        image: Input image
        param1: Parameter description
        
    Returns:
        Processed image
    """
    result = image.copy()
    # Your processing code here
    return result
```

See `guides/script_writing_guide.md` for detailed instructions.

### Validating Scripts
```bash
cd guides
uv run python validate_script.py ../App/scripts/your_script.py
```

## Documentation

### In the `guides/` Directory:
- **script_writing_guide.md**: Comprehensive guide for creating compatible scripts
- **quick_start_guide.md**: Step-by-step getting started instructions
- **quick_reference.md**: One-page summary of key concepts
- **example_custom_script.py**: Full-featured example with multiple parameters
- **minimal_script_template.py**: Basic template to start from

### PDF Resources:
- **Breakdowns.pdf**: OpenCV function breakdowns and tutorials
- **research.pdf**: Academic paper on fiber optic defect detection

## Workflow Example

1. **Launch the GUI**: `uv run python image_processor_gui.py`
2. **Load an image**: Click "Load Image" button
3. **Build a pipeline**:
   - Search for "blur" in the function library
   - Double-click to add to pipeline
   - Add more functions (edge detection, threshold, etc.)
4. **Process**: Click "PROCESS IMAGE"
5. **Save results**: Save the processed image or pipeline configuration

## Dependencies

Core requirements (automatically installed via setup):
- OpenCV (cv2)
- NumPy
- PyQt5
- Pillow

## Notes

- The GUI displays the exact filename of each executing script for easy debugging
- Scripts can still run standalone outside the GUI
- All processing is done on image copies to preserve originals
- The system supports both grayscale and color images

## Contributing

When adding new processing functions:
1. Follow the script template in `guides/minimal_script_template.py`
2. Validate using `guides/validate_script.py`
3. Place in `App/scripts/` directory
4. Use descriptive filenames (they appear in the GUI)

## Support

For issues or questions:
1. Check the guides in the `guides/` directory
2. Validate your scripts with the validation tool
3. Ensure all dependencies are installed via `requirements.txt`
