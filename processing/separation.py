import os  # Provides functions for interacting with the operating system (file paths, environment variables)
import sys  # Provides access to system-specific parameters and functions (command line args, python executable path)
import json  # Enables encoding/decoding of JSON data for saving/loading results and configuration
import time  # Provides time-related functions for measuring execution duration and timestamps
import numpy as np  # Numerical computing library for efficient array operations and mathematical computations
import cv2  # OpenCV library for computer vision operations (image loading, processing, morphological operations)
from pathlib import Path  # Object-oriented filesystem path handling for cross-platform compatibility
from datetime import datetime  # Date and time handling for timestamps (though not actively used in this script)
import subprocess  # Enables running external Python scripts in isolated processes for method execution
import tempfile  # Creates temporary directories and files for intermediate processing results
from typing import Dict, List, Tuple, Optional, Any  # Type hints for better code documentation and IDE support
import shutil  # High-level file operations like copying and removing directories (though not actively used here)
import warnings  # Controls warning messages from libraries
warnings.filterwarnings('ignore')  # Suppresses all warning messages to keep console output clean

# Import matplotlib for visualizations
try:  # Attempt to import matplotlib for creating summary visualizations
    import matplotlib  # Main matplotlib package for plotting
    matplotlib.use('Agg')  # Sets backend to non-interactive mode for server/headless environments
    import matplotlib.pyplot as plt  # Pyplot interface for creating figures and plots
    HAS_MATPLOTLIB = True  # Flag indicating matplotlib is available for visualization features
except ImportError:  # Handle case where matplotlib is not installed
    HAS_MATPLOTLIB = False  # Disable visualization features that require matplotlib
    print("Warning: matplotlib not available, some visualizations will be skipped")  # Inform user about limited functionality

# Import scipy components for enhanced logic
try:  # Attempt to import advanced image processing functions from scipy
    from scipy.ndimage import median_filter, gaussian_filter  # Image filtering functions for noise reduction
    from scipy.ndimage import binary_opening, binary_closing  # Morphological operations for cleaning binary masks
    HAS_SCIPY_FULL = True  # Flag indicating full scipy functionality is available
except ImportError:  # Handle case where scipy is not installed or partially installed
    HAS_SCIPY_FULL = False  # Disable advanced post-processing features
    print("Warning: Some scipy components not available, using basic post-processing")  # Inform user about reduced functionality

class NumpyEncoder(json.JSONEncoder):  # Custom JSON encoder to handle NumPy data types
    """Custom encoder for numpy data types for JSON serialization."""  # Docstring explaining class purpose
    def default(self, obj):  # Override default method to handle NumPy types
        if isinstance(obj, (np.integer, np.int_)):  # Check if object is NumPy integer type
            return int(obj)  # Convert NumPy integer to Python int for JSON compatibility
        if isinstance(obj, (np.floating, np.float_)):  # Check if object is NumPy float type
            return float(obj)  # Convert NumPy float to Python float for JSON compatibility
        if isinstance(obj, np.ndarray):  # Check if object is NumPy array
            return obj.tolist()  # Convert NumPy array to Python list for JSON compatibility
        return super(NumpyEncoder, self).default(obj)  # Fall back to default encoder for other types

class EnhancedConsensusSystem:  # Core consensus algorithm for combining multiple segmentation results
    """
    model aware voting system
    """  # Docstring explaining the advanced consensus approach
    def __init__(self, min_agreement_ratio=0.3):  # Initialize consensus system with configurable threshold
        self.min_agreement_ratio = min_agreement_ratio  # Minimum ratio of methods that must agree for consensus (0.3 = 30%)

    def _calculate_iou(self, mask1, mask2):  # Private method to compute Intersection over Union metric
        """Calculates Intersection over Union for two binary masks."""  # IoU measures overlap between two masks
        if mask1 is None or mask2 is None:  # Check if either mask is missing
            return 0.0  # Return zero overlap if either mask doesn't exist
        intersection = np.logical_and(mask1, mask2)  # Compute pixel-wise AND to find overlapping regions
        union = np.logical_or(mask1, mask2)  # Compute pixel-wise OR to find total coverage
        iou_score = np.sum(intersection) / (np.sum(union) + 1e-6)  # Ratio of intersection to union area (epsilon prevents division by zero)
        return iou_score  # Return IoU score between 0 (no overlap) and 1 (perfect overlap)

    def generate_consensus(self,
                           results: List['SegmentationResult'],  # List of segmentation results from different methods
                           method_scores: Dict[str, float],  # Historical performance scores for each method
                           image_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:  # Image dimensions (height, width)
        """
        Generates a final consensus model by first running a preliminary pixel
        vote to identify high-agreement methods, then calculating a weighted
        average of their geometric parameters to create an ideal final mask.
        """  # Multi-stage consensus approach: vote → filter → average → generate
        valid_results = [r for r in results if r.error is None and r.masks is not None]  # Filter out failed segmentation attempts
        
        if len(valid_results) < 2:  # Check if we have enough results for meaningful consensus
            print("! Not enough valid results to form a consensus.")  # Need at least 2 methods for comparison
            return None  # Exit early if consensus is impossible

        print(f"\nGenerating consensus from {len(valid_results)} valid results...")  # Progress indicator
        h, w = image_shape  # Extract height and width for mask creation

        # --- Stage 1: Preliminary Weighted Pixel Vote ---
        weighted_votes = np.zeros((h, w, 3), dtype=np.float32)  # 3-channel array for core/cladding/ferrule votes
        for r in valid_results:  # Iterate through each valid segmentation result
            weight = method_scores.get(r.method_name, 1.0) * r.confidence  # Combine historical score with result confidence
            if r.masks.get('core') is not None:  # Check if core mask exists
                weighted_votes[:, :, 0] += (r.masks['core'] > 0).astype(np.float32) * weight  # Add weighted vote for core pixels
            if r.masks.get('cladding') is not None:  # Check if cladding mask exists
                weighted_votes[:, :, 1] += (r.masks['cladding'] > 0).astype(np.float32) * weight  # Add weighted vote for cladding pixels
            if r.masks.get('ferrule') is not None:  # Check if ferrule mask exists
                weighted_votes[:, :, 2] += (r.masks['ferrule'] > 0).astype(np.float32) * weight  # Add weighted vote for ferrule pixels

        preliminary_classification = np.argmax(weighted_votes, axis=2)  # Find winning class for each pixel (0=core, 1=cladding, 2=ferrule)
        prelim_core_mask = (preliminary_classification == 0)  # Create binary mask for preliminary core region
        prelim_cladding_mask = (preliminary_classification == 1)  # Create binary mask for preliminary cladding region

        # --- Stage 2: Identify High-Agreement Methods ---
        high_agreement_results = []  # List to store methods that agree with preliminary consensus
        for r in valid_results:  # Check each result against preliminary consensus
            core_iou = self._calculate_iou(r.masks['core'], prelim_core_mask)  # Measure core mask agreement
            cladding_iou = self._calculate_iou(r.masks['cladding'], prelim_cladding_mask)  # Measure cladding mask agreement
            if core_iou > 0.6 and cladding_iou > 0.6:  # Require 60% overlap for both regions
                high_agreement_results.append(r)  # Add to high-agreement list for parameter averaging

        if not high_agreement_results:  # Check if any methods passed the agreement threshold
            print("! No methods passed the high-agreement threshold. Using all valid results for fallback.")  # Warning message
            high_agreement_results = valid_results  # Fall back to using all valid results
        
        print(f"  Found {len(high_agreement_results)} methods for parameter averaging.")  # Progress update

        # --- Stage 3: Parameter-Space Consensus ---
        consensus_params = {'cx': [], 'cy': [], 'core_r': [], 'clad_r': []}  # Lists to collect geometric parameters
        weights = []  # List to store weights for weighted averaging
        for r in high_agreement_results:  # Extract parameters from high-agreement methods
            weight = method_scores.get(r.method_name, 1.0) * r.confidence  # Calculate weight for this result
            if r.center and r.core_radius is not None and r.cladding_radius is not None:  # Verify all parameters exist
                consensus_params['cx'].append(r.center[0])  # Collect center x-coordinate
                consensus_params['cy'].append(r.center[1])  # Collect center y-coordinate
                consensus_params['core_r'].append(r.core_radius)  # Collect core radius
                consensus_params['clad_r'].append(r.cladding_radius)  # Collect cladding radius
                weights.append(weight)  # Store weight for this set of parameters

        if not weights:  # Check if we collected any valid parameters
             print("! No valid parameters to average. Consensus failed.")  # Error message
             return None  # Exit if no parameters available

        final_center = (  # Calculate weighted average center position
            np.average(consensus_params['cx'], weights=weights),  # Weighted average of x-coordinates
            np.average(consensus_params['cy'], weights=weights)  # Weighted average of y-coordinates
        )
        final_core_radius = np.average(consensus_params['core_r'], weights=weights)  # Weighted average core radius
        final_cladding_radius = np.average(consensus_params['clad_r'], weights=weights)  # Weighted average cladding radius

        # --- Stage 4: Generate Final Ideal Masks ---
        final_masks = self.create_masks_from_params(  # Convert averaged parameters to binary masks
            final_center, final_core_radius, final_cladding_radius, image_shape  # Pass geometric parameters and image size
        )
        
        final_masks['core'], final_masks['cladding'], final_masks['ferrule'] = self.ensure_mask_consistency(  # Clean up masks
             final_masks['core'], final_masks['cladding'], final_masks['ferrule']  # Ensure masks are mutually exclusive
        )
        
        return {  # Return comprehensive consensus result
            'masks': final_masks,  # Final segmentation masks
            'center': final_center,  # Consensus center coordinates
            'core_radius': final_core_radius,  # Consensus core radius
            'cladding_radius': final_cladding_radius,  # Consensus cladding radius
            'contributing_methods': [r.method_name for r in high_agreement_results],  # List of methods that contributed
            'num_valid_results': len(valid_results),  # Total number of successful segmentations
            'all_results': [r.to_dict() for r in results]  # Complete results for analysis
        }

    def create_masks_from_params(self, center: Tuple[float, float], core_radius: float, 
                               cladding_radius: float, image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:  # Convert geometric parameters to masks
        """Creates binary masks from geometric parameters."""  # Generate pixel-wise masks from circle parameters
        h, w = image_shape  # Extract image dimensions
        cx, cy = center  # Extract center coordinates
        y_grid, x_grid = np.ogrid[:h, :w]  # Create coordinate grids for distance calculation
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)  # Euclidean distance from each pixel to center
        
        core_mask = (dist_from_center <= core_radius).astype(np.uint8)  # Pixels within core radius
        cladding_mask = ((dist_from_center > core_radius) &   # Pixels outside core but inside cladding
                        (dist_from_center <= cladding_radius)).astype(np.uint8)  # Boolean AND for ring shape
        ferrule_mask = (dist_from_center > cladding_radius).astype(np.uint8)  # Pixels outside cladding radius
        
        return {'core': core_mask, 'cladding': cladding_mask, 'ferrule': ferrule_mask}  # Return dictionary of masks

    def ensure_mask_consistency(self, core_mask, cladding_mask, ferrule_mask):  # Clean and ensure mask exclusivity
        """Ensure masks are mutually exclusive and spatially clean."""  # Remove overlaps and noise from masks
        if not HAS_SCIPY_FULL:  # Check if advanced morphological operations are available
            return core_mask, cladding_mask, ferrule_mask  # Return unchanged if scipy not available
            
        kernel = np.ones((5, 5), dtype=np.uint8)  # 5x5 square structuring element for morphological operations
        
        core_mask = binary_closing(binary_opening(core_mask, kernel), kernel).astype(np.uint8)  # Remove small holes and protrusions
        cladding_mask = binary_closing(binary_opening(cladding_mask, kernel), kernel).astype(np.uint8)  # Clean cladding mask
        
        cladding_mask[core_mask == 1] = 0  # Remove cladding pixels that overlap with core
        ferrule_mask[core_mask == 1] = 0  # Remove ferrule pixels that overlap with core
        ferrule_mask[cladding_mask == 1] = 0  # Remove ferrule pixels that overlap with cladding
        
        return core_mask, cladding_mask, ferrule_mask  # Return cleaned, mutually exclusive masks

class SegmentationResult:  # Container class for storing results from individual segmentation methods
    """Standardized result format for all segmentation methods"""  # Ensures consistent data structure across methods
    def __init__(self, method_name: str, image_path: str):  # Initialize result object with method and image info
        self.method_name = method_name  # Name of the segmentation method
        self.image_path = image_path  # Path to the processed image
        self.center = None  # Tuple of (x, y) center coordinates
        self.core_radius = None  # Radius of the fiber core in pixels
        self.cladding_radius = None  # Radius of the fiber cladding in pixels
        self.masks = None  # Dictionary containing core, cladding, and ferrule masks
        self.confidence = 0.5  # Confidence score for this segmentation (0-1)
        self.execution_time = 0.0  # Time taken to run this method in seconds
        self.error = None  # Error message if method failed
        
    def to_dict(self):  # Convert result object to dictionary for JSON serialization
        return {  # Create dictionary representation of result
            'method_name': self.method_name,  # Include method identifier
            'center': self.center,  # Include center coordinates (may be None)
            'core_radius': self.core_radius,  # Include core radius (may be None)
            'cladding_radius': self.cladding_radius,  # Include cladding radius (may be None)
            'confidence': self.confidence,  # Include confidence score
            'execution_time': self.execution_time,  # Include execution duration
            'error': self.error,  # Include error message (None if successful)
            'has_masks': self.masks is not None  # Boolean flag indicating if masks were generated
        }

class UnifiedSegmentationSystem:  # Main orchestrator class that manages the entire segmentation pipeline
    """Main unifier system that orchestrates all segmentation methods"""  # Coordinates multiple methods and consensus
    
    def __init__(self, methods_dir: str = "zones_methods"):  # Initialize system with directory containing method scripts
        self.methods_dir = Path(methods_dir)  # Convert to Path object for cross-platform compatibility
        self.output_dir = Path("output")  # Directory for saving results
        self.output_dir.mkdir(exist_ok=True)  # Create output directory if it doesn't exist
        
        self.dataset_stats = {'method_scores': {}, 'method_accuracy': {}}  # Initialize performance tracking
        
        self.knowledge_file = self.output_dir / "segmentation_knowledge.json"  # Path to persistent knowledge storage
        self.load_knowledge()  # Load historical performance data
        
        self.methods = {}  # Dictionary to store available segmentation methods
        self.load_methods()  # Discover and load method scripts
        
        self.consensus_system = EnhancedConsensusSystem()  # Initialize consensus algorithm
        
        self.vulnerable_methods = [  # List of methods that benefit from pre-processed (inpainted) images
            'adaptive_intensity', 'gradient_approach', 'guess_approach', 'threshold_separation', 'intelligent_segmenter'
        ]  # These methods are sensitive to image defects

    def load_knowledge(self):  # Load historical performance data from previous runs
        if self.knowledge_file.exists():  # Check if knowledge file exists
            try:  # Attempt to load JSON data
                with open(self.knowledge_file, 'r') as f:  # Open knowledge file for reading
                    self.dataset_stats.update(json.load(f))  # Update stats with loaded data
                    print(f"✓ Loaded knowledge from {self.knowledge_file}")  # Success message
            except Exception as e:  # Handle any loading errors
                print(f"! Could not load knowledge ({e}), starting fresh")  # Error message with fallback
    
    def save_knowledge(self):  # Persist performance data for future runs
        with open(self.knowledge_file, 'w') as f:  # Open knowledge file for writing
            json.dump(self.dataset_stats, f, indent=4, cls=NumpyEncoder)  # Save stats with custom encoder
        print(f"✓ Saved updated knowledge to {self.knowledge_file}")  # Confirmation message
    
    def load_methods(self):  # Discover and register available segmentation methods
        method_files = [  # List of expected method script filenames
            'adaptive_intensity.py', 'bright_core_extractor.py', 'computational_separation.py',
            'geometric_approach.py', 'gradient_approach.py', 'guess_approach.py',
            'hough_separation.py', 'segmentation.py', 'threshold_separation.py',
            'unified_core_cladding_detector.py', 'intelligent_segmenter.py' 
        ]  # Each file contains a different segmentation approach
        
        for method_file in method_files:  # Iterate through expected method files
            method_name = Path(method_file).stem  # Extract method name from filename (without .py)
            method_path = self.methods_dir / method_file  # Construct full path to method script
            if method_path.exists():  # Check if method script exists
                self.methods[method_name] = {  # Register method with its metadata
                    'path': method_path,  # Store path to script
                    'score': self.dataset_stats['method_scores'].get(method_name, 1.0)  # Load historical score or default to 1.0
                }
                print(f"✓ Loaded method: {method_name} (score: {self.methods[method_name]['score']:.2f})")  # Confirm loading with score

    def detect_and_inpaint_anomalies(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # Pre-process image to remove defects
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image  # Convert to grayscale if needed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Create elliptical structuring element for morphology
        blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)  # Black-hat transform to detect dark spots
        _, defect_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)  # Threshold to create binary defect mask
        if HAS_SCIPY_FULL:  # Check if advanced morphological operations available
            defect_mask = binary_opening(defect_mask, structure=np.ones((3,3)), iterations=2).astype(np.uint8)  # Clean up small noise
        inpainted_image = cv2.inpaint(image, defect_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)  # Fill defects using Telea algorithm
        return inpainted_image, defect_mask  # Return cleaned image and defect locations

    def run_method_isolated(self, method_name: str, image_path: Path, temp_output: Path) -> dict:  # Execute method in isolated subprocess
        """Generates a wrapper script to run a method in isolation and captures its JSON output."""  # Prevents method crashes from affecting main process
        result_file = temp_output / f"{method_name}_result.json"  # Path for method to write its results
        runner_script_path = temp_output / "runner.py"  # Path for temporary runner script
        
        script_content = f"""  # Begin generating Python script as string
import sys, json, os  # Essential imports for the isolated script
from pathlib import Path  # Path handling in isolated environment
import matplotlib  # Import matplotlib to set backend
matplotlib.use('Agg')  # Force non-interactive backend in subprocess
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Prevent Qt GUI initialization
sys.path.insert(0, r"{self.methods_dir.resolve()}")  # Add methods directory to Python path for imports

def main():  # Main function for isolated execution
    image_path_str = r"{image_path.resolve()}"  # Absolute path to input image
    output_dir_str = r"{temp_output.resolve()}"  # Absolute path to output directory
    result = {{'success': False, 'error': 'Unknown execution error'}}  # Default failure result
    try:  # Wrap method execution in try-except
"""  # Script header with environment setup
        method_map = {  # Dictionary mapping method names to their specific import and call syntax
            'adaptive_intensity': "from adaptive_intensity import adaptive_segment_image\n        result = adaptive_segment_image(image_path_str, output_dir=output_dir_str)",
            'bright_core_extractor': "from bright_core_extractor import analyze_core\n        result = analyze_core(image_path_str, output_dir_str)",
            'computational_separation': "from computational_separation import process_fiber_image_veridian\n        result = process_fiber_image_veridian(image_path_str, output_dir_str)",
            'geometric_approach': "from geometric_approach import segment_with_geometric\n        result = segment_with_geometric(image_path_str, output_dir_str)",
            'gradient_approach': "from gradient_approach import segment_with_gradient\n        result = segment_with_gradient(image_path_str, output_dir_str)",
            'guess_approach': "from guess_approach import segment_fiber_with_multimodal_analysis\n        result = segment_fiber_with_multimodal_analysis(image_path_str, output_dir_str)",
            'hough_separation': "from hough_separation import segment_with_hough\n        result = segment_with_hough(image_path_str, output_dir_str)",
            'segmentation': "from segmentation import run_segmentation_pipeline, DEFAULT_CONFIG\n        pipeline_result = run_segmentation_pipeline(Path(image_path_str), {{}}, DEFAULT_CONFIG, Path(output_dir_str))\n        result = pipeline_result['result'] if pipeline_result and 'result' in pipeline_result else {{'success': False, 'error': 'Pipeline failed'}}",
            'threshold_separation': "from threshold_separation import segment_with_threshold\n        result = segment_with_threshold(image_path_str, output_dir_str)",
            'unified_core_cladding_detector': "from unified_core_cladding_detector import detect_core_cladding\n        result = detect_core_cladding(image_path_str, output_dir_str)",

            'intelligent_segmenter': "from intelligent_segmenter import run_intelligent_segmentation\n        result = run_intelligent_segmentation(image_path_str, output_dir_str)", 

        }  # Each entry contains the specific import and function call for that method
        
        call_logic = method_map.get(method_name)  # Look up the call syntax for requested method
        if not call_logic:  # Check if method is implemented
             return {'success': False, 'error': f'Runner for method {method_name} not implemented.'}  # Return error if method unknown
        script_content += f"        {call_logic}\n"  # Add method-specific code to script

        script_content += f"""  # Continue building script with error handling and output
    except Exception as e:  # Catch any exceptions from method execution
        import traceback  # Import traceback for detailed error information
        result['error'] = f"Exception in {{method_name}}: {{e}}\\n{{traceback.format_exc()}}"  # Format error with stack trace
    
    with open(r"{result_file.resolve()}", 'w') as f:  # Open result file for writing
        json.dump(result, f, indent=4)  # Write result dictionary as JSON

if __name__ == "__main__":  # Script entry point
    main()  # Execute main function
"""  # Complete script generation
        with open(runner_script_path, 'w') as f: f.write(script_content)  # Write generated script to file

        try:  # Attempt to run the generated script
            subprocess.run(  # Execute script as subprocess
                [sys.executable, str(runner_script_path)],  # Use current Python interpreter
                capture_output=True, text=True, timeout=120, check=False,  # Capture output, 2-minute timeout
                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen', 'MPLBACKEND': 'Agg'}  # Set environment for headless operation
            )
            if result_file.exists():  # Check if method wrote results
                with open(result_file, 'r') as f: return json.load(f)  # Read and return JSON results
            return {'success': False, 'error': 'No result file produced.'}  # Error if no output file
        except subprocess.TimeoutExpired:  # Handle methods that run too long
            return {'success': False, 'error': 'Method timed out'}  # Return timeout error
        except Exception as e:  # Handle any other subprocess errors
            return {'success': False, 'error': f'Subprocess execution failed: {e}'}  # Return execution error

    def run_method(self, method_name: str, image_path: Path, image_shape: Tuple[int, int]) -> SegmentationResult:  # Execute single method and process results
        result = SegmentationResult(method_name, str(image_path))  # Create result object for this method
        start_time = time.time()  # Record start time for execution timing
        
        with tempfile.TemporaryDirectory() as temp_dir:  # Create temporary directory for method output
            method_output = self.run_method_isolated(method_name, image_path, Path(temp_dir))  # Run method in isolation
            
            if method_output and method_output.get('success'):  # Check if method succeeded
                result.center = tuple(method_output.get('center')) if method_output.get('center') else None  # Extract center coordinates
                result.core_radius = method_output.get('core_radius')  # Extract core radius
                result.cladding_radius = method_output.get('cladding_radius')  # Extract cladding radius
                result.confidence = method_output.get('confidence', 0.5)  # Extract confidence or use default

                if all([result.center, result.core_radius, result.cladding_radius]):  # Check if all parameters present
                    result.masks = self.consensus_system.create_masks_from_params(  # Generate masks from parameters
                        result.center, result.core_radius, result.cladding_radius, image_shape  # Pass geometric data
                    )
                    if result.masks and result.masks.get('core') is not None:  # Verify core mask exists
                        contours, _ = cv2.findContours(result.masks['core'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find core contours
                        if contours:  # Check if contours found
                            cnt = max(contours, key=cv2.contourArea)  # Select largest contour (main core)
                            area = cv2.contourArea(cnt)  # Calculate contour area
                            perimeter = cv2.arcLength(cnt, True)  # Calculate contour perimeter
                            if perimeter > 0:  # Avoid division by zero
                                circularity = (4 * np.pi * area) / (perimeter**2)  # Compute circularity metric (1.0 = perfect circle)
                                if circularity < 0.85:  # Check if shape is sufficiently circular
                                    result.confidence *= 0.5  # Penalize confidence for non-circular cores
                                    print(f"  ! Penalizing {method_name} for low circularity ({circularity:.2f})")  # Log penalty
                else:  # Missing required parameters
                    result.error = "Method returned invalid/missing parameters."  # Set error message
            else:  # Method execution failed
                result.error = method_output.get('error', 'Unknown failure') if method_output else "Empty method output"  # Extract or generate error message
        
        result.execution_time = time.time() - start_time  # Calculate total execution time
        return result  # Return completed result object

    def update_learning(self, consensus: Dict, all_results: List[SegmentationResult]):  # Update method performance scores based on consensus
        print("\nUpdating learning model...")  # Progress indicator
        consensus_masks = consensus['masks']  # Extract consensus masks for comparison
        
        for result in all_results:  # Evaluate each method against consensus
            if result.error or not result.masks: continue  # Skip failed methods
            core_iou = self.consensus_system._calculate_iou(result.masks.get('core'), consensus_masks.get('core'))  # Compare core masks
            cladding_iou = self.consensus_system._calculate_iou(result.masks.get('cladding'), consensus_masks.get('cladding'))  # Compare cladding masks
            avg_iou = (core_iou + cladding_iou) / 2  # Average IoU across both regions
            
            current_score = self.dataset_stats['method_scores'].get(result.method_name, 1.0)  # Get current performance score
            learning_rate = 0.1  # Learning rate for exponential moving average
            target_score = 0.1 + (1.9 * avg_iou)  # Map IoU [0,1] to score [0.1,2.0]
            new_score = current_score * (1 - learning_rate) + target_score * learning_rate  # Exponential moving average update
            
            self.dataset_stats['method_scores'][result.method_name] = new_score  # Update method score
            self.dataset_stats['method_accuracy'][result.method_name] = avg_iou  # Store accuracy for this run
            self.methods[result.method_name]['score'] = new_score  # Update in-memory score
        
        print("  ✓ Method scores updated.")  # Confirmation message
        self.save_knowledge()  # Persist updated scores
    
    def process_image(self, image_path: Path, output_dir_str: str) -> Optional[Dict]:  # Main processing pipeline for single image
        print(f"\n{'='*25} Processing: {image_path.name} {'='*25}")  # Visual separator for new image
        original_img = cv2.imread(str(image_path))  # Load image from disk
        if original_img is None: return None  # Exit if image loading failed
        image_shape = original_img.shape[:2]  # Extract height and width (ignore channels)
        
        print("\nRunning pre-processing: Anomaly detection and inpainting...")  # Status update
        inpainted_img, defect_mask = self.detect_and_inpaint_anomalies(original_img)  # Remove image defects
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:  # Create temporary file for inpainted image
            inpainted_image_path = Path(tmp_f.name)  # Get path to temporary file
            cv2.imwrite(str(inpainted_image_path), inpainted_img)  # Save inpainted image
        print("  ✓ Inpainting complete.")  # Confirmation message

        all_results = []  # List to collect results from all methods
        for method_name in self.methods:  # Run each registered method
            use_inpainted = method_name in self.vulnerable_methods  # Check if method needs clean image
            current_image_path = inpainted_image_path if use_inpainted else image_path  # Select appropriate image
            
            print(f"\nRunning {method_name} (using {'inpainted' if use_inpainted else 'original'} image)...")  # Status update
            result = self.run_method(method_name, current_image_path, image_shape)  # Execute method
            all_results.append(result)  # Store result
            
            if result.error: print(f"  ✗ Failed: {result.error}")  # Log failure
            else: print(f"  ✓ Success - Confidence: {result.confidence:.2f}, Time: {result.execution_time:.2f}s")  # Log success with metrics
        
        consensus = self.consensus_system.generate_consensus(  # Combine all results into consensus
            all_results, {name: info['score'] for name, info in self.methods.items()}, image_shape  # Pass results, scores, and image size
        )
        
        if consensus:  # Check if consensus was achieved
            print("\n✓ Model-driven consensus achieved.")  # Success message
            print(f"  Contributing methods: {', '.join(consensus['contributing_methods'])}")  # List methods that agreed
            self.update_learning(consensus, all_results)  # Update performance scores
            self.save_results(image_path, consensus, original_img, output_dir_str, defect_mask)  # Save all outputs
        else:  # Consensus failed
            print("\n✗ FINAL: No consensus could be reached.")  # Failure message
            
        os.remove(inpainted_image_path)  # Clean up temporary inpainted image
        return consensus  # Return consensus result (or None)

    def save_results(self, image_path: Path, consensus: Dict, image: np.ndarray, output_dir: str, defect_mask: np.ndarray):  # Save all outputs to disk
        """Saves consensus results, separated region images, and visualizations."""  # Comprehensive output generation
        result_dir = Path(output_dir)  # Convert to Path object
        result_dir.mkdir(exist_ok=True)  # Create output directory if needed
        
        # Save JSON report
        report = {k: v for k, v in consensus.items() if k not in ['masks', 'all_results']}  # Extract serializable data
        report['method_accuracies'] = self.dataset_stats.get('method_accuracy', {})  # Add accuracy metrics
        with open(result_dir / "consensus_report.json", 'w') as f:  # Open report file
            json.dump(report, f, indent=4, cls=NumpyEncoder)  # Write JSON with custom encoder
        
        # Get final masks
        masks = consensus['masks']  # Extract mask dictionary
        
        # Save the raw mask images
        cv2.imwrite(str(result_dir / "mask_core.png"), masks['core'] * 255)  # Save core mask as binary image
        cv2.imwrite(str(result_dir / "mask_cladding.png"), masks['cladding'] * 255)  # Save cladding mask as binary image
        cv2.imwrite(str(result_dir / "mask_ferrule.png"), masks['ferrule'] * 255)  # Save ferrule mask as binary image
        cv2.imwrite(str(result_dir / "detected_defects.png"), defect_mask)  # Save defect locations
        
        # Apply masks to the original image to get separated regions
        print("  Applying final masks to create separated region images...")  # Progress update
        region_core = cv2.bitwise_and(image, image, mask=masks['core'])  # Extract core pixels only
        region_cladding = cv2.bitwise_and(image, image, mask=masks['cladding'])  # Extract cladding pixels only
        region_ferrule = cv2.bitwise_and(image, image, mask=masks['ferrule'])  # Extract ferrule pixels only

        # Save the separated region images
        cv2.imwrite(str(result_dir / "region_core.png"), region_core)  # Save isolated core region
        cv2.imwrite(str(result_dir / "region_cladding.png"), region_cladding)  # Save isolated cladding region
        cv2.imwrite(str(result_dir / "region_ferrule.png"), region_ferrule)  # Save isolated ferrule region

        # Pass the regions to the visualization function
        regions = {'core': region_core, 'cladding': region_cladding, 'ferrule': region_ferrule}  # Package regions for visualization
        if HAS_MATPLOTLIB:  # Check if visualization library available
            self.create_summary_visualization(result_dir, image, masks, defect_mask, consensus, regions)  # Generate summary plot
        
        print(f"\n✓ All results, masks, and region images saved to: {result_dir}")  # Confirm save location
    
    def create_summary_visualization(self, result_dir: Path, original_image: np.ndarray, masks: Dict, defect_mask: np.ndarray, consensus: Dict, regions: Dict):  # Generate comprehensive visual summary
        """Creates a comprehensive summary plot including the separated regions."""  # Four-panel visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)  # Create 2x2 subplot grid
        fig.suptitle(f'Unified Segmentation Analysis: {result_dir.name}', fontsize=16)  # Add main title
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
        
        # Plot 1: Original Image with Final Boundaries
        axes[0, 0].imshow(img_rgb)  # Display original image
        axes[0, 0].set_title('Original with Final Boundaries')  # Subplot title
        theta = np.linspace(0, 2 * np.pi, 100)  # Angle values for drawing circles
        cx, cy = consensus['center']  # Extract consensus center
        axes[0, 0].plot(cx + consensus['core_radius'] * np.cos(theta), cy + consensus['core_radius'] * np.sin(theta), 'lime', linewidth=2)  # Draw core boundary
        axes[0, 0].plot(cx + consensus['cladding_radius'] * np.cos(theta), cy + consensus['cladding_radius'] * np.sin(theta), 'cyan', linewidth=2)  # Draw cladding boundary

        # Plot 2: Final Segmentation Mask
        final_mask_viz = np.zeros_like(img_rgb)  # Create blank RGB image
        final_mask_viz[masks['core'] > 0] = [255, 0, 0]  # Color core pixels red
        final_mask_viz[masks['cladding'] > 0] = [0, 255, 0]  # Color cladding pixels green
        final_mask_viz[masks['ferrule'] > 0] = [0, 0, 255]  # Color ferrule pixels blue
        axes[0, 1].imshow(final_mask_viz)  # Display colored mask
        axes[0, 1].set_title('Final Segmentation Masks')  # Subplot title
        
        # Plot 3: Method Performance Text
        axes[1, 0].axis('off')  # Turn off axis for text display
        text_content = "Method Performance (IoU):\n" + "-"*25  # Header for performance table
        accuracies = self.dataset_stats.get('method_accuracy', {})  # Get accuracy metrics
        sorted_methods = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)  # Sort by accuracy descending
        for method, acc in sorted_methods:  # Format each method's performance
            text_content += f"\n{method[:20].ljust(22)}: {acc:.3f}"  # Truncate long names, align columns
        axes[1, 0].text(0.05, 0.95, text_content, family='monospace', verticalalignment='top', fontsize=10)  # Display performance text

        # Plot 4: Separated Regions Combined
        composite_image = cv2.cvtColor(sum(regions.values()), cv2.COLOR_BGR2RGB)  # Combine all regions and convert color
        axes[1, 1].imshow(composite_image)  # Display combined regions
        axes[1, 1].set_title('Final Separated Regions')  # Subplot title

        for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])  # Remove tick marks from all subplots
        plt.savefig(result_dir / "summary_analysis.png", dpi=150)  # Save high-resolution summary image
        plt.close()  # Close figure to free memory

    def run(self):  # Main entry point for interactive execution
        print("\n" + "="*80)  # Visual separator
        print("UNIFIED FIBER OPTIC SEGMENTATION SYSTEM".center(80))  # Centered title
        print("Model-Driven Consensus Edition".center(80))  # Centered subtitle
        print("="*80)  # Visual separator
        if not self.methods:  # Check if any methods were loaded
            print(f"✗ Error: No methods found in '{self.methods_dir}'.")  # Error message
            return  # Exit if no methods available
        folder_path = Path(input("\nEnter the folder path containing images: ").strip().strip('"\''))  # Get input folder from user
        if not folder_path.is_dir():  # Validate folder exists
            print(f"✗ Folder not found: {folder_path}")  # Error message
            return  # Exit if folder invalid
        image_files = list(folder_path.glob('*.png')) + list(folder_path.glob('*.jpg'))  # Find all image files
        print(f"Found {len(image_files)} images to process.")  # Report number of images
        for img_path in image_files:  # Process each image
            self.process_image(img_path, str(self.output_dir / img_path.stem))  # Run full pipeline
        print("\n" + "="*80 + "\n" + "Processing complete.".center(80) + "\n" + "="*80)  # Final status message

def main():  # Script entry point
    methods_dir = sys.argv[1] if len(sys.argv) > 1 else "zones_methods"  # Get methods directory from command line or use default
    system = UnifiedSegmentationSystem(methods_dir)  # Create system instance
    system.run()  # Start interactive processing

if __name__ == "__main__":  # Check if script is being run directly (not imported)
    main()  # Execute main function