import os
import sys
import json
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Any
import shutil
import warnings
warnings.filterwarnings('ignore')

# Import matplotlib for visualizations
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, some visualizations will be skipped")

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class SegmentationResult:
    """Standardized result format for all segmentation methods"""
    def __init__(self, method_name: str, image_path: str):
        self.method_name = method_name
        self.image_path = image_path
        self.center = None
        self.core_radius = None
        self.cladding_radius = None
        self.masks = None  # Will store the actual mask arrays
        self.confidence = 0.0
        self.execution_time = 0.0
        self.error = None
        
    def to_dict(self):
        return {
            'method_name': self.method_name,
            'center': self.center,
            'core_radius': self.core_radius,
            'cladding_radius': self.cladding_radius,
            'confidence': self.confidence,
            'execution_time': self.execution_time,
            'error': self.error,
            'has_masks': self.masks is not None
        }

class UnifiedSegmentationSystem:
    """Main unifier system that orchestrates all segmentation methods"""
    
    def __init__(self, methods_dir: str = "zones_methods"):
        self.methods_dir = Path(methods_dir)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Learning parameters
        self.dataset_stats = {
            'avg_core_radius_ratio': 0.15,
            'avg_cladding_radius_ratio': 0.5,
            'avg_center_offset': 0.02,
            'method_scores': {},
            'method_accuracy': {}  # Track pixel-level accuracy
        }
        
        # Load existing knowledge if available
        self.knowledge_file = self.output_dir / "segmentation_knowledge.json"
        self.load_knowledge()
        
        # Method modules
        self.methods = {}
        self.load_methods()
        
    def load_knowledge(self):
        """Load previously learned parameters"""
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r') as f:
                    saved_knowledge = json.load(f)
                    self.dataset_stats.update(saved_knowledge)
                    print(f"✓ Loaded existing knowledge from {self.knowledge_file}")
            except:
                print("! Could not load existing knowledge, starting fresh")
    
    def save_knowledge(self):
        """Save learned parameters"""
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.dataset_stats, f, indent=4, cls=NumpyEncoder)
        print(f"✓ Saved knowledge to {self.knowledge_file}")
    
    def load_methods(self):
        """Dynamically load all segmentation methods"""
        method_files = [
            'guess_approach.py',
            'hough_seperation.py', 
            'segmentation.py',
            'threshold_seperation.py',
            'adaptive_intensity_approach.py',
            'computational_separation.py',
            'gradient_approach.py'
        ]
        
        for method_file in method_files:
            method_path = self.methods_dir / method_file
            if method_path.exists():
                method_name = method_file.replace('.py', '')
                try:
                    self.methods[method_name] = {
                        'path': method_path,
                        'name': method_name,
                        'score': self.dataset_stats['method_scores'].get(method_name, 1.0),
                        'accuracy': self.dataset_stats['method_accuracy'].get(method_name, 0.5)
                    }
                    print(f"✓ Loaded method: {method_name} (score: {self.methods[method_name]['score']:.2f})")
                except Exception as e:
                    print(f"✗ Failed to load {method_name}: {e}")
    
    def create_masks_from_params(self, center: Tuple[float, float], core_radius: float, 
                               cladding_radius: float, image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Create binary masks from center and radius parameters"""
        h, w = image_shape
        
        # Validate parameters
        if center is None or core_radius is None or cladding_radius is None:
            return None
            
        cx, cy = center
        
        # Check if parameters are reasonable
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        if core_radius <= 0 or cladding_radius <= 0:
            return None
        if core_radius > min(w, h) or cladding_radius > min(w, h):
            return None
            
        # Create masks
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        core_mask = (dist_from_center <= core_radius).astype(np.uint8)
        cladding_mask = ((dist_from_center > core_radius) & 
                        (dist_from_center <= cladding_radius)).astype(np.uint8)
        ferrule_mask = (dist_from_center > cladding_radius).astype(np.uint8)
        
        return {
            'core': core_mask,
            'cladding': cladding_mask,
            'ferrule': ferrule_mask
        }
    
    def run_method_isolated(self, method_name: str, image_path: Path, temp_output: Path) -> SegmentationResult:
        """Run a method in isolation using subprocess to avoid interference"""
        result = SegmentationResult(method_name, str(image_path))
        
        # Create a Python script to run the method in isolation
        runner_script = temp_output / "runner.py"
        with open(runner_script, 'w') as f:
            f.write(f"""
import sys
import json
import os
import numpy as np
sys.path.insert(0, r"{self.methods_dir}")

# Method-specific imports and execution
""")
            
            if method_name == 'guess_approach':
                f.write(f"""
from guess_approach import segment_fiber_with_multimodal_analysis
result = segment_fiber_with_multimodal_analysis(r"{image_path}", r"{temp_output}")
if isinstance(result, dict):
    with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
        json.dump(result, outf)
""")
            elif method_name == 'hough_seperation':
                f.write(f"""
from hough_seperation import segment_with_hough
result = segment_with_hough(r"{image_path}", r"{temp_output}")
with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
    json.dump(result, outf)
""")
            elif method_name == 'segmentation':
                f.write(f"""
from segmentation import run_segmentation_pipeline, DEFAULT_CONFIG
from pathlib import Path
priors = {json.dumps(self.dataset_stats)}
seg_result = run_segmentation_pipeline(Path(r"{image_path}"), priors, DEFAULT_CONFIG, Path(r"{temp_output}"))
if seg_result and 'result' in seg_result:
    with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
        json.dump(seg_result['result'], outf)
""")
            elif method_name == 'threshold_seperation':
                f.write(f"""
from threshold_seperation import segment_with_threshold
result = segment_with_threshold(r"{image_path}", r"{temp_output}")
with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
    json.dump(result, outf)
""")
            elif method_name == 'adaptive_intensity_approach':
                f.write(f"""
from adaptive_intensity_approach import adaptive_segment_image
result = adaptive_segment_image(r"{image_path}", output_dir=r"{temp_output}")
with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
    json.dump(result, outf)
""")
            elif method_name == 'computational_separation':
                f.write(f"""
from computational_separation import process_fiber_image_veridian
result = process_fiber_image_veridian(r"{image_path}", r"{temp_output}")
with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
    json.dump(result, outf)
""")
            elif method_name == 'gradient_approach':
                f.write(f"""
from gradient_approach import segment_with_gradient
result = segment_with_gradient(r"{image_path}", r"{temp_output}")
with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
    json.dump(result, outf)
""")
        
        # Run the script in a subprocess
        try:
            process = subprocess.run(
                [sys.executable, str(runner_script)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Read the result
            result_file = temp_output / 'method_result.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    method_result = json.load(f)
                    
                # Parse the result
                if method_result.get('success'):
                    # Convert numpy types to Python types
                    center = method_result.get('center')
                    if center and isinstance(center, (list, tuple)) and len(center) >= 2:
                        result.center = (float(center[0]), float(center[1]))
                    
                    core_r = method_result.get('core_radius')
                    if core_r is not None:
                        result.core_radius = float(core_r)
                        
                    clad_r = method_result.get('cladding_radius')
                    if clad_r is not None:
                        result.cladding_radius = float(clad_r)
                        
                    result.confidence = float(method_result.get('confidence', 0.5))
                else:
                    result.error = method_result.get('error', 'Unknown error')
            else:
                result.error = f"No result file generated. Stderr: {process.stderr}"
                
        except subprocess.TimeoutExpired:
            result.error = "Method timed out after 60 seconds"
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def run_method(self, method_name: str, image_path: Path, image_shape: Tuple[int, int]) -> SegmentationResult:
        """Run a single segmentation method and return standardized results with masks"""
        result = SegmentationResult(method_name, str(image_path))
        start_time = time.time()
        
        # Create temporary output directory for this method
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = Path(temp_dir) / method_name
            temp_output.mkdir(exist_ok=True)
            
            # Run method in isolation
            result = self.run_method_isolated(method_name, image_path, temp_output)
            
            # Generate masks from parameters if successful
            if result.error is None and result.center is not None:
                masks = self.create_masks_from_params(
                    result.center, result.core_radius, 
                    result.cladding_radius, image_shape
                )
                result.masks = masks
            
        result.execution_time = time.time() - start_time
        return result
    
    def pixel_voting_consensus(self, results: List[SegmentationResult], image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Find consensus using pixel-by-pixel voting"""
        valid_results = [r for r in results if r.error is None and r.masks is not None]
        
        if not valid_results:
            return None
            
        print(f"\nPerforming pixel-by-pixel voting with {len(valid_results)} valid results...")
        
        h, w = image_shape
        
        # Initialize vote accumulation arrays
        core_votes = np.zeros((h, w), dtype=np.int32)
        cladding_votes = np.zeros((h, w), dtype=np.int32)
        ferrule_votes = np.zeros((h, w), dtype=np.int32)
        
        # Accumulate votes from each method
        method_weights = {}
        for r in valid_results:
            # Weight votes by method confidence and score
            weight = r.confidence * self.methods[r.method_name]['score']
            method_weights[r.method_name] = weight
            
            # Add votes (weighted)
            core_votes += (r.masks['core'] * weight).astype(np.int32)
            cladding_votes += (r.masks['cladding'] * weight).astype(np.int32)
            ferrule_votes += (r.masks['ferrule'] * weight).astype(np.int32)
        
        # For each pixel, assign it to the region with the most votes
        total_votes = core_votes + cladding_votes + ferrule_votes
        
        # Avoid division by zero
        total_votes[total_votes == 0] = 1
        
        # Create final masks based on which region got the most votes for each pixel
        final_core_mask = np.zeros((h, w), dtype=np.uint8)
        final_cladding_mask = np.zeros((h, w), dtype=np.uint8)
        final_ferrule_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Assign each pixel to the region with highest vote
        max_votes = np.maximum(core_votes, np.maximum(cladding_votes, ferrule_votes))
        
        final_core_mask[core_votes == max_votes] = 1
        final_cladding_mask[cladding_votes == max_votes] = 1
        final_ferrule_mask[ferrule_votes == max_votes] = 1
        
        # Handle ties by preferring core > cladding > ferrule
        tie_mask = ((core_votes == cladding_votes) | 
                   (core_votes == ferrule_votes) | 
                   (cladding_votes == ferrule_votes))
        
        if np.any(tie_mask):
            # In case of ties, use proximity to center as tiebreaker
            # Find approximate center from core mask
            core_points = np.where(final_core_mask > 0)
            if len(core_points[0]) > 0:
                center_y = np.mean(core_points[0])
                center_x = np.mean(core_points[1])
                
                y_grid, x_grid = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                
                # Use distance to resolve ties
                tie_pixels = np.where(tie_mask)
                for i in range(len(tie_pixels[0])):
                    y, x = tie_pixels[0][i], tie_pixels[1][i]
                    if core_votes[y, x] == max_votes[y, x]:
                        final_core_mask[y, x] = 1
                        final_cladding_mask[y, x] = 0
                        final_ferrule_mask[y, x] = 0
                    elif cladding_votes[y, x] == max_votes[y, x]:
                        final_core_mask[y, x] = 0
                        final_cladding_mask[y, x] = 1
                        final_ferrule_mask[y, x] = 0
        
        # Calculate consensus metrics
        total_pixels = h * w
        consensus_strength = {
            'core': np.sum(core_votes) / (total_pixels * len(valid_results)),
            'cladding': np.sum(cladding_votes) / (total_pixels * len(valid_results)),
            'ferrule': np.sum(ferrule_votes) / (total_pixels * len(valid_results))
        }
        
        # Find which methods contributed most to the consensus
        contributing_methods = [r.method_name for r in valid_results]
        
        return {
            'masks': {
                'core': final_core_mask,
                'cladding': final_cladding_mask,
                'ferrule': final_ferrule_mask
            },
            'consensus_strength': consensus_strength,
            'contributing_methods': contributing_methods,
            'num_valid_results': len(valid_results),
            'method_weights': method_weights,
            'all_results': [r.to_dict() for r in results]
        }
    
    def apply_binary_filter(self, image, mask, threshold_percentile=75, keep_bright=True):
        """Apply binary filter to clean up a region"""
        # Extract the region
        region = cv2.bitwise_and(image, image, mask=(mask * 255).astype(np.uint8))
        
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        # Only process non-zero pixels
        non_zero_pixels = gray[mask > 0]
        if len(non_zero_pixels) == 0:
            return region
        
        # Apply threshold based on percentile of the region
        threshold = np.percentile(non_zero_pixels, threshold_percentile)
        
        # Create binary mask
        if keep_bright:
            # Keep pixels brighter than threshold
            binary_mask = (gray >= threshold) & (mask > 0)
        else:
            # Keep pixels darker than threshold
            binary_mask = (gray < threshold) & (mask > 0)
        
        # Apply morphological operation to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Apply the binary mask
        filtered_region = np.zeros_like(image)
        if len(image.shape) == 3:
            for i in range(3):
                filtered_region[:,:,i] = np.where(binary_mask, image[:,:,i], 0)
        else:
            filtered_region = np.where(binary_mask, image, 0)
        
        return filtered_region
    
    def save_results(self, image_path: Path, consensus: Dict[str, Any], image: np.ndarray):
        """Save consensus results with binary filtering applied only to final output"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = image_path.stem
        result_dir = self.output_dir / f"{base_name}_{timestamp}"
        result_dir.mkdir(exist_ok=True)
        
        # Extract masks
        core_mask = consensus['masks']['core']
        cladding_mask = consensus['masks']['cladding']
        ferrule_mask = consensus['masks']['ferrule']
        
        # Save consensus data (without the large mask arrays)
        consensus_save = consensus.copy()
        consensus_save.pop('masks', None)
        with open(result_dir / "consensus.json", 'w') as f:
            json.dump(consensus_save, f, indent=4, cls=NumpyEncoder)
        
        # Save the mask arrays separately
        np.save(result_dir / "core_mask.npy", core_mask)
        np.save(result_dir / "cladding_mask.npy", cladding_mask)
        np.save(result_dir / "ferrule_mask.npy", ferrule_mask)
        
        # Create visualization of voting results
        self.create_voting_visualization(result_dir, core_mask, cladding_mask, ferrule_mask, image)
        
        # Extract regions using masks
        region_core = cv2.bitwise_and(image, image, mask=(core_mask * 255).astype(np.uint8))
        region_cladding = cv2.bitwise_and(image, image, mask=(cladding_mask * 255).astype(np.uint8))
        region_ferrule = cv2.bitwise_and(image, image, mask=(ferrule_mask * 255).astype(np.uint8))
        
        # Save original segmented regions
        cv2.imwrite(str(result_dir / "region_core_original.png"), region_core)
        cv2.imwrite(str(result_dir / "region_cladding_original.png"), region_cladding)
        cv2.imwrite(str(result_dir / "region_ferrule_original.png"), region_ferrule)
        
        # Apply binary filtering to final results
        print("  Applying binary filters to final results...")
        
        # Core: Keep only bright pixels
        refined_core = self.apply_binary_filter(image, core_mask, threshold_percentile=75, keep_bright=True)
        
        # Cladding: Remove bright pixels (keep dark pixels)
        refined_cladding = self.apply_binary_filter(image, cladding_mask, threshold_percentile=85, keep_bright=False)
        
        # Save refined regions
        cv2.imwrite(str(result_dir / "region_core_refined.png"), refined_core)
        cv2.imwrite(str(result_dir / "region_cladding_refined.png"), refined_cladding)
        
        # Create comparison visualization
        self.create_refinement_visualization(result_dir, region_core, refined_core, 
                                           region_cladding, refined_cladding)
        
        print(f"\n✓ Results saved to: {result_dir}")
    
    def create_voting_visualization(self, result_dir: Path, core_mask: np.ndarray, 
                                   cladding_mask: np.ndarray, ferrule_mask: np.ndarray, 
                                   original_image: np.ndarray):
        """Create visualization of the voting results"""
        # Create color-coded mask visualization
        h, w = core_mask.shape
        mask_viz = np.zeros((h, w, 3), dtype=np.uint8)
        mask_viz[core_mask > 0] = [255, 0, 0]      # Red for core
        mask_viz[cladding_mask > 0] = [0, 255, 0]  # Green for cladding
        mask_viz[ferrule_mask > 0] = [0, 0, 255]   # Blue for ferrule
        
        # Create overlay on original
        overlay = original_image.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        
        # Add semi-transparent colored overlay
        alpha = 0.3
        overlay = cv2.addWeighted(overlay, 1-alpha, mask_viz, alpha, 0)
        
        # Save visualizations
        cv2.imwrite(str(result_dir / "voting_mask_visualization.png"), mask_viz)
        cv2.imwrite(str(result_dir / "voting_overlay.png"), overlay)
        
        print("  ✓ Created voting visualization")
    
    def create_refinement_visualization(self, result_dir: Path, core_orig: np.ndarray, core_refined: np.ndarray, 
                                      clad_orig: np.ndarray, clad_refined: np.ndarray):
        """Create visualization comparing original and refined regions"""
        if not HAS_MATPLOTLIB:
            print("  ! Skipping refinement visualization (matplotlib not available)")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Binary Filter Refinement Results (Applied to Final Output)', fontsize=16)
        
        # Convert to RGB for display if needed
        def to_rgb(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Core comparison
        axes[0, 0].imshow(to_rgb(core_orig))
        axes[0, 0].set_title('Core Region - Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(to_rgb(core_refined))
        axes[0, 1].set_title('Core Region - Refined (Bright Pixels Only)')
        axes[0, 1].axis('off')
        
        # Cladding comparison
        axes[1, 0].imshow(to_rgb(clad_orig))
        axes[1, 0].set_title('Cladding Region - Original')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(to_rgb(clad_refined))
        axes[1, 1].set_title('Cladding Region - Refined (Dark Pixels Only)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(result_dir / "refinement_comparison.png"), dpi=150)
        plt.close()
        
        print("  ✓ Created refinement comparison visualization")
    
    def update_learning(self, consensus: Dict[str, Any], image_shape: Tuple[int, int]):
        """Update learning parameters based on consensus results"""
        if not consensus:
            return
            
        # Update method scores based on participation
        for method in self.methods:
            if method in consensus['contributing_methods']:
                # Increase score for contributing methods
                current_score = self.dataset_stats['method_scores'].get(method, 1.0)
                self.dataset_stats['method_scores'][method] = min(current_score * 1.05, 2.0)
            else:
                # Slightly decrease score for non-contributing methods
                current_score = self.dataset_stats['method_scores'].get(method, 1.0)
                self.dataset_stats['method_scores'][method] = max(current_score * 0.95, 0.1)
                
        # Update method scores in memory
        for method in self.methods:
            self.methods[method]['score'] = self.dataset_stats['method_scores'].get(method, 1.0)
    
    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image through all methods"""
        print(f"\nProcessing: {image_path.name}")
        print("=" * 60)
        
        # Load image to get shape
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"✗ Could not load image: {image_path}")
            return None
            
        image_shape = img.shape[:2]
        
        # Run all methods
        results = []
        for method_name in self.methods:
            print(f"\nRunning {method_name}...")
            result = self.run_method(method_name, image_path, image_shape)
            results.append(result)
            
            if result.error is None:
                print(f"  ✓ Success - Center: {result.center}, Core: {result.core_radius}, Cladding: {result.cladding_radius}")
            else:
                print(f"  ✗ Failed: {result.error}")
        
        # Find consensus using pixel voting
        consensus = self.pixel_voting_consensus(results, image_shape)
        
        if consensus:
            print(f"\n✓ Pixel voting consensus achieved:")
            print(f"  Contributing methods: {', '.join(consensus['contributing_methods'])}")
            print(f"  Consensus strength - Core: {consensus['consensus_strength']['core']:.2f}, "
                  f"Cladding: {consensus['consensus_strength']['cladding']:.2f}, "
                  f"Ferrule: {consensus['consensus_strength']['ferrule']:.2f}")
            
            # Update learning
            self.update_learning(consensus, image_shape)
            
            # Save results
            self.save_results(image_path, consensus, img)
        else:
            print("\n✗ No consensus could be reached")
            
        return consensus
    
    def process_folder(self, folder_path: Path) -> List[Dict[str, Any]]:
        """Process all images in a folder"""
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.json']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {folder_path}")
            return []
        
        print(f"\nFound {len(image_files)} images in {folder_path}")
        
        # Process each image
        all_results = []
        for img_path in sorted(image_files):
            result = self.process_image(img_path)
            if result:
                all_results.append(result)
        
        return all_results
    
    def training_mode(self):
        """Run in training mode for continuous improvement"""
        print("\n=== TRAINING MODE ===")
        print("This mode will continuously process images to improve the system")
        
        # Ask for training parameters
        default_runs = input("How many runs do you want to complete? (0 for unlimited): ").strip()
        max_runs = int(default_runs) if default_runs.isdigit() else 0
        
        run_count = 0
        continue_training = True
        
        while continue_training:
            run_count += 1
            print(f"\n--- Training Run #{run_count} ---")
            
            # Get images for this run
            process_type = input("\nProcess individual images (i) or folder (f)? (i/f): ").strip().lower()
            
            if process_type == 'f':
                folder_path = self.ask_for_folder()
                if folder_path:
                    self.process_folder(folder_path)
            else:
                image_paths = self.ask_for_images()
                if image_paths:
                    for img_path in image_paths:
                        self.process_image(img_path)
                else:
                    print("No images provided. Exiting training mode.")
                    break
                
            # Save updated knowledge after each run
            self.save_knowledge()
            
            # Print current method scores
            print("\nCurrent Method Scores:")
            for method, info in sorted(self.methods.items(), key=lambda x: x[1]['score'], reverse=True):
                print(f"  {method}: {info['score']:.3f}")
                
            # Check if we should continue
            if max_runs > 0 and run_count >= max_runs:
                print(f"\nCompleted {max_runs} training runs.")
                continue_training = False
            else:
                response = input("\nRun another training iteration? (y/n): ").strip().lower()
                continue_training = response == 'y'
                
        print("\nTraining complete. Knowledge saved.")
    
    def ask_for_dataset(self) -> Optional[Path]:
        """Interactive dataset selection"""
        response = input("\nDo you want to use an existing dataset? (y/n): ").strip().lower()
        
        if response == 'y':
            dataset_path = input("Enter the path to the dataset directory: ").strip().strip('"\'')
            dataset_path = Path(dataset_path)
            
            if dataset_path.exists() and dataset_path.is_dir():
                # Load any existing knowledge from the dataset
                json_files = list(dataset_path.glob("*_seg_report.json"))
                if json_files:
                    print(f"Found {len(json_files)} existing reports in dataset")
                return dataset_path
            else:
                print(f"✗ Dataset directory not found: {dataset_path}")
                
        return None
    
    def ask_for_folder(self) -> Optional[Path]:
        """Ask for a folder path"""
        folder_path = input("\nEnter the folder path containing images: ").strip().strip('"\'')
        folder_path = Path(folder_path)
        
        if folder_path.exists() and folder_path.is_dir():
            return folder_path
        else:
            print(f"✗ Folder not found: {folder_path}")
            return None
    
    def ask_for_images(self) -> List[Path]:
        """Interactive image selection"""
        print("\nEnter image paths (space-separated, use quotes for paths with spaces):")
        print("Supported formats: .jpg, .png, .json")
        paths_input = input("> ").strip()
        
        if not paths_input:
            return []
            
        # Parse paths (handling quoted paths)
        import shlex
        path_strings = shlex.split(paths_input)
        
        valid_paths = []
        for path_str in path_strings:
            path = Path(path_str)
            if path.exists() and path.is_file():
                valid_paths.append(path)
            else:
                print(f"✗ Invalid path: {path}")
                
        return valid_paths
    
    def run(self):
        """Main execution flow"""
        print("\n" + "="*80)
        print("UNIFIED FIBER OPTIC SEGMENTATION SYSTEM".center(80))
        print("Pixel-by-Pixel Voting Edition".center(80))
        print("="*80)
        
        # Check if methods directory exists
        if not self.methods_dir.exists():
            print(f"\n✗ Error: Methods directory not found: {self.methods_dir}")
            print(f"Please ensure the directory exists at: {self.methods_dir.absolute()}")
            return
            
        # Ask about dataset
        dataset_path = self.ask_for_dataset()
        
        # Ask about mode
        mode = input("\nSelect mode:\n1. Process images\n2. Training mode\nChoice (1/2): ").strip()
        
        if mode == '2':
            self.training_mode()
        else:
            # Normal processing mode
            process_type = input("\nProcess individual images (i) or folder (f)? (i/f): ").strip().lower()
            
            if process_type == 'f':
                folder_path = self.ask_for_folder()
                if folder_path:
                    self.process_folder(folder_path)
            else:
                image_paths = self.ask_for_images()
                if image_paths:
                    for img_path in image_paths:
                        self.process_image(img_path)
                else:
                    print("No images provided. Exiting.")
                    return
                
            # Save knowledge
            self.save_knowledge()
            
        print("\n" + "="*80)
        print("Processing complete. Thank you!".center(80))
        print("="*80)


def main():
    # Allow custom methods directory via command line argument
    import sys
    methods_dir = sys.argv[1] if len(sys.argv) > 1 else "zones_methods"
    
    system = UnifiedSegmentationSystem(methods_dir)
    system.run()


if __name__ == "__main__":
    main()