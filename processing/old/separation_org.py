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
        # Fixed mapping of file names to method names
        method_files = [
            ('guess_approach.py', 'guess_approach'),
            ('hough_separation.py', 'hough_separation'),  # Fixed: was 'hough_seperation'
            ('segmentation.py', 'segmentation'),
            ('threshold_separation.py', 'threshold_separation'),  # Fixed: was 'threshold_seperation'
            ('adaptive_intensity.py', 'adaptive_intensity'),  # Fixed: was 'adaptive_intensity_approach'
            ('computational_separation.py', 'computational_separation'),
            ('gradient_approach.py', 'gradient_approach'),
            ('bright_core_extractor.py', 'bright_core_extractor')
        ]
        
        for method_file, method_name in method_files:
            method_path = self.methods_dir / method_file
            if method_path.exists():
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
        if center is None or core_radius is None:
            return None
            
        cx, cy = center
        
        # Check if parameters are reasonable
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        if core_radius <= 0:
            return None
        if core_radius > min(w, h):
            return None

        # Create masks
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        core_mask = (dist_from_center <= core_radius).astype(np.uint8)

        # Handle case where cladding is not found by a method
        if cladding_radius is not None and cladding_radius > core_radius:
            cladding_mask = ((dist_from_center > core_radius) & 
                            (dist_from_center <= cladding_radius)).astype(np.uint8)
            ferrule_mask = (dist_from_center > cladding_radius).astype(np.uint8)
        else:
            # If no valid cladding, everything outside the core is ferrule
            cladding_mask = np.zeros_like(core_mask)
            ferrule_mask = (dist_from_center > core_radius).astype(np.uint8)
        
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
from pathlib import Path
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
            elif method_name == 'hough_separation':  # Fixed name
                f.write(f"""
from hough_separation import segment_with_hough
result = segment_with_hough(r"{image_path}", r"{temp_output}")
with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
    json.dump(result, outf)
""")
            elif method_name == 'segmentation':
                f.write(f"""
from segmentation import run_segmentation_pipeline, DEFAULT_CONFIG
priors = {json.dumps(self.dataset_stats)}
seg_result = run_segmentation_pipeline(Path(r"{image_path}"), priors, DEFAULT_CONFIG, Path(r"{temp_output}"))
if seg_result and 'result' in seg_result:
    with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
        json.dump(seg_result['result'], outf)
""")
            elif method_name == 'threshold_separation':  # Fixed name
                f.write(f"""
from threshold_separation import segment_with_threshold
result = segment_with_threshold(r"{image_path}", r"{temp_output}")
with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
    json.dump(result, outf)
""")
            elif method_name == 'adaptive_intensity':  # Fixed name
                f.write(f"""
from adaptive_intensity import adaptive_segment_image
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
            elif method_name == 'bright_core_extractor':
                f.write(f"""
from bright_core_extractor import analyze_core
result = analyze_core(r"{image_path}", r"{temp_output}")
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
                error_details = (
                "No result file generated. This usually means the subprocess crashed on import.\n"
                f"--> Exit Code: {process.returncode}\n"
                f"--> Stderr: {process.stderr.strip()}\n"
                f"--> Stdout: {process.stdout.strip()}"
                )
                result.error = error_details
                
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
        """Find consensus using TRUE pixel-by-pixel MAJORITY voting"""
        valid_results = [r for r in results if r.error is None and r.masks is not None]
        
        if not valid_results:
            return None
            
        print(f"\nPerforming pixel-by-pixel MAJORITY voting with {len(valid_results)} valid results...")
        
        h, w = image_shape
        
        # For each pixel, count votes for each region
        # Each method gets ONE vote per pixel
        pixel_votes = np.zeros((h, w, 3), dtype=np.int32)  # 3 regions: core, cladding, ferrule
        
        # Collect votes from each method
        for r in valid_results:
            # For each pixel, the method votes for ONE region
            core_votes = r.masks['core'] > 0
            cladding_votes = r.masks['cladding'] > 0
            ferrule_votes = r.masks['ferrule'] > 0
            
            # Add votes (each method gets exactly one vote per pixel)
            pixel_votes[:, :, 0] += core_votes.astype(np.int32)
            pixel_votes[:, :, 1] += cladding_votes.astype(np.int32)
            pixel_votes[:, :, 2] += ferrule_votes.astype(np.int32)
        
        # For each pixel, assign it to the region with the MOST votes (majority rules)
        # Even if only 2 out of 7 methods agree, that's still the majority for that pixel
        final_core_mask = np.zeros((h, w), dtype=np.uint8)
        final_cladding_mask = np.zeros((h, w), dtype=np.uint8)
        final_ferrule_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Find which region has the most votes for each pixel
        winning_region = np.argmax(pixel_votes, axis=2)
        
        # Assign pixels based on majority vote
        final_core_mask[winning_region == 0] = 1
        final_cladding_mask[winning_region == 1] = 1
        final_ferrule_mask[winning_region == 2] = 1
        
        # Handle ties (if votes are equal)
        # Check for ties between regions
        max_votes = np.max(pixel_votes, axis=2)
        
        # Count how many regions have the max vote count for each pixel
        tie_count = np.sum(pixel_votes == max_votes[:, :, np.newaxis], axis=2)
        tie_pixels = tie_count > 1
        
        if np.any(tie_pixels):
            print(f"  Found {np.sum(tie_pixels)} pixels with tied votes")
            # For ties, prefer: core > cladding > ferrule
            tie_coords = np.where(tie_pixels)
            for i in range(len(tie_coords[0])):
                y, x = tie_coords[0][i], tie_coords[1][i]
                votes = pixel_votes[y, x]
                max_vote = np.max(votes)
                
                # Clear current assignments
                final_core_mask[y, x] = 0
                final_cladding_mask[y, x] = 0
                final_ferrule_mask[y, x] = 0
                
                # Assign based on preference
                if votes[0] == max_vote:  # Core
                    final_core_mask[y, x] = 1
                elif votes[1] == max_vote:  # Cladding
                    final_cladding_mask[y, x] = 1
                else:  # Ferrule
                    final_ferrule_mask[y, x] = 1
        
        # Calculate consensus metrics
        total_pixels = h * w
        consensus_strength = {
            'core': np.sum(pixel_votes[:, :, 0]) / (total_pixels * len(valid_results)),
            'cladding': np.sum(pixel_votes[:, :, 1]) / (total_pixels * len(valid_results)),
            'ferrule': np.sum(pixel_votes[:, :, 2]) / (total_pixels * len(valid_results))
        }
        
        # Calculate agreement statistics
        agreement_stats = {
            'unanimous_pixels': np.sum(max_votes == len(valid_results)),
            'majority_pixels': np.sum(max_votes > len(valid_results) / 2),
            'minority_pixels': np.sum(max_votes <= len(valid_results) / 2),
            'min_agreement': 2  # Even 2 methods agreeing is valid
        }
        
        print(f"  Unanimous agreement: {agreement_stats['unanimous_pixels']} pixels")
        print(f"  Majority agreement: {agreement_stats['majority_pixels']} pixels")
        print(f"  Minority agreement: {agreement_stats['minority_pixels']} pixels")
        
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
            'agreement_stats': agreement_stats,
            'vote_counts': pixel_votes,  # Keep vote counts for visualization
            'all_results': [r.to_dict() for r in results]
        }
    
    def save_results(self, image_path: Path, consensus: Dict[str, Any], image: np.ndarray, output_dir: str) -> List[str]:
        """Save consensus results and return paths of saved regions."""
        result_dir = Path(output_dir) # Use the passed output directory
        result_dir.mkdir(exist_ok=True) # Ensure it exists
        
        # Extract masks
        core_mask = consensus['masks']['core']
        cladding_mask = consensus['masks']['cladding']
        ferrule_mask = consensus['masks']['ferrule']
        
        # Save consensus data (without the large mask arrays)
        consensus_save = consensus.copy()
        consensus_save.pop('masks', None)
        consensus_save.pop('vote_counts', None)
        with open(result_dir / "consensus.json", 'w') as f:
            json.dump(consensus_save, f, indent=4, cls=NumpyEncoder)
        
        # Save the mask arrays separately
        np.save(result_dir / "core_mask.npy", core_mask)
        np.save(result_dir / "cladding_mask.npy", cladding_mask)
        np.save(result_dir / "ferrule_mask.npy", ferrule_mask)
        
        # Create visualization of voting results
        self.create_voting_visualization(result_dir, core_mask, cladding_mask, ferrule_mask, 
                                       image, consensus.get('vote_counts'))
        
        # Extract regions using masks (NO BINARY FILTERING)
        region_core = cv2.bitwise_and(image, image, mask=(core_mask * 255).astype(np.uint8))
        region_cladding = cv2.bitwise_and(image, image, mask=(cladding_mask * 255).astype(np.uint8))
        region_ferrule = cv2.bitwise_and(image, image, mask=(ferrule_mask * 255).astype(np.uint8))
        
        # Save segmented regions
        cv2.imwrite(str(result_dir / "region_core.png"), region_core)
        cv2.imwrite(str(result_dir / "region_cladding.png"), region_cladding)
        cv2.imwrite(str(result_dir / "region_ferrule.png"), region_ferrule)
        
        print(f"\n✓ Results saved to: {result_dir}")
        saved_region_paths = [
        str(result_dir / "region_core.png"),
        str(result_dir / "region_cladding.png"),
        str(result_dir / "region_ferrule.png")
        ]
        return saved_region_paths
    
    def create_voting_visualization(self, result_dir: Path, core_mask: np.ndarray, 
                                   cladding_mask: np.ndarray, ferrule_mask: np.ndarray, 
                                   original_image: np.ndarray, vote_counts: Optional[np.ndarray] = None):
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
        
        # If we have vote counts, create a vote strength visualization
        if vote_counts is not None and HAS_MATPLOTLIB:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Pixel-by-Pixel Voting Results', fontsize=16)
            
            # Show vote counts for each region
            im0 = axes[0, 0].imshow(vote_counts[:, :, 0], cmap='Reds', vmin=0, vmax=vote_counts.max())
            axes[0, 0].set_title('Core Votes')
            plt.colorbar(im0, ax=axes[0, 0])
            
            im1 = axes[0, 1].imshow(vote_counts[:, :, 1], cmap='Greens', vmin=0, vmax=vote_counts.max())
            axes[0, 1].set_title('Cladding Votes')
            plt.colorbar(im1, ax=axes[0, 1])
            
            im2 = axes[1, 0].imshow(vote_counts[:, :, 2], cmap='Blues', vmin=0, vmax=vote_counts.max())
            axes[1, 0].set_title('Ferrule Votes')
            plt.colorbar(im2, ax=axes[1, 0])
            
            # Show final result
            axes[1, 1].imshow(mask_viz)
            axes[1, 1].set_title('Final Segmentation (Majority Vote)')
            
            for ax in axes.flat:
                ax.axis('off')
                
            plt.tight_layout()
            plt.savefig(str(result_dir / "voting_analysis.png"), dpi=150)
            plt.close()
        
        print("  ✓ Created voting visualization")
    
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
    
    def process_image(self, image_path: Path, output_dir: str) -> Dict[str, Any]:
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
            saved_regions = self.save_results(image_path, consensus, img, output_dir)
            consensus['saved_regions'] = saved_regions
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
        print("True Pixel-by-Pixel Majority Voting Edition".center(80))
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