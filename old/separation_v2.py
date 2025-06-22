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

# Import scipy components for enhanced voting
try:
    from scipy.ndimage import median_filter, gaussian_filter
    from scipy.ndimage import binary_opening, binary_closing
    from scipy.signal import find_peaks, savgol_filter
    from scipy.sparse.linalg import svds
    from scipy.optimize import minimize
    HAS_SCIPY_FULL = True
except ImportError:
    HAS_SCIPY_FULL = False
    print("Warning: Some scipy components not available, using basic voting")

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

class EnhancedVotingSystem:
    """Enhanced voting system with weighted votes, confidence scoring, and spatial coherence"""
    
    def __init__(self, min_agreement_ratio=0.3, use_spatial_coherence=True):
        self.min_agreement_ratio = min_agreement_ratio
        self.use_spatial_coherence = use_spatial_coherence
        
    def weighted_pixel_voting_consensus(self, 
                                      results: List['SegmentationResult'], 
                                      method_scores: Dict[str, float],
                                      image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Enhanced consensus using weighted voting with confidence scoring
        """
        valid_results = [r for r in results if r.error is None and r.masks is not None]
        
        if not valid_results:
            return None
            
        print(f"\nPerforming weighted pixel voting with {len(valid_results)} valid results...")
        
        h, w = image_shape
        num_methods = len(valid_results)
        
        # Create weighted vote accumulator
        weighted_votes = np.zeros((h, w, 3), dtype=np.float32)
        
        # Track which methods vote for each pixel/region
        method_support = [[[] for _ in range(3)] for _ in range(h * w)]
        
        # Collect weighted votes
        for r in valid_results:
            # Get method weight (default to 1.0 if not found)
            weight = method_scores.get(r.method_name, 1.0)
            
            # Apply method confidence if available
            if hasattr(r, 'confidence') and r.confidence > 0:
                weight *= r.confidence
            
            # Add weighted votes
            weighted_votes[:, :, 0] += (r.masks['core'] > 0).astype(np.float32) * weight
            weighted_votes[:, :, 1] += (r.masks['cladding'] > 0).astype(np.float32) * weight
            weighted_votes[:, :, 2] += (r.masks['ferrule'] > 0).astype(np.float32) * weight
            
            # Track which methods support each region
            for y in range(h):
                for x in range(w):
                    idx = y * w + x
                    if r.masks['core'][y, x] > 0:
                        method_support[idx][0].append(r.method_name)
                    if r.masks['cladding'][y, x] > 0:
                        method_support[idx][1].append(r.method_name)
                    if r.masks['ferrule'][y, x] > 0:
                        method_support[idx][2].append(r.method_name)
        
        # Calculate total weight per pixel
        total_weights = np.sum(weighted_votes, axis=2)
        
        # Create confidence map (0-1) based on agreement strength
        max_votes = np.max(weighted_votes, axis=2)
        confidence_map = np.zeros((h, w), dtype=np.float32)
        
        # Avoid division by zero
        mask = total_weights > 0
        confidence_map[mask] = max_votes[mask] / total_weights[mask]
        
        # Initial classification based on weighted votes
        initial_classification = np.argmax(weighted_votes, axis=2)
        
        # Apply minimum agreement threshold
        max_possible_weight = sum(method_scores.get(r.method_name, 1.0) * r.confidence 
                                 for r in valid_results)
        agreement_threshold = self.min_agreement_ratio * max_possible_weight
        sufficient_agreement = max_votes >= agreement_threshold
        
        # For pixels with insufficient agreement, mark as uncertain
        uncertain_pixels = ~sufficient_agreement
        
        # Apply spatial coherence if enabled
        if self.use_spatial_coherence and HAS_SCIPY_FULL:
            classification = self.apply_spatial_coherence(
                initial_classification, confidence_map, uncertain_pixels
            )
        else:
            classification = initial_classification
            
        # Handle remaining uncertain pixels
        if np.any(uncertain_pixels):
            classification = self.resolve_uncertain_pixels(
                classification, uncertain_pixels, weighted_votes
            )
        
        # Create final masks
        final_core_mask = (classification == 0).astype(np.uint8)
        final_cladding_mask = (classification == 1).astype(np.uint8)
        final_ferrule_mask = (classification == 2).astype(np.uint8)
        
        # Post-process masks to ensure consistency
        final_core_mask, final_cladding_mask, final_ferrule_mask = self.ensure_mask_consistency(
            final_core_mask, final_cladding_mask, final_ferrule_mask
        )
        
        # Calculate detailed statistics
        consensus_strength = {
            'core': float(np.sum(weighted_votes[:, :, 0]) / (h * w * max_possible_weight)) if max_possible_weight > 0 else 0,
            'cladding': float(np.sum(weighted_votes[:, :, 1]) / (h * w * max_possible_weight)) if max_possible_weight > 0 else 0,
            'ferrule': float(np.sum(weighted_votes[:, :, 2]) / (h * w * max_possible_weight)) if max_possible_weight > 0 else 0
        }
        
        # Calculate agreement statistics
        high_confidence_pixels = np.sum(confidence_map > 0.7)
        medium_confidence_pixels = np.sum((confidence_map > 0.4) & (confidence_map <= 0.7))
        low_confidence_pixels = np.sum(confidence_map <= 0.4)
        
        agreement_stats = {
            'high_confidence_pixels': int(high_confidence_pixels),
            'medium_confidence_pixels': int(medium_confidence_pixels),
            'low_confidence_pixels': int(low_confidence_pixels),
            'mean_confidence': float(np.mean(confidence_map)),
            'uncertain_pixels_resolved': int(np.sum(uncertain_pixels))
        }
        
        # Identify most reliable methods
        method_agreement_scores = self.calculate_method_agreement_scores(
            valid_results, classification, method_scores
        )
        
        return {
            'masks': {
                'core': final_core_mask,
                'cladding': final_cladding_mask,
                'ferrule': final_ferrule_mask
            },
            'confidence_map': confidence_map,
            'consensus_strength': consensus_strength,
            'contributing_methods': [r.method_name for r in valid_results],
            'num_valid_results': len(valid_results),
            'agreement_stats': agreement_stats,
            'method_agreement_scores': method_agreement_scores,
            'weighted_votes': weighted_votes,
            'all_results': [r.to_dict() for r in results]
        }
    
    def apply_spatial_coherence(self, classification, confidence_map, uncertain_pixels):
        """Apply spatial coherence to reduce noise and resolve uncertainties"""
        if not HAS_SCIPY_FULL:
            return classification
            
        # Use median filter on classification to enforce spatial coherence
        class_float = classification.astype(np.float32)
        
        # Apply median filter with larger kernel for uncertain areas
        filtered = median_filter(class_float, size=5)
        
        # For uncertain pixels, use the filtered result
        result = classification.copy()
        result[uncertain_pixels] = np.round(filtered[uncertain_pixels]).astype(int)
        
        # Apply additional Gaussian smoothing to confidence map
        smoothed_confidence = gaussian_filter(confidence_map, sigma=2)
        
        # For very low confidence areas, use neighborhood majority
        very_low_conf = smoothed_confidence < 0.3
        if np.any(very_low_conf):
            for y, x in np.argwhere(very_low_conf):
                # Get neighborhood
                y_min, y_max = max(0, y-3), min(classification.shape[0], y+4)
                x_min, x_max = max(0, x-3), min(classification.shape[1], x+4)
                neighborhood = result[y_min:y_max, x_min:x_max]
                
                # Use mode of neighborhood
                if neighborhood.size > 0:
                    counts = np.bincount(neighborhood.ravel(), minlength=3)
                    result[y, x] = np.argmax(counts)
        
        return result
    
    def resolve_uncertain_pixels(self, classification, uncertain_mask, weighted_votes):
        """Resolve uncertain pixels using advanced strategies"""
        result = classification.copy()
        
        # Strategy 1: Use relative vote strengths even if below threshold
        uncertain_coords = np.argwhere(uncertain_mask)
        for y, x in uncertain_coords:
            votes = weighted_votes[y, x]
            if np.sum(votes) > 0:
                # Use relative strengths
                result[y, x] = np.argmax(votes)
            else:
                # No votes at all - use spatial context
                # Get neighborhood mode
                y_min, y_max = max(0, y-5), min(classification.shape[0], y+6)
                x_min, x_max = max(0, x-5), min(classification.shape[1], x+6)
                neighborhood = result[y_min:y_max, x_min:x_max]
                
                if neighborhood.size > 0:
                    counts = np.bincount(neighborhood.ravel(), minlength=3)
                    result[y, x] = np.argmax(counts)
        
        return result
    
    def ensure_mask_consistency(self, core_mask, cladding_mask, ferrule_mask):
        """Ensure masks are mutually exclusive and spatially consistent"""
        if not HAS_SCIPY_FULL:
            return core_mask, cladding_mask, ferrule_mask
            
        # Remove small isolated regions
        kernel = np.ones((3, 3), dtype=np.uint8)
        
        # Clean each mask
        core_mask = binary_opening(core_mask, kernel).astype(np.uint8)
        core_mask = binary_closing(core_mask, kernel).astype(np.uint8)
        
        cladding_mask = binary_opening(cladding_mask, kernel).astype(np.uint8)
        cladding_mask = binary_closing(cladding_mask, kernel).astype(np.uint8)
        
        ferrule_mask = binary_opening(ferrule_mask, kernel).astype(np.uint8)
        ferrule_mask = binary_closing(ferrule_mask, kernel).astype(np.uint8)
        
        # Ensure mutual exclusivity (shouldn't be necessary but just in case)
        overlap = (core_mask + cladding_mask + ferrule_mask) > 1
        if np.any(overlap):
            # Resolve overlaps by distance to expected regions
            h, w = core_mask.shape
            center = (h // 2, w // 2)
            
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_coords - center[1])**2 + (y_coords - center[0])**2)
            
            # For overlapping pixels, assign based on distance
            overlap_coords = np.argwhere(overlap)
            for y, x in overlap_coords:
                dist = dist_from_center[y, x]
                # Simple radial assignment
                if dist < h * 0.1:  # Inner region -> core
                    core_mask[y, x] = 1
                    cladding_mask[y, x] = 0
                    ferrule_mask[y, x] = 0
                elif dist < h * 0.4:  # Middle region -> cladding
                    core_mask[y, x] = 0
                    cladding_mask[y, x] = 1
                    ferrule_mask[y, x] = 0
                else:  # Outer region -> ferrule
                    core_mask[y, x] = 0
                    cladding_mask[y, x] = 0
                    ferrule_mask[y, x] = 1
        
        return core_mask, cladding_mask, ferrule_mask
    
    def calculate_method_agreement_scores(self, valid_results, final_classification, method_scores):
        """Calculate how well each method agrees with the final consensus"""
        agreement_scores = {}
        h, w = final_classification.shape
        
        for result in valid_results:
            # Calculate pixel-wise agreement
            method_classification = np.zeros((h, w), dtype=int)
            method_classification[result.masks['core'] > 0] = 0
            method_classification[result.masks['cladding'] > 0] = 1
            method_classification[result.masks['ferrule'] > 0] = 2
            
            agreement = np.sum(method_classification == final_classification) / (h * w)
            
            # Weight by method score
            weighted_agreement = agreement * method_scores.get(result.method_name, 1.0)
            
            agreement_scores[result.method_name] = {
                'raw_agreement': float(agreement),
                'weighted_agreement': float(weighted_agreement),
                'weight': float(method_scores.get(result.method_name, 1.0))
            }
        
        return agreement_scores

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
            ('guess_approach.py', 'guess_approach'),
            ('hough_separation.py', 'hough_separation'),
            ('segmentation.py', 'segmentation'),
            ('threshold_separation.py', 'threshold_separation'),
            ('adaptive_intensity.py', 'adaptive_intensity'),
            ('computational_separation.py', 'computational_separation'),
            ('gradient_approach.py', 'gradient_approach'),
            ('bright_core_extractor.py', 'bright_core_extractor'),
            ('geometric_approach.py', 'geometric_approach'),
            ('unified_core_cladding_detector.py', 'unified_detector'),
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
            # Add common imports and matplotlib backend fix
            f.write(f"""
import sys
import json
import os
import numpy as np
from pathlib import Path

# Fix matplotlib backend BEFORE any imports
import matplotlib
matplotlib.use('Agg')

# Disable Qt and other GUI backends
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

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
            elif method_name == 'hough_separation':
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
            elif method_name == 'threshold_separation':
                f.write(f"""
from threshold_separation import segment_with_threshold
result = segment_with_threshold(r"{image_path}", r"{temp_output}")
with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
    json.dump(result, outf)
""")
            elif method_name == 'adaptive_intensity':
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
                
            elif method_name == 'geometric_approach':  # ADD THIS ENTIRE BLOCK
                f.write(f"""
from geometric_approach import segment_with_geometric
result = segment_with_geometric(r"{image_path}", r"{temp_output}")
with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
    json.dump(result, outf)
""")
                
            elif method_name == 'unified_detector':
                f.write(f"""
            from unified_core_cladding_detector import detect_core_cladding
            result = detect_core_cladding(r"{image_path}", r"{temp_output}")
            with open(r"{temp_output / 'method_result.json'}", 'w') as outf:
                json.dump(result, outf)
            """)               
        
        # Run the script in a subprocess with increased timeout for complex methods
        timeout = 90 if method_name in ['segmentation', 'computational_separation'] else 60
        
        try:
            # Set environment variables to prevent GUI
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            env['MPLBACKEND'] = 'Agg'
            
            process = subprocess.run(
                [sys.executable, str(runner_script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
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
                # Clean up error messages
                stderr = process.stderr.strip()
                # Remove Qt warnings
                error_lines = []
                for line in stderr.split('\n'):
                    if not any(skip in line for skip in ['QObject::', 'qt.qpa', 'Warning:', 'matplotlib']):
                        error_lines.append(line)
                
                error_details = (
                    "No result file generated.\n"
                    f"--> Exit Code: {process.returncode}\n"
                )
                if error_lines:
                    error_details += f"--> Error: {' '.join(error_lines[:3])}"  # First 3 error lines
                    
                result.error = error_details
                
        except subprocess.TimeoutExpired:
            result.error = f"Method timed out after {timeout} seconds"
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
        """Enhanced consensus using weighted voting with confidence scoring"""
        
        # Create enhanced voting system
        voting_system = EnhancedVotingSystem(
            min_agreement_ratio=0.25,  # Lower threshold for more methods
            use_spatial_coherence=True
        )
        
        # Use existing method scores from the system
        method_scores = self.dataset_stats.get('method_scores', {})
        
        # If no scores exist, use uniform weights
        if not method_scores:
            method_scores = {method: 1.0 for method in self.methods}
        
        # Run enhanced voting
        consensus = voting_system.weighted_pixel_voting_consensus(
            results, method_scores, image_shape
        )
        
        return consensus
    
    def save_results(self, image_path: Path, consensus: Dict[str, Any], image: np.ndarray, output_dir: str) -> List[str]:
        """Save consensus results and return paths of saved regions."""
        result_dir = Path(output_dir)
        result_dir.mkdir(exist_ok=True)
        
        # Extract masks
        core_mask = consensus['masks']['core']
        cladding_mask = consensus['masks']['cladding']
        ferrule_mask = consensus['masks']['ferrule']
        
        # Save consensus data (without the large mask arrays)
        consensus_save = consensus.copy()
        consensus_save.pop('masks', None)
        consensus_save.pop('weighted_votes', None)
        consensus_save.pop('confidence_map', None)
        with open(result_dir / "consensus.json", 'w') as f:
            json.dump(consensus_save, f, indent=4, cls=NumpyEncoder)
        
        # Save the mask arrays separately
        np.save(result_dir / "core_mask.npy", core_mask)
        np.save(result_dir / "cladding_mask.npy", cladding_mask)
        np.save(result_dir / "ferrule_mask.npy", ferrule_mask)
        
        # Save confidence map
        if 'confidence_map' in consensus:
            np.save(result_dir / "confidence_map.npy", consensus['confidence_map'])
        
        # Create visualization of voting results
        self.create_voting_visualization(result_dir, core_mask, cladding_mask, ferrule_mask, 
                                       image, consensus)
        
        # Extract regions using masks
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
                                   original_image: np.ndarray, consensus_data: Optional[Dict] = None):
        """Create visualization of the voting results with confidence map"""
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
        
        # Save confidence map if available
        if consensus_data and 'confidence_map' in consensus_data:
            confidence_map = consensus_data['confidence_map']
            # Convert to uint8 for visualization (0-255)
            confidence_viz = (confidence_map * 255).astype(np.uint8)
            confidence_colored = cv2.applyColorMap(confidence_viz, cv2.COLORMAP_JET)
            cv2.imwrite(str(result_dir / "voting_confidence_map.png"), confidence_colored)
        
        # If we have weighted votes, create enhanced visualization
        if consensus_data and 'weighted_votes' in consensus_data and HAS_MATPLOTLIB:
            weighted_votes = consensus_data['weighted_votes']
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Enhanced Weighted Voting Results', fontsize=16)
            
            # Show weighted votes for each region
            im0 = axes[0, 0].imshow(weighted_votes[:, :, 0], cmap='Reds')
            axes[0, 0].set_title('Core Weighted Votes')
            plt.colorbar(im0, ax=axes[0, 0])
            
            im1 = axes[0, 1].imshow(weighted_votes[:, :, 1], cmap='Greens')
            axes[0, 1].set_title('Cladding Weighted Votes')
            plt.colorbar(im1, ax=axes[0, 1])
            
            im2 = axes[0, 2].imshow(weighted_votes[:, :, 2], cmap='Blues')
            axes[0, 2].set_title('Ferrule Weighted Votes')
            plt.colorbar(im2, ax=axes[0, 2])
            
            # Show confidence map
            if 'confidence_map' in consensus_data:
                im3 = axes[1, 0].imshow(consensus_data['confidence_map'], cmap='viridis', vmin=0, vmax=1)
                axes[1, 0].set_title('Voting Confidence Map')
                plt.colorbar(im3, ax=axes[1, 0])
            
            # Show final result
            axes[1, 1].imshow(mask_viz)
            axes[1, 1].set_title('Final Segmentation')
            
            # Show agreement statistics
            axes[1, 2].axis('off')
            if 'agreement_stats' in consensus_data:
                stats = consensus_data['agreement_stats']
                stats_text = f"Agreement Statistics:\n"
                stats_text += f"High confidence: {stats['high_confidence_pixels']:,} px\n"
                stats_text += f"Medium confidence: {stats['medium_confidence_pixels']:,} px\n"
                stats_text += f"Low confidence: {stats['low_confidence_pixels']:,} px\n"
                stats_text += f"Mean confidence: {stats['mean_confidence']:.3f}\n"
                stats_text += f"Uncertain resolved: {stats['uncertain_pixels_resolved']:,} px"
                axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                               fontsize=10, verticalalignment='center')
            
            for ax in axes.flat[:5]:
                ax.axis('off')
                
            plt.tight_layout()
            plt.savefig(str(result_dir / "enhanced_voting_analysis.png"), dpi=150)
            plt.close()
        
        print("  ✓ Created enhanced voting visualization")
    
    def update_learning(self, consensus: Dict[str, Any], image_shape: Tuple[int, int]):
        """Update learning parameters based on consensus results and agreement scores"""
        if not consensus:
            return
            
        # Update method scores based on agreement with consensus
        if 'method_agreement_scores' in consensus:
            for method, scores in consensus['method_agreement_scores'].items():
                current_score = self.dataset_stats['method_scores'].get(method, 1.0)
                
                # Adjust score based on agreement
                agreement = scores['raw_agreement']
                if agreement > 0.8:  # High agreement
                    new_score = current_score * 1.1
                elif agreement > 0.6:  # Good agreement
                    new_score = current_score * 1.05
                elif agreement < 0.4:  # Poor agreement
                    new_score = current_score * 0.9
                else:  # Moderate agreement
                    new_score = current_score  # No change
                
                # Clamp between 0.1 and 2.0
                self.dataset_stats['method_scores'][method] = max(0.1, min(2.0, new_score))
        else:
            # Fallback to old method if agreement scores not available
            for method in self.methods:
                if method in consensus['contributing_methods']:
                    current_score = self.dataset_stats['method_scores'].get(method, 1.0)
                    self.dataset_stats['method_scores'][method] = min(current_score * 1.05, 2.0)
                else:
                    current_score = self.dataset_stats['method_scores'].get(method, 1.0)
                    self.dataset_stats['method_scores'][method] = max(current_score * 0.95, 0.1)
        
        # Update method scores in memory
        for method in self.methods:
            self.methods[method]['score'] = self.dataset_stats['method_scores'].get(method, 1.0)
        
        # Track accuracy if available
        if 'method_agreement_scores' in consensus:
            for method, scores in consensus['method_agreement_scores'].items():
                self.dataset_stats['method_accuracy'][method] = scores['raw_agreement']
    
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
                # Print cleaned error message
                error_msg = result.error.split('\n')[0]  # First line only
                print(f"  ✗ Failed: {error_msg}")
        
        # Find consensus using enhanced voting
        consensus = self.pixel_voting_consensus(results, image_shape)
        
        if consensus:
            print(f"\n✓ Enhanced voting consensus achieved:")
            print(f"  Contributing methods: {', '.join(consensus['contributing_methods'])}")
            print(f"  Consensus strength - Core: {consensus['consensus_strength']['core']:.2f}, "
                  f"Cladding: {consensus['consensus_strength']['cladding']:.2f}, "
                  f"Ferrule: {consensus['consensus_strength']['ferrule']:.2f}")
            
            if 'agreement_stats' in consensus:
                print(f"  Mean confidence: {consensus['agreement_stats']['mean_confidence']:.3f}")
            
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
                accuracy = self.dataset_stats['method_accuracy'].get(method, 'N/A')
                if isinstance(accuracy, float):
                    print(f"  {method}: Score={info['score']:.3f}, Accuracy={accuracy:.3f}")
                else:
                    print(f"  {method}: Score={info['score']:.3f}, Accuracy={accuracy}")
                
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
        print("Enhanced Weighted Voting Edition".center(80))
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