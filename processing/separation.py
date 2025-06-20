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

# Import scipy components for enhanced logic
try:
    from scipy.ndimage import median_filter, gaussian_filter
    from scipy.ndimage import binary_opening, binary_closing
    HAS_SCIPY_FULL = True
except ImportError:
    HAS_SCIPY_FULL = False
    print("Warning: Some scipy components not available, using basic post-processing")

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

class EnhancedConsensusSystem:
    """
    Advanced consensus system that shifts from simple pixel voting to a
    model-driven, geometrically-aware consensus.
    """
    def __init__(self, min_agreement_ratio=0.3):
        self.min_agreement_ratio = min_agreement_ratio

    def _calculate_iou(self, mask1, mask2):
        """Calculates Intersection over Union for two binary masks."""
        if mask1 is None or mask2 is None:
            return 0.0
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou_score = np.sum(intersection) / (np.sum(union) + 1e-6) # Add epsilon to avoid division by zero
        return iou_score

    def generate_consensus(self,
                           results: List['SegmentationResult'],
                           method_scores: Dict[str, float],
                           image_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Generates a final consensus model by first running a preliminary pixel
        vote to identify high-agreement methods, then calculating a weighted
        average of their geometric parameters to create an ideal final mask.
        """
        valid_results = [r for r in results if r.error is None and r.masks is not None]
        
        if len(valid_results) < 2:
            print("! Not enough valid results to form a consensus.")
            return None

        print(f"\nGenerating consensus from {len(valid_results)} valid results...")
        h, w = image_shape

        # --- Stage 1: Preliminary Weighted Pixel Vote ---
        weighted_votes = np.zeros((h, w, 3), dtype=np.float32)
        for r in valid_results:
            weight = method_scores.get(r.method_name, 1.0) * r.confidence
            if r.masks.get('core') is not None:
                weighted_votes[:, :, 0] += (r.masks['core'] > 0).astype(np.float32) * weight
            if r.masks.get('cladding') is not None:
                weighted_votes[:, :, 1] += (r.masks['cladding'] > 0).astype(np.float32) * weight
            if r.masks.get('ferrule') is not None:
                weighted_votes[:, :, 2] += (r.masks['ferrule'] > 0).astype(np.float32) * weight

        preliminary_classification = np.argmax(weighted_votes, axis=2)
        prelim_core_mask = (preliminary_classification == 0)
        prelim_cladding_mask = (preliminary_classification == 1)

        # --- Stage 2: Identify High-Agreement Methods ---
        high_agreement_results = []
        for r in valid_results:
            core_iou = self._calculate_iou(r.masks['core'], prelim_core_mask)
            cladding_iou = self._calculate_iou(r.masks['cladding'], prelim_cladding_mask)
            if core_iou > 0.6 and cladding_iou > 0.6:
                high_agreement_results.append(r)

        if not high_agreement_results:
            print("! No methods passed the high-agreement threshold. Using all valid results for fallback.")
            high_agreement_results = valid_results
        
        print(f"  Found {len(high_agreement_results)} methods for parameter averaging.")

        # --- Stage 3: Parameter-Space Consensus ---
        consensus_params = {'cx': [], 'cy': [], 'core_r': [], 'clad_r': []}
        weights = []
        for r in high_agreement_results:
            weight = method_scores.get(r.method_name, 1.0) * r.confidence
            if r.center and r.core_radius is not None and r.cladding_radius is not None:
                consensus_params['cx'].append(r.center[0])
                consensus_params['cy'].append(r.center[1])
                consensus_params['core_r'].append(r.core_radius)
                consensus_params['clad_r'].append(r.cladding_radius)
                weights.append(weight)

        if not weights:
             print("! No valid parameters to average. Consensus failed.")
             return None

        final_center = (
            np.average(consensus_params['cx'], weights=weights),
            np.average(consensus_params['cy'], weights=weights)
        )
        final_core_radius = np.average(consensus_params['core_r'], weights=weights)
        final_cladding_radius = np.average(consensus_params['clad_r'], weights=weights)

        # --- Stage 4: Generate Final Ideal Masks ---
        final_masks = self.create_masks_from_params(
            final_center, final_core_radius, final_cladding_radius, image_shape
        )
        
        final_masks['core'], final_masks['cladding'], final_masks['ferrule'] = self.ensure_mask_consistency(
             final_masks['core'], final_masks['cladding'], final_masks['ferrule']
        )
        
        return {
            'masks': final_masks,
            'center': final_center,
            'core_radius': final_core_radius,
            'cladding_radius': final_cladding_radius,
            'contributing_methods': [r.method_name for r in high_agreement_results],
            'num_valid_results': len(valid_results),
            'all_results': [r.to_dict() for r in results]
        }

    def create_masks_from_params(self, center: Tuple[float, float], core_radius: float, 
                               cladding_radius: float, image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Creates binary masks from geometric parameters."""
        h, w = image_shape
        cx, cy = center
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        core_mask = (dist_from_center <= core_radius).astype(np.uint8)
        cladding_mask = ((dist_from_center > core_radius) & 
                        (dist_from_center <= cladding_radius)).astype(np.uint8)
        ferrule_mask = (dist_from_center > cladding_radius).astype(np.uint8)
        
        return {'core': core_mask, 'cladding': cladding_mask, 'ferrule': ferrule_mask}

    def ensure_mask_consistency(self, core_mask, cladding_mask, ferrule_mask):
        """Ensure masks are mutually exclusive and spatially clean."""
        if not HAS_SCIPY_FULL:
            return core_mask, cladding_mask, ferrule_mask
            
        kernel = np.ones((5, 5), dtype=np.uint8)
        
        core_mask = binary_closing(binary_opening(core_mask, kernel), kernel).astype(np.uint8)
        cladding_mask = binary_closing(binary_opening(cladding_mask, kernel), kernel).astype(np.uint8)
        
        cladding_mask[core_mask == 1] = 0
        ferrule_mask[core_mask == 1] = 0
        ferrule_mask[cladding_mask == 1] = 0
        
        return core_mask, cladding_mask, ferrule_mask

class SegmentationResult:
    """Standardized result format for all segmentation methods"""
    def __init__(self, method_name: str, image_path: str):
        self.method_name = method_name
        self.image_path = image_path
        self.center = None
        self.core_radius = None
        self.cladding_radius = None
        self.masks = None
        self.confidence = 0.5
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
        
        self.dataset_stats = {'method_scores': {}, 'method_accuracy': {}}
        
        self.knowledge_file = self.output_dir / "segmentation_knowledge.json"
        self.load_knowledge()
        
        self.methods = {}
        self.load_methods()
        
        self.consensus_system = EnhancedConsensusSystem()
        
        self.vulnerable_methods = [
            'adaptive_intensity', 'gradient_approach', 'guess_approach', 'threshold_separation'
        ]

    def load_knowledge(self):
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r') as f:
                    self.dataset_stats.update(json.load(f))
                    print(f"✓ Loaded knowledge from {self.knowledge_file}")
            except Exception as e:
                print(f"! Could not load knowledge ({e}), starting fresh")
    
    def save_knowledge(self):
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.dataset_stats, f, indent=4, cls=NumpyEncoder)
        print(f"✓ Saved updated knowledge to {self.knowledge_file}")
    
    def load_methods(self):
        method_files = [
            'adaptive_intensity.py', 'bright_core_extractor.py', 'computational_separation.py',
            'geometric_approach.py', 'gradient_approach.py', 'guess_approach.py',
            'hough_separation.py', 'segmentation.py', 'threshold_separation.py',
            'unified_core_cladding_detector.py'
        ]
        
        for method_file in method_files:
            method_name = Path(method_file).stem
            method_path = self.methods_dir / method_file
            if method_path.exists():
                self.methods[method_name] = {
                    'path': method_path,
                    'score': self.dataset_stats['method_scores'].get(method_name, 1.0)
                }
                print(f"✓ Loaded method: {method_name} (score: {self.methods[method_name]['score']:.2f})")

    def detect_and_inpaint_anomalies(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        _, defect_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
        if HAS_SCIPY_FULL:
            defect_mask = binary_opening(defect_mask, structure=np.ones((3,3)), iterations=2).astype(np.uint8)
        inpainted_image = cv2.inpaint(image, defect_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return inpainted_image, defect_mask

    def run_method_isolated(self, method_name: str, image_path: Path, temp_output: Path) -> dict:
        """Generates a wrapper script to run a method in isolation and captures its JSON output."""
        result_file = temp_output / f"{method_name}_result.json"
        runner_script_path = temp_output / "runner.py"
        
        script_content = f"""
import sys, json, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, r"{self.methods_dir.resolve()}")

def main():
    image_path_str = r"{image_path.resolve()}"
    output_dir_str = r"{temp_output.resolve()}"
    result = {{'success': False, 'error': 'Unknown execution error'}}
    try:
"""
        method_map = {
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
        }
        
        call_logic = method_map.get(method_name)
        if not call_logic:
             return {'success': False, 'error': f'Runner for method {method_name} not implemented.'}
        script_content += f"        {call_logic}\n"

        script_content += f"""
    except Exception as e:
        import traceback
        result['error'] = f"Exception in {{method_name}}: {{e}}\\n{{traceback.format_exc()}}"
    
    with open(r"{result_file.resolve()}", 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
"""
        with open(runner_script_path, 'w') as f: f.write(script_content)

        try:
            subprocess.run(
                [sys.executable, str(runner_script_path)],
                capture_output=True, text=True, timeout=120, check=False,
                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen', 'MPLBACKEND': 'Agg'}
            )
            if result_file.exists():
                with open(result_file, 'r') as f: return json.load(f)
            return {'success': False, 'error': 'No result file produced.'}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Method timed out'}
        except Exception as e:
            return {'success': False, 'error': f'Subprocess execution failed: {e}'}

    def run_method(self, method_name: str, image_path: Path, image_shape: Tuple[int, int]) -> SegmentationResult:
        result = SegmentationResult(method_name, str(image_path))
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            method_output = self.run_method_isolated(method_name, image_path, Path(temp_dir))
            
            if method_output and method_output.get('success'):
                result.center = tuple(method_output.get('center')) if method_output.get('center') else None
                result.core_radius = method_output.get('core_radius')
                result.cladding_radius = method_output.get('cladding_radius')
                result.confidence = method_output.get('confidence', 0.5)

                if all([result.center, result.core_radius, result.cladding_radius]):
                    result.masks = self.consensus_system.create_masks_from_params(
                        result.center, result.core_radius, result.cladding_radius, image_shape
                    )
                    if result.masks and result.masks.get('core') is not None:
                        contours, _ = cv2.findContours(result.masks['core'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cnt = max(contours, key=cv2.contourArea)
                            area = cv2.contourArea(cnt)
                            perimeter = cv2.arcLength(cnt, True)
                            if perimeter > 0:
                                circularity = (4 * np.pi * area) / (perimeter**2)
                                if circularity < 0.85:
                                    result.confidence *= 0.5
                                    print(f"  ! Penalizing {method_name} for low circularity ({circularity:.2f})")
                else:
                    result.error = "Method returned invalid/missing parameters."
            else:
                result.error = method_output.get('error', 'Unknown failure') if method_output else "Empty method output"
        
        result.execution_time = time.time() - start_time
        return result

    def update_learning(self, consensus: Dict, all_results: List[SegmentationResult]):
        print("\nUpdating learning model...")
        consensus_masks = consensus['masks']
        
        for result in all_results:
            if result.error or not result.masks: continue
            core_iou = self.consensus_system._calculate_iou(result.masks.get('core'), consensus_masks.get('core'))
            cladding_iou = self.consensus_system._calculate_iou(result.masks.get('cladding'), consensus_masks.get('cladding'))
            avg_iou = (core_iou + cladding_iou) / 2
            
            current_score = self.dataset_stats['method_scores'].get(result.method_name, 1.0)
            learning_rate = 0.1
            target_score = 0.1 + (1.9 * avg_iou)
            new_score = current_score * (1 - learning_rate) + target_score * learning_rate
            
            self.dataset_stats['method_scores'][result.method_name] = new_score
            self.dataset_stats['method_accuracy'][result.method_name] = avg_iou
            self.methods[result.method_name]['score'] = new_score
        
        print("  ✓ Method scores updated.")
        self.save_knowledge()
    
    def process_image(self, image_path: Path, output_dir_str: str) -> Optional[Dict]:
        print(f"\n{'='*25} Processing: {image_path.name} {'='*25}")
        original_img = cv2.imread(str(image_path))
        if original_img is None: return None
        image_shape = original_img.shape[:2]
        
        print("\nRunning pre-processing: Anomaly detection and inpainting...")
        inpainted_img, defect_mask = self.detect_and_inpaint_anomalies(original_img)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
            inpainted_image_path = Path(tmp_f.name)
            cv2.imwrite(str(inpainted_image_path), inpainted_img)
        print("  ✓ Inpainting complete.")

        all_results = []
        for method_name in self.methods:
            use_inpainted = method_name in self.vulnerable_methods
            current_image_path = inpainted_image_path if use_inpainted else image_path
            
            print(f"\nRunning {method_name} (using {'inpainted' if use_inpainted else 'original'} image)...")
            result = self.run_method(method_name, current_image_path, image_shape)
            all_results.append(result)
            
            if result.error: print(f"  ✗ Failed: {result.error}")
            else: print(f"  ✓ Success - Confidence: {result.confidence:.2f}, Time: {result.execution_time:.2f}s")
        
        consensus = self.consensus_system.generate_consensus(
            all_results, {name: info['score'] for name, info in self.methods.items()}, image_shape
        )
        
        if consensus:
            print("\n✓ Model-driven consensus achieved.")
            print(f"  Contributing methods: {', '.join(consensus['contributing_methods'])}")
            self.update_learning(consensus, all_results)
            self.save_results(image_path, consensus, original_img, output_dir_str, defect_mask)
        else:
            print("\n✗ FINAL: No consensus could be reached.")
            
        os.remove(inpainted_image_path)
        return consensus

    def save_results(self, image_path: Path, consensus: Dict, image: np.ndarray, output_dir: str, defect_mask: np.ndarray):
        """Saves consensus results, separated region images, and visualizations."""
        result_dir = Path(output_dir)
        result_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        report = {k: v for k, v in consensus.items() if k not in ['masks', 'all_results']}
        report['method_accuracies'] = self.dataset_stats.get('method_accuracy', {})
        with open(result_dir / "consensus_report.json", 'w') as f:
            json.dump(report, f, indent=4, cls=NumpyEncoder)
        
        # Get final masks
        masks = consensus['masks']
        
        # Save the raw mask images
        cv2.imwrite(str(result_dir / "mask_core.png"), masks['core'] * 255)
        cv2.imwrite(str(result_dir / "mask_cladding.png"), masks['cladding'] * 255)
        cv2.imwrite(str(result_dir / "mask_ferrule.png"), masks['ferrule'] * 255)
        cv2.imwrite(str(result_dir / "detected_defects.png"), defect_mask)
        
        # Apply masks to the original image to get separated regions
        print("  Applying final masks to create separated region images...")
        region_core = cv2.bitwise_and(image, image, mask=masks['core'])
        region_cladding = cv2.bitwise_and(image, image, mask=masks['cladding'])
        region_ferrule = cv2.bitwise_and(image, image, mask=masks['ferrule'])

        # Save the separated region images
        cv2.imwrite(str(result_dir / "region_core.png"), region_core)
        cv2.imwrite(str(result_dir / "region_cladding.png"), region_cladding)
        cv2.imwrite(str(result_dir / "region_ferrule.png"), region_ferrule)

        # Pass the regions to the visualization function
        regions = {'core': region_core, 'cladding': region_cladding, 'ferrule': region_ferrule}
        if HAS_MATPLOTLIB:
            self.create_summary_visualization(result_dir, image, masks, defect_mask, consensus, regions)
        
        print(f"\n✓ All results, masks, and region images saved to: {result_dir}")
    
    def create_summary_visualization(self, result_dir: Path, original_image: np.ndarray, masks: Dict, defect_mask: np.ndarray, consensus: Dict, regions: Dict):
        """Creates a comprehensive summary plot including the separated regions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
        fig.suptitle(f'Unified Segmentation Analysis: {result_dir.name}', fontsize=16)
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Plot 1: Original Image with Final Boundaries
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original with Final Boundaries')
        theta = np.linspace(0, 2 * np.pi, 100)
        cx, cy = consensus['center']
        axes[0, 0].plot(cx + consensus['core_radius'] * np.cos(theta), cy + consensus['core_radius'] * np.sin(theta), 'lime', linewidth=2)
        axes[0, 0].plot(cx + consensus['cladding_radius'] * np.cos(theta), cy + consensus['cladding_radius'] * np.sin(theta), 'cyan', linewidth=2)

        # Plot 2: Final Segmentation Mask
        final_mask_viz = np.zeros_like(img_rgb)
        final_mask_viz[masks['core'] > 0] = [255, 0, 0]
        final_mask_viz[masks['cladding'] > 0] = [0, 255, 0]
        final_mask_viz[masks['ferrule'] > 0] = [0, 0, 255]
        axes[0, 1].imshow(final_mask_viz)
        axes[0, 1].set_title('Final Segmentation Masks')
        
        # Plot 3: Method Performance Text
        axes[1, 0].axis('off')
        text_content = "Method Performance (IoU):\n" + "-"*25
        accuracies = self.dataset_stats.get('method_accuracy', {})
        sorted_methods = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
        for method, acc in sorted_methods:
            text_content += f"\n{method[:20].ljust(22)}: {acc:.3f}"
        axes[1, 0].text(0.05, 0.95, text_content, family='monospace', verticalalignment='top', fontsize=10)

        # Plot 4: Separated Regions Combined
        composite_image = cv2.cvtColor(sum(regions.values()), cv2.COLOR_BGR2RGB)
        axes[1, 1].imshow(composite_image)
        axes[1, 1].set_title('Final Separated Regions')

        for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
        plt.savefig(result_dir / "summary_analysis.png", dpi=150)
        plt.close()

    def run(self):
        print("\n" + "="*80)
        print("UNIFIED FIBER OPTIC SEGMENTATION SYSTEM".center(80))
        print("Model-Driven Consensus Edition".center(80))
        print("="*80)
        if not self.methods:
            print(f"✗ Error: No methods found in '{self.methods_dir}'.")
            return
        folder_path = Path(input("\nEnter the folder path containing images: ").strip().strip('"\''))
        if not folder_path.is_dir():
            print(f"✗ Folder not found: {folder_path}")
            return
        image_files = list(folder_path.glob('*.png')) + list(folder_path.glob('*.jpg'))
        print(f"Found {len(image_files)} images to process.")
        for img_path in image_files:
            self.process_image(img_path, str(self.output_dir / img_path.stem))
        print("\n" + "="*80 + "\n" + "Processing complete.".center(80) + "\n" + "="*80)

def main():
    methods_dir = sys.argv[1] if len(sys.argv) > 1 else "zones_methods"
    system = UnifiedSegmentationSystem(methods_dir)
    system.run()

if __name__ == "__main__":
    main()
