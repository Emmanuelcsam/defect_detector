import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import Dict, Any, List, Optional, Tuple
import json
import os
import glob
import warnings
import time
from pathlib import Path
import shlex # Used for parsing file paths with spaces

# Suppress runtime warnings from calculations
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration ---
DEFAULT_CONFIG = {
    "center_finding": {
        "blur_ksize": (11, 11),
        "blur_sigma": 5,
        "brightness_percentile": 98,
        "hough_dp": 1.2,
        "hough_param1": 50,
        "hough_param2": 30,
        "weights": {"moments": 2.0, "hough": 1.5}
    },
    "radial_analysis": {
        "prominence_std_factor": 0.1,
        "smoothing_sigma": 2
    },
    "contour_analysis": {
        "blur_ksize": (11, 11),
        "min_area": 100,
        "min_circularity": 0.7,
        "max_center_offset": 50
    },
    "ransac_analysis": {
        "blur_ksize": (5, 5),
        "blur_sigma": 1.5,
        "canny_low": 50,
        "canny_high": 150,
        "min_edge_points": 50,
        "hist_bins": 100,
        "prominence_edge_factor": 0.01
    },
    "final_refinement": {
        "morph_ksize": (5, 5)
    },
    "priors": {
        "default_min_core_radius": 15,
        "default_min_cladding_thickness": 15,
        "default_cladding_radius_ratio": 0.8
    }
}

# --- 1. I/O and Utility Functions ---

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_image(path: Path) -> Optional[np.ndarray]:
    """Loads an image, handling both standard formats and our custom JSON format."""
    if path.suffix.lower() == '.json':
        return load_image_from_json(path)
    
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img

def load_image_from_json(path: Path) -> Optional[np.ndarray]:
    """Efficiently loads an image matrix from a JSON file."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)

        dims = data['image_dimensions']
        h, w = dims['height'], dims['width']
        c = dims.get('channels', 3)
        
        pixels = data['pixels']
        coords_y = np.zeros(len(pixels), dtype=np.int32)
        coords_x = np.zeros(len(pixels), dtype=np.int32)
        values = np.zeros((len(pixels), c), dtype=np.uint8)

        for i, p in enumerate(pixels):
            coords_y[i] = p['coordinates']['y']
            coords_x[i] = p['coordinates']['x']
            bgr = p.get('bgr_intensity', p.get('intensity', [0, 0, 0]))
            values[i] = bgr if isinstance(bgr, list) else [bgr] * c

        matrix = np.zeros((h, w, c), dtype=np.uint8)
        matrix[coords_y, coords_x] = values
        
        return matrix
    except (Exception, KeyError) as e:
        return None

# --- 2. Core Analysis Functions ---

def find_robust_center(gray_img: np.ndarray, config: Dict) -> Tuple[float, float]:
    """Finds the fiber center using a weighted average of two methods."""
    h, w = gray_img.shape
    centers, weights = [], []
    
    cfg = config['center_finding']
    smoothed = cv2.GaussianBlur(gray_img, cfg['blur_ksize'], cfg['blur_sigma'])
    _, bright_mask = cv2.threshold(smoothed, np.percentile(smoothed, cfg['brightness_percentile']), 255, cv2.THRESH_BINARY)
    moments = cv2.moments(bright_mask)
    if moments['m00'] > 0:
        centers.append((moments['m10'] / moments['m00'], moments['m01'] / moments['m00']))
        weights.append(cfg['weights']['moments'])
        
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray_img, (9, 9), 2), cv2.HOUGH_GRADIENT, 
        dp=cfg['hough_dp'], minDist=w // 2, param1=cfg['hough_param1'], 
        param2=cfg['hough_param2'], minRadius=0, maxRadius=0
    )
    if circles is not None:
        centers.append((circles[0, 0][0], circles[0, 0][1]))
        weights.append(cfg['weights']['hough'])

    if not centers:
        return w / 2, h / 2
        
    centers_arr = np.array(centers)
    return (
        np.average(centers_arr[:, 0], weights=weights),
        np.average(centers_arr[:, 1], weights=weights)
    )

def analyze_with_radial_profiles(
    gray_img: np.ndarray, center: Tuple[float, float], min_r: int, max_r: int, min_thick: int, config: Dict
) -> Optional[List[float]]:
    """Analyzes the 2nd derivative of the radial intensity profile to find boundaries."""
    h, w = gray_img.shape
    cx, cy = center
    max_radius = int(min(cx, cy, w - cx, h - cy)) - 1
    if max_radius <= 0: return None
    
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    r_int = r.round().astype(int)
    radial_sum = np.bincount(r_int.ravel(), weights=gray_img.ravel())
    counts = np.bincount(r_int.ravel())
    radial_mean = radial_sum / np.clip(counts, 1, None)
    radial_mean = radial_mean[:max_radius]

    smooth_intensity = gaussian_filter1d(radial_mean, sigma=config['radial_analysis']['smoothing_sigma'])
    gradient = np.gradient(smooth_intensity)
    second_deriv = np.gradient(gradient)
    
    peaks, props = find_peaks(-second_deriv, distance=min_thick, prominence=np.std(second_deriv) * config['radial_analysis']['prominence_std_factor'])
    valid_peaks = sorted([p for p in peaks if min_r < p < max_r])
    
    if len(valid_peaks) >= 2:
        prominences = props['prominences']
        sorted_by_prominence = sorted(zip(valid_peaks, prominences), key=lambda x: x[1], reverse=True)
        return sorted([p[0] for p in sorted_by_prominence[:2]])
    return None

def analyze_with_contours(
    gray_img: np.ndarray, center: Tuple[float, float], min_r: int, max_r: int, config: Dict
) -> Optional[List[float]]:
    """Analyzes contours from a thresholded image to find circular boundaries."""
    cfg = config['contour_analysis']
    blurred = cv2.GaussianBlur(gray_img, cfg['blur_ksize'], 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg['min_area']: continue
        
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        
        circularity = 4 * np.pi * (area / (perimeter**2))
        center_offset = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        if (circularity > cfg['min_circularity'] and 
            center_offset < cfg['max_center_offset'] and 
            min_r < radius < max_r):
            circles.append(radius)
            
    if len(circles) >= 2:
        return [sorted(circles)[0], sorted(circles)[-1]]
    return None

def analyze_with_ransac_histogram(
    gray_img: np.ndarray, center: Tuple[float, float], min_r: int, max_r: int, min_thick: int, config: Dict
) -> Optional[List[float]]:
    """Uses a histogram of edge point distances (RANSAC-like) to find boundaries."""
    cfg = config['ransac_analysis']
    edges = cv2.Canny(cv2.GaussianBlur(gray_img, cfg['blur_ksize'], cfg['blur_sigma']), cfg['canny_low'], cfg['canny_high'])
    edge_points = np.column_stack(np.where(edges.T > 0))
    
    if len(edge_points) < cfg['min_edge_points']: return None

    dists = np.linalg.norm(edge_points - center, axis=1)
    hist, bins = np.histogram(dists, bins=cfg['hist_bins'], range=(0, max_r * 1.2))
    bin_width = bins[1] - bins[0]
    
    peaks, _ = find_peaks(hist, distance=max(1, int(min_thick / bin_width)), prominence=len(edge_points) * cfg['prominence_edge_factor'])
    
    if len(peaks) >= 2:
        radii = sorted([bins[p] + bin_width / 2 for p in np.argsort(hist[peaks])[-2:]])
        return radii
    return None

# --- 3. Main Pipeline Orchestration ---

def get_priors_from_dataset(dataset_path: Path) -> Dict[str, float]:
    """Loads segmentation priors from a directory of JSON reports."""
    json_files = list(dataset_path.glob("*_seg_report.json"))
    if not json_files:
        return {}
        
    radii_ratios, thicknesses = [], []
    for f_path in json_files:
        with open(f_path, 'r') as f:
            data = json.load(f)
        b, info = data.get('consensus_boundaries'), data.get('image_info')
        if b and info and len(b) == 2:
            radii_ratios.append(b[1] / info['width'])
            thicknesses.append(b[1] - b[0])
            
    if not radii_ratios: return {}
    
    priors = {
        'avg_cladding_radius_ratio': np.median(radii_ratios),
        'avg_min_cladding_thickness': np.percentile(thicknesses, 25),
        'avg_min_core_radius': np.percentile(thicknesses, 25)
    }
    return priors

def run_segmentation_pipeline(
    image_path: Path, 
    priors: Dict[str, float],
    config: Dict[str, Any] = DEFAULT_CONFIG,
    output_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Executes the full fiber segmentation pipeline - modified for unified system."""
    
    # Initialize result dictionary
    result = {
        'method': 'segmentation',
        'image_path': str(image_path),
        'success': False,
        'center': None,
        'core_radius': None,
        'cladding_radius': None,
        'confidence': 0.0
    }

    original_img = load_image(image_path)
    if original_img is None: 
        result['error'] = f"Could not load image: {image_path}"
        if output_dir:
            with open(output_dir / f'{image_path.stem}_segmentation_result.json', 'w') as f:
                json.dump(result, f, indent=4, cls=NumpyEncoder)
        return result
    
    h, w = original_img.shape[:2]
    is_color = len(original_img.shape) == 3
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if is_color else original_img.copy()

    prior_cfg = config['priors']
    min_core_r = int(priors.get('avg_min_core_radius', prior_cfg['default_min_core_radius']))
    min_clad_thick = int(priors.get('avg_min_cladding_thickness', prior_cfg['default_min_cladding_thickness']))
    avg_clad_ratio = priors.get('avg_cladding_radius_ratio', prior_cfg['default_cladding_radius_ratio'])
    max_clad_r = w * 0.5 * avg_clad_ratio

    center = find_robust_center(gray_img, config)

    all_boundaries = []
    analyses = {
        "Radial": analyze_with_radial_profiles(gray_img, center, min_core_r, max_clad_r, min_clad_thick, config),
        "Contours": analyze_with_contours(gray_img, center, min_core_r, max_clad_r, config),
        "RANSAC": analyze_with_ransac_histogram(gray_img, center, min_core_r, max_clad_r, min_clad_thick, config)
    }
    
    for name, res in analyses.items():
        if res:
            all_boundaries.append(res)
    
    if len(all_boundaries) < 2:
        result['error'] = "Not enough evidence to form a consensus"
        if output_dir:
            with open(output_dir / f'{image_path.stem}_segmentation_result.json', 'w') as f:
                json.dump(result, f, indent=4, cls=NumpyEncoder)
        return result

    inner_radii = [b[0] for b in all_boundaries]
    outer_radii = [b[1] for b in all_boundaries]
    inner_boundary = int(np.median(inner_radii))
    outer_boundary = int(np.median(outer_radii))
    
    if inner_boundary >= outer_boundary:
        result['error'] = "Consensus boundaries are invalid (inner >= outer)"
        if output_dir:
            with open(output_dir / f'{image_path.stem}_segmentation_result.json', 'w') as f:
                json.dump(result, f, indent=4, cls=NumpyEncoder)
        return result
    
    # Set successful results
    result['success'] = True
    result['center'] = (int(center[0]), int(center[1]))
    result['core_radius'] = inner_boundary
    result['cladding_radius'] = outer_boundary
    result['confidence'] = 0.9  # High confidence for consensus method
    
    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        base_filename = image_path.stem
        
        # Save result data
        with open(output_dir / f'{base_filename}_segmentation_result.json', 'w') as f:
            json.dump(result, f, indent=4, cls=NumpyEncoder)
            
        # Create and save visualization
        Y, X = np.ogrid[:h, :w]
        dist_map = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        masks = {
            'core': (dist_map <= inner_boundary).astype(np.uint8),
            'cladding': ((dist_map > inner_boundary) & (dist_map <= outer_boundary)).astype(np.uint8),
            'ferrule': (dist_map > outer_boundary).astype(np.uint8)
        }

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['final_refinement']['morph_ksize'])
        for name in masks:
            masks[name] = cv2.morphologyEx(masks[name], cv2.MORPH_CLOSE, kernel)
            masks[name] = cv2.morphologyEx(masks[name], cv2.MORPH_OPEN, kernel)
            masks[name] *= 255
            
        regions = {name: cv2.bitwise_and(original_img, original_img, mask=mask) for name, mask in masks.items()}
        
        # Save regions
        cv2.imwrite(str(output_dir / f"{base_filename}_segmentation_core.png"), regions['core'])
        cv2.imwrite(str(output_dir / f"{base_filename}_segmentation_cladding.png"), regions['cladding'])
        cv2.imwrite(str(output_dir / f"{base_filename}_segmentation_ferrule.png"), regions['ferrule'])
        
        # Create annotated image
        annotated = original_img.copy()
        cv2.circle(annotated, (int(center[0]), int(center[1])), 3, (0, 255, 255), -1)
        cv2.circle(annotated, (int(center[0]), int(center[1])), inner_boundary, (0, 255, 0), 2)
        cv2.circle(annotated, (int(center[0]), int(center[1])), outer_boundary, (0, 0, 255), 2)
        cv2.imwrite(str(output_dir / f"{base_filename}_segmentation_annotated.png"), annotated)
    
    # Return full pipeline result for compatibility
    return {
        'image_info': {'path': str(image_path), 'width': w, 'height': h},
        'original_image': original_img,
        'final_center': {'x': center[0], 'y': center[1]},
        'consensus_boundaries': [inner_boundary, outer_boundary],
        'result': result  # Include standardized result
    }

def generate_segmentation_report(results: Dict, output_prefix: Path):
    """Saves output images and a summary visualization."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    for name, image in results['regions'].items():
        cv2.imwrite(f"{output_prefix}_{name}.png", image)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Segmentation Analysis: {Path(results['image_info']['path']).name}", fontsize=16)
    
    orig_rgb = cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original with Boundaries")
    theta = np.linspace(0, 2 * np.pi, 100)
    cx, cy = results['final_center']['x'], results['final_center']['y']
    axes[0].plot(cx + results['consensus_boundaries'][0] * np.cos(theta), cy + results['consensus_boundaries'][0] * np.sin(theta), 'lime')
    axes[0].plot(cx + results['consensus_boundaries'][1] * np.cos(theta), cy + results['consensus_boundaries'][1] * np.sin(theta), 'cyan')
    
    mask_overlay = np.zeros_like(orig_rgb)
    mask_overlay[results['masks']['core'] > 0] = [255, 0, 0]
    mask_overlay[results['masks']['cladding'] > 0] = [0, 255, 0]
    axes[1].imshow(mask_overlay)
    axes[1].set_title("Region Masks")

    composite = sum(results['regions'].values())
    axes[2].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Segmented Regions")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    report_path = f"{output_prefix}_segmentation_summary.png"
    plt.savefig(report_path)
    plt.close()

# Simplified main function for standalone testing
def main():
    """Main function for standalone testing"""
    import sys
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = Path(input("Enter image path: ").strip().strip('"').strip("'"))
    
    priors = {}  # Use defaults
    result = run_segmentation_pipeline(image_path, priors, output_dir=Path("output_segmentation"))
    
    if result and result.get('result', {}).get('success'):
        r = result['result']
        print(f"Success! Center: {r['center']}, Core: {r['core_radius']}, Cladding: {r['cladding_radius']}")
    else:
        print(f"Failed: {result.get('result', {}).get('error', 'Unknown error')}")


if __name__ == '__main__':
    main()