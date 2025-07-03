import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from datetime import datetime
from collections import defaultdict
import shutil
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class DefectAggregator:
    """Aggregates and analyzes defects from multiple detection results with improved data handling"""
    
    def __init__(self, results_dir: Path, original_image_path: Path, 
                 clustering_eps: float = 30.0, min_cluster_size: int = 1):
        self.results_dir = Path(results_dir)
        self.original_image_path = Path(original_image_path)
        
        # Validate inputs
        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {self.results_dir}")
        if not self.original_image_path.exists():
            raise ValueError(f"Original image does not exist: {self.original_image_path}")
            
        self.original_image = cv2.imread(str(self.original_image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load original image: {self.original_image_path}")
            
        self.height, self.width = self.original_image.shape[:2]
        self.clustering_eps = clustering_eps
        self.min_cluster_size = min_cluster_size
        
        # Data storage
        self.all_defects = []
        self.region_masks = {}
        self.detection_results = []
        self.region_offsets = {}  # Store region bounding boxes for better mapping
        self.data_integrity_log = []
        
        self.logger = logging.getLogger(__name__)
        
    def validate_detection_report(self, report: Dict, file_path: Path) -> bool:
        """Validate the structure and content of a detection report"""
        required_fields = ['defects', 'timestamp']
        
        for field in required_fields:
            if field not in report:
                self.data_integrity_log.append({
                    'file': str(file_path),
                    'issue': f'Missing required field: {field}',
                    'severity': 'WARNING'
                })
                return False
                
        # Validate defect structure
        for i, defect in enumerate(report.get('defects', [])):
            if 'location_xy' not in defect or not isinstance(defect['location_xy'], (list, tuple)):
                self.data_integrity_log.append({
                    'file': str(file_path),
                    'issue': f'Invalid location_xy in defect {i}',
                    'severity': 'ERROR'
                })
                return False
                
        return True
        
    def load_all_detection_results(self):
        """Load all detection results with validation and error tracking"""
        self.logger.info(f"Loading detection results from: {self.results_dir}")
        
        # Check for expected directory structure
        detection_dir = self.results_dir / "3_detected"
        if not detection_dir.exists():
            # Try alternative structure
            detection_dir = self.results_dir
            self.logger.warning(f"No '3_detected' subdirectory found, using: {detection_dir}")
            
        # Find all JSON report files
        report_files = list(detection_dir.rglob("*_report.json"))
        self.logger.info(f"Found {len(report_files)} detection reports")
        
        successful_loads = 0
        failed_loads = 0
        
        for report_file in report_files:
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    
                # Validate report structure
                if not self.validate_detection_report(report, report_file):
                    failed_loads += 1
                    continue
                    
                # Extract source information
                source_name = report_file.stem.replace('_report', '')
                
                # Determine region type more reliably
                region_type = self._determine_region_type(source_name, report)
                is_region = region_type is not None
                
                # Load associated mask if exists
                mask = None
                mask_path = report_file.parent / f"{source_name}_mask.npy"
                if mask_path.exists():
                    try:
                        mask = np.load(mask_path)
                        # Validate mask dimensions
                        if mask.shape != (self.height, self.width):
                            self.logger.warning(f"Mask dimensions mismatch for {source_name}")
                            mask = None
                    except Exception as e:
                        self.logger.error(f"Error loading mask {mask_path}: {str(e)}")
                
                # Store detection result
                detection_data = {
                    'report': report,
                    'source_name': source_name,
                    'is_region': is_region,
                    'region_type': region_type,
                    'mask': mask,
                    'file_path': report_file,
                    'defect_count': len(report.get('defects', []))
                }
                self.detection_results.append(detection_data)
                
                # Process defects with ID tracking
                base_defect_id = hashlib.md5(source_name.encode()).hexdigest()[:8]
                
                for idx, defect in enumerate(report.get('defects', [])):
                    # Create unique defect ID
                    defect['unique_id'] = f"{base_defect_id}_{idx:04d}"
                    defect['source_image'] = source_name
                    defect['is_region'] = is_region
                    defect['region_type'] = region_type
                    defect['source_file'] = str(report_file)
                    
                    # Validate defect data
                    if self._validate_defect_data(defect):
                        self.all_defects.append(defect)
                    else:
                        self.data_integrity_log.append({
                            'file': str(report_file),
                            'issue': f'Invalid defect data at index {idx}',
                            'severity': 'WARNING'
                        })
                        
                successful_loads += 1
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error in {report_file}: {str(e)}")
                failed_loads += 1
            except Exception as e:
                self.logger.error(f"Error loading {report_file}: {str(e)}")
                failed_loads += 1
                
        self.logger.info(f"Successfully loaded {successful_loads} reports, {failed_loads} failed")
        self.logger.info(f"Total defects loaded: {len(self.all_defects)}")
        
    def _determine_region_type(self, source_name: str, report: Dict) -> Optional[str]:
        """Determine region type from source name and report metadata"""
        # Check source name
        for zone in ['core', 'cladding', 'ferrule']:
            if f'_{zone}' in source_name.lower():
                return zone
                
        # Check report metadata
        if 'region_type' in report:
            return report['region_type'].lower()
            
        if 'metadata' in report and 'region' in report['metadata']:
            return report['metadata']['region'].lower()
            
        return None
        
    def _validate_defect_data(self, defect: Dict) -> bool:
        """Validate individual defect data"""
        required_fields = ['location_xy']
        
        for field in required_fields:
            if field not in defect:
                return False
                
        # Validate location
        loc = defect['location_xy']
        if not isinstance(loc, (list, tuple)) or len(loc) != 2:
            return False
            
        # Ensure numeric coordinates
        try:
            x, y = float(loc[0]), float(loc[1])
            defect['location_xy'] = (int(x), int(y))
        except (ValueError, TypeError):
            return False
            
        return True
        
    def load_separation_masks(self):
        """Load separation masks and calculate region offsets for accurate mapping"""
        separation_dir = self.results_dir / "2_separated" / self.original_image_path.stem
        
        if not separation_dir.exists():
            # Try alternative paths
            alt_paths = [
                self.results_dir / "2_separated",
                self.results_dir / "separated",
                self.results_dir
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    separation_dir = alt_path
                    break
                    
        self.logger.info(f"Loading separation masks from: {separation_dir}")
        
        for mask_type in ['core', 'cladding', 'ferrule']:
            mask_file = separation_dir / f"{mask_type}_mask.npy"
            if mask_file.exists():
                try:
                    mask = np.load(mask_file)
                    
                    # Validate mask
                    if mask.shape != (self.height, self.width):
                        self.logger.warning(f"Mask shape mismatch for {mask_type}: "
                                          f"{mask.shape} vs expected {(self.height, self.width)}")
                        continue
                        
                    self.region_masks[mask_type] = mask
                    
                    # Calculate region bounding box for offset mapping
                    coords = np.where(mask > 0)
                    if len(coords[0]) > 0:
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        self.region_offsets[mask_type] = {
                            'x_offset': x_min,
                            'y_offset': y_min,
                            'width': x_max - x_min + 1,
                            'height': y_max - y_min + 1
                        }
                        self.logger.info(f"Loaded {mask_type} mask with offset "
                                       f"({x_min}, {y_min})")
                        
                except Exception as e:
                    self.logger.error(f"Error loading {mask_type} mask: {str(e)}")
                    
    def map_defect_to_global_coords(self, defect: Dict) -> Optional[Tuple[int, int]]:
        """Map defect coordinates to global image coordinates with improved accuracy"""
        if not defect.get('is_region'):
            # Already in global coordinates
            loc = defect.get('location_xy', (0, 0))
            return (int(loc[0]), int(loc[1]))
            
        region_type = defect.get('region_type')
        if not region_type or region_type not in self.region_offsets:
            # Fallback to original coordinates
            loc = defect.get('location_xy', (0, 0))
            return (int(loc[0]), int(loc[1]))
            
        # Use pre-calculated offsets for accurate mapping
        offset_info = self.region_offsets[region_type]
        local_x, local_y = defect.get('location_xy', (0, 0))
        
        # Map to global coordinates
        global_x = offset_info['x_offset'] + local_x
        global_y = offset_info['y_offset'] + local_y
        
        # Validate bounds
        global_x = max(0, min(global_x, self.width - 1))
        global_y = max(0, min(global_y, self.height - 1))
        
        return (int(global_x), int(global_y))
        
    def cluster_defects(self, custom_eps: Optional[float] = None) -> List[Dict]:
        """Cluster defects with adaptive parameters based on defect density"""
        if not self.all_defects:
            return []
            
        eps = custom_eps if custom_eps is not None else self.clustering_eps
        
        # Extract and validate coordinates
        coords = []
        valid_defects = []
        
        for defect in self.all_defects:
            global_coord = self.map_defect_to_global_coords(defect)
            if global_coord:
                coords.append(global_coord)
                defect['global_location'] = global_coord
                valid_defects.append(defect)
                
        if not coords:
            return []
            
        coords = np.array(coords)
        
        # Adaptive clustering based on defect density
        if len(coords) > 100:
            # For high density, use smaller eps to avoid over-merging
            eps = min(eps, 20)
            
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=self.min_cluster_size).fit(coords)
        
        # Group defects by cluster
        clustered_defects = defaultdict(list)
        for defect, label in zip(valid_defects, clustering.labels_):
            defect['cluster_label'] = label
            clustered_defects[label].append(defect)
            
        # Process clusters
        merged_defects = []
        total_clusters = len(set(clustering.labels_) - {-1})
        
        self.logger.info(f"Clustering resulted in {total_clusters} clusters and "
                        f"{sum(1 for l in clustering.labels_ if l == -1)} noise points")
        
        for cluster_id, cluster_defects in clustered_defects.items():
            if cluster_id == -1:  # Noise points - keep as individual defects
                for defect in cluster_defects:
                    defect['was_clustered'] = False
                    merged_defects.append(defect)
            else:
                # Intelligent merging based on defect similarity
                merged = self.intelligent_merge(cluster_defects)
                merged['was_clustered'] = True
                merged['cluster_id'] = cluster_id
                merged_defects.append(merged)
                
        return merged_defects
        
    def intelligent_merge(self, defects: List[Dict]) -> Dict:
        """Intelligently merge defects considering type compatibility"""
        if len(defects) == 1:
            return defects[0].copy()
            
        # Group by defect type for smarter merging
        type_groups = defaultdict(list)
        for d in defects:
            type_groups[d.get('defect_type', 'UNKNOWN')].append(d)
            
        # If all same type, merge normally
        if len(type_groups) == 1:
            return self.merge_defect_cluster(defects)
            
        # If different types but close together, might be same defect detected differently
        # Check if they're very close (within 10 pixels)
        coords = np.array([d['global_location'] for d in defects])
        distances = np.sqrt(np.sum((coords[:, np.newaxis] - coords) ** 2, axis=2))
        
        if np.max(distances) < 10:
            # Very close - likely same defect
            return self.merge_defect_cluster(defects)
        else:
            # Different defects - return the most severe one
            severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE']
            defects_sorted = sorted(defects, 
                                  key=lambda d: severity_order.index(d.get('severity', 'LOW')))
            
            # Return most severe but note it had neighbors
            result = defects_sorted[0].copy()
            result['nearby_defects'] = len(defects) - 1
            result['nearby_types'] = list(set(d.get('defect_type', 'UNKNOWN') 
                                            for d in defects[1:]))
            return result
            
    def merge_defect_cluster(self, defects: List[Dict]) -> Dict:
        """Enhanced defect merging that preserves more information"""
        if len(defects) == 1:
            return defects[0].copy()
            
        # Calculate weighted centroid based on confidence
        weights = np.array([d.get('confidence', 0.5) for d in defects])
        weights = weights / weights.sum()
        
        coords = np.array([d['global_location'] for d in defects])
        centroid = np.average(coords, weights=weights, axis=0).astype(int)
        
        # Preserve all unique IDs for traceability
        all_ids = [d.get('unique_id', 'unknown') for d in defects]
        
        # Aggregate properties intelligently
        merged = {
            'unique_id': f"merged_{hashlib.md5('_'.join(all_ids).encode()).hexdigest()[:8]}",
            'constituent_ids': all_ids,
            'global_location': tuple(centroid),
            'location_xy': tuple(centroid),
            'cluster_size': len(defects),
            'sources': list(set(d['source_image'] for d in defects)),
            'detection_confidence': float(np.average([d.get('confidence', 0.5) for d in defects], 
                                                   weights=weights)),
            'confidence_std': float(np.std([d.get('confidence', 0.5) for d in defects])),
            'area_px': int(np.sum([d.get('area_px', 0) for d in defects])),
            'severity': max(d.get('severity', 'LOW') for d in defects),
            'defect_types': list(set(d.get('defect_type', 'UNKNOWN') for d in defects)),
            'contributing_algorithms': list(set(
                alg for d in defects 
                for alg in d.get('contributing_algorithms', [])
            )),
            'detection_methods': list(set(
                d.get('detection_method', 'unknown') for d in defects
            )),
        }
        
        # Determine primary defect type by consensus
        type_counts = defaultdict(int)
        for d in defects:
            type_counts[d.get('defect_type', 'UNKNOWN')] += 1
        merged['defect_type'] = max(type_counts, key=type_counts.get)
        merged['type_confidence'] = type_counts[merged['defect_type']] / len(defects)
        
        # Handle directional defects
        if merged['defect_type'] in ['SCRATCH', 'CRACK']:
            orientations = [d.get('orientation', 0) for d in defects if 'orientation' in d]
            if orientations:
                # Circular mean for orientation
                angles = np.array(orientations) * np.pi / 180
                mean_x = np.mean(np.cos(angles))
                mean_y = np.mean(np.sin(angles))
                merged['orientation'] = np.arctan2(mean_y, mean_x) * 180 / np.pi
                merged['orientation_std'] = np.std(orientations)
                merged['direction'] = self.orientation_to_direction(merged['orientation'])
                
        # Preserve any custom fields from original defects
        custom_fields = set()
        for d in defects:
            custom_fields.update(k for k in d.keys() 
                               if k not in merged and not k.startswith('_'))
                               
        for field in custom_fields:
            values = [d.get(field) for d in defects if field in d]
            if values:
                # Handle different value types
                first_value = values[0]
                if isinstance(first_value, (list, dict)):
                    # For lists/dicts, just use the first one
                    merged[field] = first_value
                else:
                    # For hashable types, use most common value
                    try:
                        merged[field] = max(set(values), key=values.count)
                    except TypeError:
                        # If unhashable, use first value
                        merged[field] = first_value
                
        return merged
        
    def orientation_to_direction(self, orientation: float) -> str:
        """Convert orientation angle to cardinal direction"""
        orientation = abs(orientation % 180)
        
        if orientation < 22.5 or orientation > 157.5:
            return "Horizontal"
        elif 67.5 <= orientation <= 112.5:
            return "Vertical"
        elif 22.5 <= orientation < 67.5:
            return "Diagonal-NE"
        else:
            return "Diagonal-NW"
            
    def calculate_defect_heatmap(self, merged_defects: List[Dict], 
                                sigma: float = 20, normalize: bool = True) -> np.ndarray:
        """Create an improved heatmap with severity and confidence weighting"""
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        
        severity_weights = {
            'CRITICAL': 1.0,
            'HIGH': 0.75,
            'MEDIUM': 0.5,
            'LOW': 0.25,
            'NEGLIGIBLE': 0.1
        }
        
        for defect in merged_defects:
            x, y = defect['global_location']
            severity = defect.get('severity', 'LOW')
            weight = severity_weights.get(severity, 0.3)
            confidence = defect.get('detection_confidence', 0.5)
            area = defect.get('area_px', 10)
            
            # Create a small gaussian for each defect
            size = max(5, int(np.sqrt(area)))
            y_min = max(0, y - size)
            y_max = min(self.height, y + size)
            x_min = max(0, x - size)
            x_max = min(self.width, x + size)
            
            if y_max > y_min and x_max > x_min:
                defect_weight = weight * confidence
                heatmap[y_min:y_max, x_min:x_max] += defect_weight
                
        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize
        if normalize and heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap
        
    def create_comprehensive_visualization(self, merged_defects: List[Dict], 
                                         output_path: Path) -> None:
        """Create enhanced visualization with better layout and information"""
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, 
                             height_ratios=[2, 1, 1])
        
        # Convert image to RGB
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # 1. Original image with defect overlay (larger)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.imshow(img_rgb)
        
        # Enhanced color map
        type_colors = {
            'SCRATCH': '#FF0000',
            'CRACK': '#8B0000',
            'PIT': '#0000FF',
            'DIG': '#000080',
            'CONTAMINATION': '#FFD700',
            'CHIP': '#FF8C00',
            'BUBBLE': '#00CED1',
            'BURN': '#FF1493',
            'UNKNOWN': '#808080'
        }
        
        # Plot defects with size variation
        for defect in merged_defects:
            x, y = defect['global_location']
            defect_type = defect.get('defect_type', 'UNKNOWN')
            color = type_colors.get(defect_type, 'gray')
            
            # Size based on area and severity
            base_size = np.sqrt(defect.get('area_px', 100))
            severity_mult = {'CRITICAL': 2.0, 'HIGH': 1.5, 'MEDIUM': 1.0, 
                           'LOW': 0.7, 'NEGLIGIBLE': 0.5}
            size = base_size * severity_mult.get(defect.get('severity', 'LOW'), 1.0)
            size = max(10, min(100, size))
            
            # Plot with confidence-based alpha
            alpha = 0.3 + 0.5 * defect.get('detection_confidence', 0.5)
            ax1.scatter(x, y, c=color, s=size, alpha=alpha, 
                       edgecolors='black', linewidth=0.5)
            
            # Add direction arrows for linear defects
            if 'orientation' in defect and defect_type in ['SCRATCH', 'CRACK']:
                angle = defect['orientation'] * np.pi / 180
                arrow_len = size / 2
                dx = np.cos(angle) * arrow_len
                dy = np.sin(angle) * arrow_len
                ax1.arrow(x-dx/2, y-dy/2, dx, dy, 
                         head_width=size/10, head_length=size/15,
                         fc=color, ec=color, alpha=alpha*0.7)
                
        ax1.set_title(f'Defect Locations (Total: {len(merged_defects)})', 
                     fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Add legend
        legend_elements = [plt.scatter([], [], c=color, s=100, label=dtype, 
                                     edgecolors='black', linewidth=0.5)
                          for dtype, color in type_colors.items() 
                          if any(d.get('defect_type') == dtype for d in merged_defects)]
        if legend_elements:
            ax1.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(1.15, 1), framealpha=0.9)
        
        # 2. Enhanced heatmap
        ax2 = fig.add_subplot(gs[0, 2:])
        heatmap = self.calculate_defect_heatmap(merged_defects, sigma=30)
        
        # Custom colormap
        colors = ['white', '#FFFFE0', '#FFD700', '#FF8C00', '#FF0000', '#8B0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('defect_heat', colors, N=n_bins)
        
        # Overlay heatmap on image
        ax2.imshow(img_rgb, alpha=0.4)
        im = ax2.imshow(heatmap, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(im, ax=ax2, label='Defect Density & Severity')
        cbar.ax.tick_params(labelsize=10)
        
        ax2.set_title('Defect Density Heatmap', fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        # 3. Defect type distribution
        ax3 = fig.add_subplot(gs[1, 0])
        type_counts = defaultdict(int)
        for defect in merged_defects:
            type_counts[defect.get('defect_type', 'UNKNOWN')] += 1
            
        if type_counts:
            types = sorted(type_counts.keys())
            counts = [type_counts[t] for t in types]
            colors_list = [type_colors.get(t, 'gray') for t in types]
            
            bars = ax3.bar(range(len(types)), counts, color=colors_list, 
                          edgecolor='black', linewidth=0.5)
            ax3.set_xticks(range(len(types)))
            ax3.set_xticklabels(types, rotation=45, ha='right')
            ax3.set_ylabel('Count')
            ax3.set_title('Defect Type Distribution', fontsize=14)
            ax3.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontsize=10)
        
        # 4. Severity distribution
        ax4 = fig.add_subplot(gs[1, 1])
        severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE']
        severity_counts = defaultdict(int)
        for defect in merged_defects:
            severity_counts[defect.get('severity', 'UNKNOWN')] += 1
            
        counts = [severity_counts.get(s, 0) for s in severity_order]
        colors_sev = ['#8B0000', '#FF0000', '#FF8C00', '#FFD700', '#90EE90']
        
        bars = ax4.bar(range(len(severity_order)), counts, color=colors_sev,
                       edgecolor='black', linewidth=0.5)
        ax4.set_xticks(range(len(severity_order)))
        ax4.set_xticklabels(severity_order, rotation=45, ha='right')
        ax4.set_ylabel('Count')
        ax4.set_title('Defect Severity Distribution', fontsize=14)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontsize=10)
        
        # 5. Detection confidence distribution
        ax5 = fig.add_subplot(gs[1, 2])
        confidences = [d.get('detection_confidence', 0.5) for d in merged_defects]
        if confidences:
            n, bins, patches = ax5.hist(confidences, bins=20, edgecolor='black', 
                                       alpha=0.7, color='skyblue')
            
            # Color bars by confidence level
            for i, patch in enumerate(patches):
                if bins[i] < 0.3:
                    patch.set_facecolor('red')
                elif bins[i] < 0.7:
                    patch.set_facecolor('orange')
                else:
                    patch.set_facecolor('green')
                    
            ax5.axvline(np.mean(confidences), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(confidences):.2f}')
            ax5.set_xlabel('Detection Confidence')
            ax5.set_ylabel('Count')
            ax5.set_title('Detection Confidence Distribution', fontsize=14)
            ax5.legend()
            ax5.grid(axis='y', alpha=0.3)
        
        # 6. Defect size distribution (log scale)
        ax6 = fig.add_subplot(gs[1, 3])
        sizes = [d.get('area_px', 0) for d in merged_defects if d.get('area_px', 0) > 0]
        if sizes:
            # Create log-spaced bins
            min_size = max(1, min(sizes))
            max_size = max(sizes)
            bins = np.logspace(np.log10(min_size), np.log10(max_size), 20)
            
            ax6.hist(sizes, bins=bins, edgecolor='black', alpha=0.7, color='coral')
            ax6.set_xlabel('Defect Size (pixels²)')
            ax6.set_ylabel('Count')
            ax6.set_title('Defect Size Distribution', fontsize=14)
            ax6.set_xscale('log')
            ax6.grid(axis='y', alpha=0.3)
            
            # Add statistics
            ax6.text(0.95, 0.95, f'Median: {np.median(sizes):.0f} px²\n'
                                f'Mean: {np.mean(sizes):.0f} px²',
                    transform=ax6.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 7. Processing statistics
        ax7 = fig.add_subplot(gs[2, :2])
        ax7.axis('off')
        
        # Calculate statistics
        stats_text = f"""Processing Statistics:
        • Total raw defects detected: {len(self.all_defects)}
        • Defects after clustering: {len(merged_defects)}
        • Reduction ratio: {(1 - len(merged_defects)/max(1, len(self.all_defects))):.1%}
        • Clustering parameters: eps={self.clustering_eps}px, min_samples={self.min_cluster_size}
        • Number of source images: {len(self.detection_results)}
        • Detection algorithms used: {len(set(alg for d in merged_defects for alg in d.get('contributing_algorithms', [])))}
        • Image dimensions: {self.width} × {self.height} pixels
        """
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, 
                fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # 8. Quality assessment
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')
        
        # Calculate quality score
        severity_scores = {'CRITICAL': 25, 'HIGH': 15, 'MEDIUM': 8, 'LOW': 3, 'NEGLIGIBLE': 1}
        quality_score = 100.0
        for d in merged_defects:
            severity = d.get('severity', 'LOW')
            quality_score -= severity_scores.get(severity, 3)
        quality_score = max(0, quality_score)
        
        # Determine pass/fail
        critical_count = sum(1 for d in merged_defects if d.get('severity') == 'CRITICAL')
        high_count = sum(1 for d in merged_defects if d.get('severity') == 'HIGH')
        
        pass_status = 'PASS'
        if critical_count > 0 or high_count > 2 or quality_score < 70:
            pass_status = 'FAIL'
            
        quality_color = 'green' if pass_status == 'PASS' else 'red'
        
        quality_text = f"""Quality Assessment:
        • Overall Quality Score: {quality_score:.1f}/100
        • Pass/Fail Status: {pass_status}
        • Critical defects: {critical_count}
        • High severity defects: {high_count}
        • Average detection confidence: {np.mean(confidences):.2%}
        • Total affected area: {sum(d.get('area_px', 0) for d in merged_defects):,} pixels
        • Affected area percentage: {sum(d.get('area_px', 0) for d in merged_defects)/(self.width*self.height)*100:.2f}%
        """
        
        ax8.text(0.05, 0.95, quality_text, transform=ax8.transAxes,
                fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.2))
        
        # Overall title
        plt.suptitle(f'Comprehensive Defect Analysis - {self.original_image_path.name}',
                    fontsize=18, fontweight='bold')
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        self.logger.info(f"Saved comprehensive visualization to {output_path}")
        
    def generate_final_report(self, merged_defects: List[Dict], 
                            output_path: Path) -> Dict:
        """Generate enhanced JSON report with complete analysis"""
        # Calculate comprehensive statistics
        total_defects = len(merged_defects)
        
        # Group defects
        defects_by_type = defaultdict(list)
        defects_by_severity = defaultdict(list)
        defects_by_source = defaultdict(list)
        
        for d in merged_defects:
            defects_by_type[d.get('defect_type', 'UNKNOWN')].append(d)
            defects_by_severity[d.get('severity', 'UNKNOWN')].append(d)
            for source in d.get('sources', ['unknown']):
                defects_by_source[source].append(d)
        
        # Quality metrics
        severity_scores = {'CRITICAL': 25, 'HIGH': 15, 'MEDIUM': 8, 'LOW': 3, 'NEGLIGIBLE': 1}
        quality_score = 100.0
        severity_deductions = {}
        
        for severity, defects in defects_by_severity.items():
            deduction = len(defects) * severity_scores.get(severity, 3)
            severity_deductions[severity] = deduction
            quality_score -= deduction
            
        quality_score = max(0, quality_score)
        
        # Pass/fail determination
        critical_count = len(defects_by_severity.get('CRITICAL', []))
        high_count = len(defects_by_severity.get('HIGH', []))
        
        pass_fail = 'PASS'
        failure_reasons = []
        
        if critical_count > 0:
            pass_fail = 'FAIL'
            failure_reasons.append(f"{critical_count} critical defect(s) found")
        if high_count > 2:
            pass_fail = 'FAIL'
            failure_reasons.append(f"{high_count} high-severity defects exceed threshold")
        if quality_score < 70:
            pass_fail = 'FAIL'
            failure_reasons.append(f"Quality score {quality_score:.1f} below threshold (70)")
            
        # Calculate additional metrics
        total_area = sum(d.get('area_px', 0) for d in merged_defects)
        image_area = self.width * self.height
        affected_percentage = (total_area / image_area) * 100 if image_area > 0 else 0
        
        confidences = [d.get('detection_confidence', 0.5) for d in merged_defects]
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'original_image': str(self.original_image_path),
            'image_properties': {
                'width': self.width,
                'height': self.height,
                'total_pixels': image_area
            },
            'analysis_summary': {
                'total_sources_analyzed': len(self.detection_results),
                'total_raw_defects': len(self.all_defects),
                'total_merged_defects': total_defects,
                'clustering_reduction': f"{(1 - total_defects/max(1, len(self.all_defects))):.1%}",
                'quality_score': round(quality_score, 2),
                'pass_fail_status': pass_fail,
                'failure_reasons': failure_reasons
            },
            'defect_statistics': {
                'by_type': {k: len(v) for k, v in defects_by_type.items()},
                'by_severity': {k: len(v) for k, v in defects_by_severity.items()},
                'by_source': {k: len(v) for k, v in defects_by_source.items()},
                'confidence_metrics': {
                    'mean': round(np.mean(confidences), 3) if confidences else 0,
                    'std': round(np.std(confidences), 3) if confidences else 0,
                    'min': round(min(confidences), 3) if confidences else 0,
                    'max': round(max(confidences), 3) if confidences else 0
                },
                'size_metrics': {
                    'total_affected_area_px': total_area,
                    'affected_percentage': round(affected_percentage, 3),
                    'average_defect_size': round(total_area / max(1, total_defects), 1),
                    'largest_defect_px': max((d.get('area_px', 0) for d in merged_defects), default=0)
                }
            },
            'quality_breakdown': {
                'initial_score': 100,
                'deductions': severity_deductions,
                'final_score': round(quality_score, 2)
            },
            'processing_parameters': {
                'clustering_eps': self.clustering_eps,
                'min_cluster_size': self.min_cluster_size,
                'detection_sources': len(self.detection_results)
            },
            'defects': [self.format_defect_for_report(d, i) 
                       for i, d in enumerate(merged_defects)],
            'data_integrity': {
                'total_warnings': len([log for log in self.data_integrity_log 
                                     if log['severity'] == 'WARNING']),
                'total_errors': len([log for log in self.data_integrity_log 
                                   if log['severity'] == 'ERROR']),
                'integrity_log': self.data_integrity_log[:10]  # First 10 issues
            }
        }
        
        # Create backup if file exists
        if output_path.exists():
            backup_path = output_path.with_suffix(f'.backup_{datetime.now():%Y%m%d_%H%M%S}.json')
            shutil.copy2(output_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
            
        self.logger.info(f"Saved final report to {output_path}")
        
        return report
        
    def format_defect_for_report(self, defect: Dict, index: int) -> Dict:
        """Format defect with complete information for report"""
        formatted = {
            'index': index,
            'unique_id': defect.get('unique_id', f'defect_{index}'),
            'type': defect.get('defect_type', 'UNKNOWN'),
            'severity': defect.get('severity', 'UNKNOWN'),
            'location': {
                'x': defect['global_location'][0],
                'y': defect['global_location'][1]
            },
            'properties': {
                'area_px': defect.get('area_px', 0),
                'confidence': round(defect.get('detection_confidence', 0.5), 3),
                'was_clustered': defect.get('was_clustered', False),
                'cluster_size': defect.get('cluster_size', 1)
            },
            'detection': {
                'algorithms': defect.get('contributing_algorithms', []),
                'sources': defect.get('sources', []),
                'methods': defect.get('detection_methods', [])
            }
        }
        
        # Add type-specific properties
        if defect.get('defect_type') in ['SCRATCH', 'CRACK'] and 'orientation' in defect:
            formatted['properties']['orientation'] = {
                'angle_deg': round(defect['orientation'], 1),
                'direction': defect.get('direction', 'unknown'),
                'std_dev': round(defect.get('orientation_std', 0), 1)
            }
            
        # Add clustering information if available
        if defect.get('was_clustered') and 'constituent_ids' in defect:
            formatted['clustering'] = {
                'merged_from': len(defect['constituent_ids']),
                'constituent_ids': defect['constituent_ids'][:5]  # First 5 for brevity
            }
            
        # Add confidence metrics
        if 'confidence_std' in defect:
            formatted['properties']['confidence_std'] = round(defect['confidence_std'], 3)
            
        # Add nearby defects info if available
        if 'nearby_defects' in defect:
            formatted['nearby_defects'] = {
                'count': defect['nearby_defects'],
                'types': defect.get('nearby_types', [])
            }
            
        return formatted
        
    def create_text_summary(self, report: Dict, output_path: Path) -> None:
        """Create an enhanced human-readable summary"""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FIBER OPTIC DEFECT ANALYSIS - COMPREHENSIVE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report Version: {report.get('version', '1.0')}\n")
            f.write(f"Analysis Date: {report['timestamp']}\n")
            f.write(f"Image: {report['original_image']}\n")
            f.write(f"Dimensions: {report['image_properties']['width']} × "
                   f"{report['image_properties']['height']} pixels\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            summary = report['analysis_summary']
            f.write(f"Overall Status: {summary['pass_fail_status']}\n")
            f.write(f"Quality Score: {summary['quality_score']}/100\n")
            f.write(f"Total Defects Found: {summary['total_merged_defects']}\n")
            
            if summary['failure_reasons']:
                f.write("\nFailure Reasons:\n")
                for reason in summary['failure_reasons']:
                    f.write(f"  ✗ {reason}\n")
            else:
                f.write("✓ All quality criteria met\n")
                
            # Processing Summary
            f.write("\nPROCESSING SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Detection Sources Analyzed: {summary['total_sources_analyzed']}\n")
            f.write(f"Raw Defects Detected: {summary['total_raw_defects']}\n")
            f.write(f"After Clustering/Merging: {summary['total_merged_defects']}\n")
            f.write(f"Reduction Ratio: {summary['clustering_reduction']}\n")
            
            # Defect Statistics
            stats = report['defect_statistics']
            f.write("\nDEFECT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            f.write("Distribution by Type:\n")
            for dtype, count in sorted(stats['by_type'].items(), 
                                     key=lambda x: x[1], reverse=True):
                percentage = (count / summary['total_merged_defects'] * 100) if summary['total_merged_defects'] > 0 else 0
                f.write(f"  • {dtype:<15} {count:3d} ({percentage:5.1f}%)\n")
                
            f.write("\nDistribution by Severity:\n")
            severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE']
            for severity in severity_order:
                count = stats['by_severity'].get(severity, 0)
                percentage = (count / summary['total_merged_defects'] * 100) if summary['total_merged_defects'] > 0 else 0
                symbol = '⚠' if severity in ['CRITICAL', 'HIGH'] else '•'
                f.write(f"  {symbol} {severity:<15} {count:3d} ({percentage:5.1f}%)\n")
                
            # Size and Coverage
            f.write("\nCOVERAGE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            size_metrics = stats['size_metrics']
            f.write(f"Total Affected Area: {size_metrics['total_affected_area_px']:,} pixels\n")
            f.write(f"Percentage of Image: {size_metrics['affected_percentage']:.3f}%\n")
            f.write(f"Average Defect Size: {size_metrics['average_defect_size']:.1f} pixels\n")
            f.write(f"Largest Defect: {size_metrics['largest_defect_px']:,} pixels\n")
            
            # Confidence Analysis
            f.write("\nDETECTION CONFIDENCE\n")
            f.write("-" * 40 + "\n")
            conf_metrics = stats['confidence_metrics']
            f.write(f"Mean Confidence: {conf_metrics['mean']:.1%}\n")
            f.write(f"Std Deviation: ±{conf_metrics['std']:.1%}\n")
            f.write(f"Range: {conf_metrics['min']:.1%} - {conf_metrics['max']:.1%}\n")
            
            # Critical Defects Detail
            critical_defects = [d for d in report['defects'] if d['severity'] == 'CRITICAL']
            if critical_defects:
                f.write("\nCRITICAL DEFECTS DETAIL\n")
                f.write("-" * 40 + "\n")
                for i, defect in enumerate(critical_defects[:5], 1):
                    f.write(f"\n{i}. {defect['type']} (ID: {defect['unique_id']})\n")
                    f.write(f"   Location: ({defect['location']['x']}, {defect['location']['y']})\n")
                    f.write(f"   Size: {defect['properties']['area_px']} pixels\n")
                    f.write(f"   Confidence: {defect['properties']['confidence']:.1%}\n")
                    f.write(f"   Detected by: {', '.join(defect['detection']['algorithms'])}\n")
                    
                if len(critical_defects) > 5:
                    f.write(f"\n   ... and {len(critical_defects) - 5} more critical defects\n")
                    
            # Data Integrity
            integrity = report.get('data_integrity', {})
            if integrity.get('total_warnings', 0) > 0 or integrity.get('total_errors', 0) > 0:
                f.write("\nDATA INTEGRITY NOTES\n")
                f.write("-" * 40 + "\n")
                f.write(f"Warnings: {integrity.get('total_warnings', 0)}\n")
                f.write(f"Errors: {integrity.get('total_errors', 0)}\n")
                
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            if summary['pass_fail_status'] == 'FAIL':
                if critical_count := len([d for d in report['defects'] if d['severity'] == 'CRITICAL']):
                    f.write(f"• Address {critical_count} critical defects immediately\n")
                if high_count := len([d for d in report['defects'] if d['severity'] == 'HIGH']):
                    f.write(f"• Review and remediate {high_count} high-severity defects\n")
                if size_metrics['affected_percentage'] > 1.0:
                    f.write(f"• Significant surface area affected ({size_metrics['affected_percentage']:.1f}%)\n")
            else:
                f.write("• Product meets quality standards\n")
                if medium_count := len([d for d in report['defects'] if d['severity'] == 'MEDIUM']):
                    f.write(f"• Monitor {medium_count} medium-severity defects\n")
                if conf_metrics['mean'] < 0.7:
                    f.write("• Consider improving detection confidence\n")
                    
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
            
        self.logger.info(f"Created text summary: {output_path}")
        
    def run_complete_analysis(self) -> Dict:
        """Run the complete analysis pipeline with enhanced error handling"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Comprehensive Defect Analysis")
        self.logger.info("=" * 60)
        
        try:
            # Load all detection results
            self.logger.info("Step 1: Loading detection results...")
            self.load_all_detection_results()
            
            if not self.all_defects:
                self.logger.warning("No defects found in detection results!")
                return self._create_empty_report()
                
            # Load separation masks
            self.logger.info("Step 2: Loading region separation masks...")
            self.load_separation_masks()
            
            # Perform clustering
            self.logger.info("Step 3: Clustering and merging defects...")
            merged_defects = self.cluster_defects()
            
            self.logger.info(f"Clustering complete: {len(self.all_defects)} → "
                           f"{len(merged_defects)} defects")
            
            # Create output directory structure
            output_dir = self.results_dir / "4_final_analysis"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Archive previous results if they exist
            self._archive_previous_results(output_dir)
            
            base_name = self.original_image_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate outputs
            self.logger.info("Step 4: Generating visualizations...")
            viz_path = output_dir / f"{base_name}_comprehensive_analysis.png"
            self.create_comprehensive_visualization(merged_defects, viz_path)
            
            self.logger.info("Step 5: Generating final report...")
            report_path = output_dir / f"{base_name}_final_report.json"
            report = self.generate_final_report(merged_defects, report_path)
            
            # Create summary
            summary_path = output_dir / f"{base_name}_summary.txt"
            self.create_text_summary(report, summary_path)
            
            # Save data integrity log
            if self.data_integrity_log:
                integrity_path = output_dir / f"{base_name}_integrity_log.json"
                with open(integrity_path, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'total_issues': len(self.data_integrity_log),
                        'issues': self.data_integrity_log
                    }, f, indent=2, cls=NumpyEncoder)
                    
            self.logger.info("=" * 60)
            self.logger.info(f"Analysis Complete! Results saved to: {output_dir}")
            self.logger.info(f"Final Status: {report['analysis_summary']['pass_fail_status']}")
            self.logger.info(f"Quality Score: {report['analysis_summary']['quality_score']}/100")
            self.logger.info("=" * 60)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Critical error during analysis: {str(e)}", exc_info=True)
            raise
            
    def _create_empty_report(self) -> Dict:
        """Create an empty report when no defects are found"""
        return {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'original_image': str(self.original_image_path),
            'analysis_summary': {
                'total_sources_analyzed': len(self.detection_results),
                'total_merged_defects': 0,
                'quality_score': 100.0,
                'pass_fail_status': 'PASS',
                'failure_reasons': []
            },
            'defects': [],
            'message': 'No defects detected in any source'
        }
        
    def _archive_previous_results(self, output_dir: Path) -> None:
        """Archive previous results if they exist"""
        existing_files = list(output_dir.glob(f"{self.original_image_path.stem}_*"))
        
        if existing_files:
            archive_dir = output_dir / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            for file in existing_files:
                if not file.is_dir():
                    shutil.move(str(file), str(archive_dir / file.name))
                    
            self.logger.info(f"Archived {len(existing_files)} previous results to {archive_dir}")


def integrate_with_pipeline(results_base_dir: str, image_name: str, 
                          clustering_eps: float = 30.0) -> Dict:
    """Enhanced integration function with better error handling"""
    results_dir = Path(results_base_dir)
    
    # Validate inputs
    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
        
    # Find original image with improved search
    search_paths = [
        results_dir.parent,
        results_dir,
        results_dir / "0_original",
        Path("."),
    ]
    
    original_image_path = None
    
    for search_dir in search_paths:
        if not search_dir.exists():
            continue
            
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            candidates = list(search_dir.glob(f"{image_name}{ext}"))
            if candidates:
                original_image_path = candidates[0]
                break
                
        if original_image_path:
            break
            
    if not original_image_path:
        raise ValueError(f"Could not find original image for '{image_name}' "
                        f"in any of the search paths: {search_paths}")
        
    logging.info(f"Found original image: {original_image_path}")
    
    # Run analysis
    try:
        aggregator = DefectAggregator(results_dir, original_image_path, 
                                    clustering_eps=clustering_eps)
        report = aggregator.run_complete_analysis()
        return report
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise


def main():
    """Enhanced main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fiber Optic Defect Data Acquisition and Analysis"
    )
    parser.add_argument("results_directory", 
                       help="Path to results directory containing detection outputs")
    parser.add_argument("image_name", 
                       help="Name of the original image (without extension)")
    parser.add_argument("--clustering-eps", type=float, default=30.0,
                       help="DBSCAN clustering epsilon parameter (default: 30.0)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        report = integrate_with_pipeline(
            args.results_directory, 
            args.image_name,
            clustering_eps=args.clustering_eps
        )
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Status: {report['analysis_summary']['pass_fail_status']}")
        print(f"Quality Score: {report['analysis_summary']['quality_score']}/100")
        print(f"Total Defects: {report['analysis_summary']['total_merged_defects']}")
        
        if report['analysis_summary']['failure_reasons']:
            print("\nFailure Reasons:")
            for reason in report['analysis_summary']['failure_reasons']:
                print(f"  • {reason}")
                
        print("=" * 60)
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()