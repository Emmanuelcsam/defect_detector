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
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

class DefectAggregator:
    """Aggregates and analyzes defects from multiple detection results"""
    
    def __init__(self, results_dir: Path, original_image_path: Path):
        self.results_dir = results_dir
        self.original_image_path = original_image_path
        self.original_image = cv2.imread(str(original_image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load original image: {original_image_path}")
            
        self.height, self.width = self.original_image.shape[:2]
        self.all_defects = []
        self.region_masks = {}
        self.detection_results = []
        self.logger = logging.getLogger(__name__)
        
    def load_all_detection_results(self):
        """Load all detection results from the results directory"""
        self.logger.info(f"Loading detection results from: {self.results_dir}")
        
        # Find all JSON report files
        detection_dir = self.results_dir / "3_detected"
        if not detection_dir.exists():
            raise ValueError(f"Detection directory not found: {detection_dir}")
            
        # Scan for all report files
        report_files = list(detection_dir.rglob("*_report.json"))
        self.logger.info(f"Found {len(report_files)} detection reports")
        
        for report_file in report_files:
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    
                # Extract source information
                source_path = Path(report.get('image_path', ''))
                source_name = source_path.stem
                
                # Determine if this is a region or full image
                is_region = any(zone in source_name for zone in ['_core', '_cladding', '_ferrule'])
                region_type = None
                if is_region:
                    for zone in ['core', 'cladding', 'ferrule']:
                        if f'_{zone}' in source_name:
                            region_type = zone
                            break
                
                # Load region mask if available
                mask_path = report_file.parent / f"{source_name.replace('_report', '_mask')}.npy"
                mask = None
                if mask_path.exists():
                    try:
                        mask = np.load(mask_path)
                    except:
                        pass
                
                # Store detection result
                self.detection_results.append({
                    'report': report,
                    'source_name': source_name,
                    'is_region': is_region,
                    'region_type': region_type,
                    'mask': mask,
                    'file_path': report_file
                })
                
                # Extract defects
                defects = report.get('defects', [])
                for defect in defects:
                    # Add source information to defect
                    defect['source_image'] = source_name
                    defect['is_region'] = is_region
                    defect['region_type'] = region_type
                    self.all_defects.append(defect)
                    
            except Exception as e:
                self.logger.error(f"Error loading {report_file}: {str(e)}")
                
        self.logger.info(f"Loaded {len(self.all_defects)} total defects from {len(self.detection_results)} sources")
        
    def load_separation_masks(self):
        """Load the separation masks to map regions back to original coordinates"""
        separation_dir = self.results_dir / "2_separated" / self.original_image_path.stem
        
        if separation_dir.exists():
            for mask_type in ['core', 'cladding', 'ferrule']:
                mask_file = separation_dir / f"{mask_type}_mask.npy"
                if mask_file.exists():
                    try:
                        self.region_masks[mask_type] = np.load(mask_file)
                        self.logger.info(f"Loaded {mask_type} mask")
                    except:
                        self.logger.warning(f"Could not load {mask_type} mask")
                        
    def map_defect_to_global_coords(self, defect: Dict) -> Optional[Tuple[int, int]]:
        """Map a defect from region coordinates to global image coordinates"""
        if not defect.get('is_region'):
            # Already in global coordinates
            return defect.get('location_xy')
            
        region_type = defect.get('region_type')
        if not region_type or region_type not in self.region_masks:
            return defect.get('location_xy')
            
        # Get the region mask
        mask = self.region_masks[region_type]
        
        # Find the bounding box of the region
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return defect.get('location_xy')
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Map local coordinates to global
        local_x, local_y = defect.get('location_xy', (0, 0))
        global_x = x_min + local_x
        global_y = y_min + local_y
        
        # Ensure within bounds
        global_x = max(0, min(global_x, self.width - 1))
        global_y = max(0, min(global_y, self.height - 1))
        
        return (global_x, global_y)
        
    def cluster_defects(self, eps=50, min_samples=1):
        """Cluster nearby defects to remove duplicates and find patterns"""
        if not self.all_defects:
            return []
            
        # Extract coordinates
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
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        # Group defects by cluster
        clustered_defects = defaultdict(list)
        for defect, label in zip(valid_defects, clustering.labels_):
            clustered_defects[label].append(defect)
            
        # Merge clustered defects
        merged_defects = []
        
        for cluster_id, cluster_defects in clustered_defects.items():
            if cluster_id == -1:  # Noise points
                merged_defects.extend(cluster_defects)
            else:
                # Merge defects in the same cluster
                merged_defect = self.merge_defect_cluster(cluster_defects)
                merged_defects.append(merged_defect)
                
        return merged_defects
        
    def merge_defect_cluster(self, defects: List[Dict]) -> Dict:
        """Merge multiple defects into a single representative defect"""
        if len(defects) == 1:
            return defects[0]
            
        # Calculate centroid
        coords = np.array([d['global_location'] for d in defects])
        centroid = coords.mean(axis=0).astype(int)
        
        # Aggregate properties
        merged = {
            'global_location': tuple(centroid),
            'location_xy': tuple(centroid),
            'cluster_size': len(defects),
            'sources': list(set(d['source_image'] for d in defects)),
            'detection_confidence': np.mean([d.get('confidence', 0.5) for d in defects]),
            'area_px': int(np.mean([d.get('area_px', 0) for d in defects])),
            'severity': max(d.get('severity', 'LOW') for d in defects),
            'defect_types': list(set(d.get('defect_type', 'UNKNOWN') for d in defects)),
            'contributing_algorithms': list(set(
                alg for d in defects 
                for alg in d.get('contributing_algorithms', [])
            )),
        }
        
        # Calculate bounding box
        all_bboxes = [d.get('bbox', (0, 0, 0, 0)) for d in defects if 'bbox' in d]
        if all_bboxes:
            x_coords = [b[0] for b in all_bboxes] + [b[0] + b[2] for b in all_bboxes]
            y_coords = [b[1] for b in all_bboxes] + [b[1] + b[3] for b in all_bboxes]
            merged['bbox'] = (
                min(x_coords), min(y_coords),
                max(x_coords) - min(x_coords),
                max(y_coords) - min(y_coords)
            )
            
        # Determine primary defect type
        type_counts = defaultdict(int)
        for d in defects:
            type_counts[d.get('defect_type', 'UNKNOWN')] += 1
        merged['defect_type'] = max(type_counts, key=type_counts.get)
        
        # Calculate direction if applicable (for scratches/cracks)
        if merged['defect_type'] in ['SCRATCH', 'CRACK']:
            orientations = [d.get('orientation', 0) for d in defects if 'orientation' in d]
            if orientations:
                # Average orientation with circular mean
                angles = np.array(orientations) * np.pi / 180
                mean_x = np.mean(np.cos(angles))
                mean_y = np.mean(np.sin(angles))
                merged['orientation'] = np.arctan2(mean_y, mean_x) * 180 / np.pi
                merged['direction'] = self.orientation_to_direction(merged['orientation'])
                
        return merged
        
    def orientation_to_direction(self, orientation: float) -> str:
        """Convert orientation angle to cardinal direction"""
        # Normalize to 0-180 range
        orientation = abs(orientation % 180)
        
        if orientation < 22.5 or orientation > 157.5:
            return "Horizontal"
        elif 67.5 <= orientation <= 112.5:
            return "Vertical"
        elif 22.5 <= orientation < 67.5:
            return "Diagonal-NE"
        else:
            return "Diagonal-NW"
            
    def calculate_defect_heatmap(self, merged_defects: List[Dict], sigma=20):
        """Create a heatmap showing defect density and severity"""
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        
        severity_weights = {
            'CRITICAL': 1.0,
            'HIGH': 0.7,
            'MEDIUM': 0.5,
            'LOW': 0.3,
            'NEGLIGIBLE': 0.1
        }
        
        for defect in merged_defects:
            x, y = defect['global_location']
            severity = defect.get('severity', 'LOW')
            weight = severity_weights.get(severity, 0.3)
            confidence = defect.get('detection_confidence', 0.5)
            
            # Add weighted point to heatmap
            if 0 <= y < self.height and 0 <= x < self.width:
                heatmap[y, x] += weight * confidence
                
        # Apply Gaussian smoothing to create density map
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap
        
    def create_comprehensive_visualization(self, merged_defects: List[Dict], output_path: Path):
        """Create a comprehensive visualization of all findings"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Original image with defect overlay
        ax1 = fig.add_subplot(gs[0, 0])
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_rgb)
        
        # Color map for defect types
        type_colors = {
            'SCRATCH': 'red',
            'CRACK': 'darkred',
            'PIT': 'blue',
            'DIG': 'navy',
            'CONTAMINATION': 'yellow',
            'CHIP': 'orange',
            'BUBBLE': 'cyan',
            'BURN': 'magenta',
            'UNKNOWN': 'gray'
        }
        
        # Plot defects
        for defect in merged_defects:
            x, y = defect['global_location']
            defect_type = defect.get('defect_type', 'UNKNOWN')
            color = type_colors.get(defect_type, 'gray')
            
            # Draw marker
            size = max(5, min(20, defect.get('area_px', 10) / 10))
            ax1.scatter(x, y, c=color, s=size**2, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Draw direction arrow for linear defects
            if 'orientation' in defect and defect_type in ['SCRATCH', 'CRACK']:
                angle = defect['orientation'] * np.pi / 180
                dx = np.cos(angle) * 20
                dy = np.sin(angle) * 20
                ax1.arrow(x, y, dx, dy, head_width=5, head_length=3, 
                         fc=color, ec=color, alpha=0.5)
                
        ax1.set_title(f'Defect Locations (Total: {len(merged_defects)})', fontsize=14)
        ax1.axis('off')
        
        # 2. Defect heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        heatmap = self.calculate_defect_heatmap(merged_defects)
        
        # Create custom colormap
        colors = ['white', 'yellow', 'orange', 'red', 'darkred']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('defect_heat', colors, N=n_bins)
        
        im = ax2.imshow(heatmap, cmap=cmap, alpha=0.8)
        ax2.imshow(img_rgb, alpha=0.3)
        plt.colorbar(im, ax=ax2, label='Defect Density')
        ax2.set_title('Defect Density Heatmap', fontsize=14)
        ax2.axis('off')
        
        # 3. Defect type distribution
        ax3 = fig.add_subplot(gs[0, 2])
        type_counts = defaultdict(int)
        for defect in merged_defects:
            type_counts[defect.get('defect_type', 'UNKNOWN')] += 1
            
        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors = [type_colors.get(t, 'gray') for t in types]
            
            bars = ax3.bar(range(len(types)), counts, color=colors)
            ax3.set_xticks(range(len(types)))
            ax3.set_xticklabels(types, rotation=45, ha='right')
            ax3.set_ylabel('Count')
            ax3.set_title('Defect Type Distribution', fontsize=14)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom')
        
        # 4. Severity distribution
        ax4 = fig.add_subplot(gs[1, 0])
        severity_counts = defaultdict(int)
        for defect in merged_defects:
            severity_counts[defect.get('severity', 'UNKNOWN')] += 1
            
        if severity_counts:
            severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE']
            counts = [severity_counts.get(s, 0) for s in severities]
            colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen']
            
            bars = ax4.bar(range(len(severities)), counts, color=colors)
            ax4.set_xticks(range(len(severities)))
            ax4.set_xticklabels(severities, rotation=45, ha='right')
            ax4.set_ylabel('Count')
            ax4.set_title('Defect Severity Distribution', fontsize=14)
            
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{count}', ha='center', va='bottom')
        
        # 5. Detection confidence distribution
        ax5 = fig.add_subplot(gs[1, 1])
        confidences = [d.get('detection_confidence', 0.5) for d in merged_defects]
        if confidences:
            ax5.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
            ax5.axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.2f}')
            ax5.set_xlabel('Detection Confidence')
            ax5.set_ylabel('Count')
            ax5.set_title('Detection Confidence Distribution', fontsize=14)
            ax5.legend()
        
        # 6. Size distribution
        ax6 = fig.add_subplot(gs[1, 2])
        sizes = [d.get('area_px', 0) for d in merged_defects if d.get('area_px', 0) > 0]
        if sizes:
            ax6.hist(sizes, bins=30, edgecolor='black', alpha=0.7)
            ax6.set_xlabel('Defect Size (pixels²)')
            ax6.set_ylabel('Count')
            ax6.set_title('Defect Size Distribution', fontsize=14)
            ax6.set_yscale('log')
            
        plt.suptitle(f'Comprehensive Defect Analysis - {self.original_image_path.name}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved comprehensive visualization to {output_path}")
        
    def generate_final_report(self, merged_defects: List[Dict], output_path: Path):
        """Generate a comprehensive JSON report with all findings"""
        # Calculate statistics
        total_defects = len(merged_defects)
        
        # Group by type
        defects_by_type = defaultdict(list)
        for d in merged_defects:
            defects_by_type[d.get('defect_type', 'UNKNOWN')].append(d)
            
        # Group by severity
        defects_by_severity = defaultdict(list)
        for d in merged_defects:
            defects_by_severity[d.get('severity', 'UNKNOWN')].append(d)
            
        # Calculate quality metrics
        severity_scores = {
            'CRITICAL': 25,
            'HIGH': 15,
            'MEDIUM': 8,
            'LOW': 3,
            'NEGLIGIBLE': 1
        }
        
        quality_score = 100.0
        for d in merged_defects:
            severity = d.get('severity', 'LOW')
            quality_score -= severity_scores.get(severity, 3)
        quality_score = max(0, quality_score)
        
        # Determine pass/fail
        critical_count = len(defects_by_severity.get('CRITICAL', []))
        high_count = len(defects_by_severity.get('HIGH', []))
        
        pass_fail = 'PASS'
        failure_reasons = []
        if critical_count > 0:
            pass_fail = 'FAIL'
            failure_reasons.append(f"{critical_count} critical defects found")
        if high_count > 2:
            pass_fail = 'FAIL'
            failure_reasons.append(f"{high_count} high-severity defects found")
        if quality_score < 70:
            pass_fail = 'FAIL'
            failure_reasons.append(f"Quality score too low ({quality_score:.1f})")
            
        # Create final report
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_image': str(self.original_image_path),
            'total_sources_analyzed': len(self.detection_results),
            'total_defects_found': total_defects,
            'quality_score': round(quality_score, 2),
            'pass_fail_status': pass_fail,
            'failure_reasons': failure_reasons,
            'statistics': {
                'by_type': {k: len(v) for k, v in defects_by_type.items()},
                'by_severity': {k: len(v) for k, v in defects_by_severity.items()},
                'average_confidence': round(np.mean([d.get('detection_confidence', 0.5) 
                                                   for d in merged_defects]), 3),
                'total_affected_area_px': sum(d.get('area_px', 0) for d in merged_defects),
            },
            'defects': [self.format_defect_for_report(d) for d in merged_defects],
            'processing_info': {
                'clustering_eps': 50,
                'total_raw_defects': len(self.all_defects),
                'defects_after_merging': total_defects,
                'reduction_ratio': round(1 - total_defects/max(1, len(self.all_defects)), 3)
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Saved final report to {output_path}")
        
        return report
        
    def format_defect_for_report(self, defect: Dict) -> Dict:
        """Format a defect for the final report"""
        return {
            'id': defect.get('defect_id', 'N/A'),
            'type': defect.get('defect_type', 'UNKNOWN'),
            'severity': defect.get('severity', 'UNKNOWN'),
            'location': defect.get('global_location', [0, 0]),
            'size_px': defect.get('area_px', 0),
            'confidence': round(defect.get('detection_confidence', 0.5), 3),
            'direction': defect.get('direction', 'N/A'),
            'orientation_deg': round(defect.get('orientation', 0), 1) if 'orientation' in defect else 'N/A',
            'detected_by': defect.get('contributing_algorithms', []),
            'sources': defect.get('sources', []),
            'cluster_size': defect.get('cluster_size', 1)
        }
        
    def run_complete_analysis(self):
        """Run the complete data acquisition and analysis pipeline"""
        self.logger.info("Starting comprehensive data acquisition and analysis...")
        
        # Load all data
        self.load_all_detection_results()
        self.load_separation_masks()
        
        # Cluster and merge defects
        self.logger.info("Clustering and merging defects...")
        merged_defects = self.cluster_defects(eps=50, min_samples=1)
        self.logger.info(f"Reduced {len(self.all_defects)} raw defects to {len(merged_defects)} unique defects")
        
        # Create output directory
        output_dir = self.results_dir / "4_final_analysis"
        output_dir.mkdir(exist_ok=True)
        
        base_name = self.original_image_path.stem
        
        # Generate comprehensive visualization
        viz_path = output_dir / f"{base_name}_comprehensive_analysis.png"
        self.create_comprehensive_visualization(merged_defects, viz_path)
        
        # Generate final report
        report_path = output_dir / f"{base_name}_final_report.json"
        report = self.generate_final_report(merged_defects, report_path)
        
        # Create summary text file
        summary_path = output_dir / f"{base_name}_summary.txt"
        self.create_text_summary(report, summary_path)
        
        self.logger.info(f"Analysis complete! Results saved to {output_dir}")
        
        return report
        
    def create_text_summary(self, report: Dict, output_path: Path):
        """Create a human-readable text summary"""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FIBER OPTIC DEFECT ANALYSIS - FINAL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Image: {report['original_image']}\n")
            f.write(f"Analysis Date: {report['timestamp']}\n")
            f.write(f"Sources Analyzed: {report['total_sources_analyzed']}\n\n")
            
            f.write("OVERALL RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Status: {report['pass_fail_status']}\n")
            f.write(f"Quality Score: {report['quality_score']}/100\n")
            f.write(f"Total Defects: {report['total_defects_found']}\n")
            
            if report['failure_reasons']:
                f.write("\nFailure Reasons:\n")
                for reason in report['failure_reasons']:
                    f.write(f"  - {reason}\n")
                    
            f.write("\nDEFECT BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            
            f.write("By Type:\n")
            for dtype, count in report['statistics']['by_type'].items():
                f.write(f"  {dtype}: {count}\n")
                
            f.write("\nBy Severity:\n")
            for severity, count in report['statistics']['by_severity'].items():
                f.write(f"  {severity}: {count}\n")
                
            f.write(f"\nAverage Detection Confidence: {report['statistics']['average_confidence']}\n")
            f.write(f"Total Affected Area: {report['statistics']['total_affected_area_px']} pixels\n")
            
            f.write("\nPROCESSING STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Raw Defects Found: {report['processing_info']['total_raw_defects']}\n")
            f.write(f"After Deduplication: {report['processing_info']['defects_after_merging']}\n")
            f.write(f"Reduction Ratio: {report['processing_info']['reduction_ratio']:.1%}\n")
            
            # List critical defects
            critical_defects = [d for d in report['defects'] if d['severity'] == 'CRITICAL']
            if critical_defects:
                f.write("\nCRITICAL DEFECTS\n")
                f.write("-" * 40 + "\n")
                for i, defect in enumerate(critical_defects[:10], 1):
                    f.write(f"{i}. Type: {defect['type']}, Location: {defect['location']}, "
                           f"Size: {defect['size_px']}px\n")
                if len(critical_defects) > 10:
                    f.write(f"   ... and {len(critical_defects) - 10} more\n")
                    
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")


def integrate_with_pipeline(results_base_dir: str, image_name: str):
    """Integration function to be called from app.py"""
    results_dir = Path(results_base_dir)
    
    # Find original image
    original_candidates = [
        results_dir.parent / f"{image_name}.png",
        results_dir.parent / f"{image_name}.jpg",
        results_dir.parent / f"{image_name}.jpeg",
        results_dir.parent / f"{image_name}.bmp",
    ]
    
    original_image_path = None
    for candidate in original_candidates:
        if candidate.exists():
            original_image_path = candidate
            break
            
    if not original_image_path:
        # Try to find it in the results directory
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            candidate = results_dir / f"{image_name}{ext}"
            if candidate.exists():
                original_image_path = candidate
                break
                
    if not original_image_path:
        raise ValueError(f"Could not find original image for {image_name}")
        
    # Run analysis
    aggregator = DefectAggregator(results_dir, original_image_path)
    report = aggregator.run_complete_analysis()
    
    return report


def main():
    """Standalone execution for testing"""
    if len(sys.argv) < 3:
        print("Usage: python data_acquisition.py <results_directory> <original_image_name>")
        print("Example: python data_acquisition.py ./processing/results/fiber1 fiber1")
        sys.exit(1)
        
    results_dir = sys.argv[1]
    image_name = sys.argv[2]
    
    try:
        report = integrate_with_pipeline(results_dir, image_name)
        print(f"\nAnalysis complete!")
        print(f"Status: {report['pass_fail_status']}")
        print(f"Quality Score: {report['quality_score']}/100")
        print(f"Total Defects: {report['total_defects_found']}")
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
    
