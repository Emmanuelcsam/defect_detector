#!/usr/bin/env python3
"""
Create a clear, final visualization showing all detected defects on the original image
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_defect_overlay(image_path, detection_report_path, output_path):
    """Create a clear visualization with all defects marked on the original image"""
    
    # Load original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load detection report
    with open(detection_report_path, 'r') as f:
        report = json.load(f)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Original image
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=16)
    ax1.axis('off')
    
    # Right: Image with defect annotations
    ax2.imshow(img_rgb)
    ax2.set_title(f'Detected Defects ({len(report["defects"])} total)', fontsize=16)
    ax2.axis('off')
    
    # Define colors for defect types
    defect_colors = {
        'SCRATCH': 'red',
        'CRACK': 'darkred',
        'DIG': 'blue',
        'PIT': 'darkblue',
        'CONTAMINATION': 'yellow',
        'CHIP': 'orange',
        'BUBBLE': 'cyan',
        'BURN': 'magenta',
        'ANOMALY': 'purple',
        'UNKNOWN': 'gray'
    }
    
    # Draw each defect
    for i, defect in enumerate(report['defects']):
        defect_type = defect.get('defect_type', 'UNKNOWN')
        color = defect_colors.get(defect_type, 'gray')
        
        # Get location
        if 'location_xy' in defect:
            x, y = defect['location_xy']
        elif 'location' in defect:
            x = defect['location'].get('x', 0)
            y = defect['location'].get('y', 0)
        else:
            continue
        
        # Get bounding box
        if 'bbox' in defect:
            bbox_x, bbox_y, bbox_w, bbox_h = defect['bbox']
            # Draw rectangle
            rect = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h,
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
        
        # Draw center point
        circle = patches.Circle((x, y), radius=5, color=color, alpha=0.8)
        ax2.add_patch(circle)
        
        # Add defect ID
        defect_id = defect.get('defect_id', f'D{i+1}')
        ax2.text(x + 10, y - 10, defect_id, color=color, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Add legend
    legend_elements = []
    defect_types = set(d.get('defect_type', 'UNKNOWN') for d in report['defects'])
    for defect_type in sorted(defect_types):
        color = defect_colors.get(defect_type, 'gray')
        count = sum(1 for d in report['defects'] if d.get('defect_type') == defect_type)
        legend_elements.append(patches.Patch(color=color, label=f'{defect_type} ({count})'))
    
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add summary text
    quality_score = report.get('overall_quality_score', 'N/A')
    summary_text = f"Quality Score: {quality_score}/100\nTotal Defects: {len(report['defects'])}"
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Also create a simple overlay version
    overlay_img = img.copy()
    
    for defect in report['defects']:
        defect_type = defect.get('defect_type', 'UNKNOWN')
        
        # BGR colors for OpenCV
        cv_colors = {
            'SCRATCH': (0, 0, 255),      # Red
            'DIG': (255, 0, 0),           # Blue
            'CONTAMINATION': (0, 255, 255), # Yellow
            'ANOMALY': (255, 0, 255),     # Purple
            'UNKNOWN': (128, 128, 128)    # Gray
        }
        color = cv_colors.get(defect_type, (128, 128, 128))
        
        if 'bbox' in defect:
            x, y, w, h = defect['bbox']
            cv2.rectangle(overlay_img, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"{defect.get('defect_id', 'D')} - {defect_type}"
            cv2.putText(overlay_img, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save OpenCV version
    cv_output = output_path.replace('.png', '_opencv.png')
    cv2.imwrite(cv_output, overlay_img)
    print(f"Saved OpenCV overlay to: {cv_output}")
    
    return len(report['defects'])

def create_aggregated_visualization(final_report_path, original_image_path, output_path):
    """Create visualization from aggregated final report"""
    
    # Load original image
    img = cv2.imread(original_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load final report
    with open(final_report_path, 'r') as f:
        report = json.load(f)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Main image
    ax_main = plt.subplot(2, 2, (1, 3))
    ax_main.imshow(img_rgb)
    total_defects = report.get('total_defects', report['analysis_summary']['total_merged_defects'])
    ax_main.set_title(f'Final Defect Analysis - {total_defects} Merged Defects', fontsize=16)
    ax_main.axis('off')
    
    # Define colors
    severity_colors = {
        'CRITICAL': 'darkred',
        'HIGH': 'red',
        'MEDIUM': 'orange',
        'LOW': 'yellow',
        'NEGLIGIBLE': 'green'
    }
    
    # Draw merged defects
    for i, defect in enumerate(report.get('defects', [])):
        severity = defect.get('severity', 'LOW')
        color = severity_colors.get(severity, 'gray')
        
        # Get location
        loc = defect.get('global_location', defect.get('location', {}))
        if isinstance(loc, dict):
            x, y = loc.get('x', 0), loc.get('y', 0)
        else:
            x, y = loc[0], loc[1]
        
        # Draw circle with size based on area
        area = defect.get('area_px', 100)
        radius = max(10, min(50, np.sqrt(area) / 2))
        circle = patches.Circle((x, y), radius=radius, color=color, alpha=0.6)
        ax_main.add_patch(circle)
        
        # Add ID
        ax_main.text(x, y, str(i+1), ha='center', va='center', fontsize=10, 
                    color='black', weight='bold')
    
    # Statistics panel
    ax_stats = plt.subplot(2, 2, 2)
    ax_stats.axis('off')
    
    stats_text = f"""Analysis Summary:
    
Total Raw Defects: {report['analysis_summary']['total_raw_defects']}
Merged Defects: {report['analysis_summary']['total_merged_defects']}
Clustering Reduction: {report['analysis_summary']['clustering_reduction']}

Quality Score: {report['analysis_summary']['quality_score']}/100
Status: {report['analysis_summary']['pass_fail_status']}

Defect Types:
"""
    
    for dtype, count in report['defect_statistics']['by_type'].items():
        stats_text += f"  • {dtype}: {count}\n"
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                 fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    # Severity distribution
    ax_sev = plt.subplot(2, 2, 4)
    severities = list(report['defect_statistics']['by_severity'].keys())
    counts = list(report['defect_statistics']['by_severity'].values())
    colors = [severity_colors.get(s, 'gray') for s in severities]
    
    ax_sev.bar(severities, counts, color=colors)
    ax_sev.set_title('Defect Severity Distribution')
    ax_sev.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved aggregated visualization to: {output_path}")

if __name__ == "__main__":
    print("Creating final defect visualizations...")
    
    # 1. Create visualization from detection report
    if Path("test_viz_check/detection/img63_report.json").exists():
        defect_count = create_defect_overlay(
            "img63.jpg",
            "test_viz_check/detection/img63_report.json",
            "final_defect_visualization.png"
        )
        print(f"\n✓ Created detection visualization with {defect_count} defects")
    
    # 2. Create visualization from aggregated report
    if Path("test_viz_check/4_final_analysis/img63_final_report.json").exists():
        create_aggregated_visualization(
            "test_viz_check/4_final_analysis/img63_final_report.json",
            "img63.jpg",
            "final_aggregated_visualization.png"
        )
        print("\n✓ Created aggregated visualization")
    
    print("\nVisualization files created:")
    print("- final_defect_visualization.png (matplotlib version with annotations)")
    print("- final_defect_visualization_opencv.png (OpenCV overlay)")
    print("- final_aggregated_visualization.png (merged defects with statistics)")