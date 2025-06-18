#! python3
"""
Command-line tool for circle image processing
Usage: python cli_tool.py <image_path> [--display] [--save] [--output-dir <dir>]
"""

import cv2
import argparse
from pathlib import Path
from circle_detector import inner_outer_split
from split_to_mask import split_circle
from visualizer import display_results, draw_circles

def process_image(img_path, display=False, save=False, output_dir='output'):
    """Process a single circle image"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot read {img_path}")
        return False
    
    # Detect circles
    inner, outer = inner_outer_split(img)
    if inner is None:
        print(f"Failed to detect circles in {img_path}")
        return False
    
    # Split image
    inner_img, ring_img = split_circle(img, inner, outer)
    # ensure non-None for type-check and runtime safety
    assert inner_img is not None and ring_img is not None, f"Error: failed to split {img_path}"
    
    # Display if requested
    if display:
        display_results(img, inner_img, ring_img, inner, outer)
    
    # Save if requested
    if save:
        Path(output_dir).mkdir(exist_ok=True)
        base = Path(img_path).stem
        
        cv2.imwrite(f'{output_dir}/{base}_inner.png', inner_img)
        cv2.imwrite(f'{output_dir}/{base}_ring.png', ring_img)
        
        vis = draw_circles(img, inner, outer)
        cv2.imwrite(f'{output_dir}/{base}_vis.png', vis)
        
        print(f"Saved: {base}_inner.png, {base}_ring.png, {base}_vis.png")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Split circle images')
    parser.add_argument('image', help='Path to circle image')
    parser.add_argument('-d', '--display', action='store_true', help='Display results')
    parser.add_argument('-s', '--save', action='store_true', help='Save results')
    parser.add_argument('-o', '--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Default to save if nothing specified
    if not args.display and not args.save:
        args.save = True
    
    process_image(args.image, args.display, args.save, args.output_dir)

if __name__ == "__main__":
    main()