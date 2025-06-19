# -*- coding: utf-8 -*-
"""
================================================================================
 Data-Tuned Hybrid Geometric-Pixel Fiber Analyzer
================================================================================
Version: 3.2 (Local Contrast & Debug Mode)
Author: Gemini
Date: 19 June 2024

Description:
This version addresses failures in the validation stage by implementing a more
robust local contrast validation method. It also introduces a crucial 'debug
mode' to visually diagnose why a particular image might be failing, providing
invaluable feedback for troubleshooting.

--------------------------------------------------------------------------------
Key Improvements in this Version:
--------------------------------------------------------------------------------
1.  **Local Contrast Validation:** The validation logic has been completely
    rewritten. Instead of comparing a circle's brightness to the global image
    average, it now compares the mean intensity *inside* the circle to the
    mean intensity in a "ring" immediately *outside* the circle. This is far
    more robust to variations in overall image lighting and contrast.

2.  **Interactive Debug Mode:** The user can now enable a debug mode at startup.
    If an image fails the new validation step, the script will save a
    diagnostic image (e.g., 'image_name_DEBUG.png'). This image shows the
    detected circle, the inner and outer areas used for contrast calculation,
    and the measured intensity values, making it clear why the rejection occurred.

3.  **Code Refinements:** The analysis pipeline is now more streamlined to
    support the new validation and debug features.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import glob

class TunedFiberAnalyzer:
    """
    Performs analysis on fiber optic images, with robust local validation.
    """
    # --- CLASS CONSTANTS (RE-TUNED FOR ROBUSTNESS) ---
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    MEDIAN_BLUR_KERNEL = 5
    HOUGH_DP = 1
    HOUGH_MIN_DIST = 100
    HOUGH_PARAM1 = 150
    HOUGH_PARAM2 = 25 # Further relaxed for maximum detection
    MIN_RADIUS = 15
    MAX_RADIUS = 150

    # --- NEW LOCAL CONTRAST VALIDATION PARAMETERS ---
    # The inner region must be this much brighter than the outer ring
    LOCAL_CONTRAST_FACTOR = 1.15
    # How wide the outer "ring" for comparison should be, in pixels
    OUTER_RING_WIDTH = 15

    def __init__(self, image_path: Path, debug_mode: bool = False):
        self.image_path = image_path
        self.debug_mode = debug_mode
        self.image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise IOError(f"Could not read image at {self.image_path}")

        self.output_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.center = None
        self.radius = None
        self.segmented_mask = None
        self.results = {}

    def run_full_pipeline(self) -> bool:
        """Executes the entire data-tuned analysis pipeline."""
        print(f"\nProcessing {self.image_path.name}...")

        processed_image = cv2.GaussianBlur(self.image, self.GAUSSIAN_BLUR_KERNEL, 0)
        processed_image = cv2.medianBlur(processed_image, self.MEDIAN_BLUR_KERNEL)

        circles = self._find_circles(processed_image)
        if circles is None:
            print("  - Failure: No circles detected.")
            return False

        for circle in circles:
            is_valid, debug_data = self._validate_with_local_contrast(circle)
            if is_valid:
                print(f"  + Success: Validated circle via local contrast at {self.center}, R={self.radius}px.")
                self._create_precise_mask()
                self._analyze_final_segment()
                return True
            else:
                # If validation fails, save a debug image if the mode is on
                if self.debug_mode:
                    self._save_debug_image(circle, debug_data)

        print(f"  - Failure: Found {len(circles)} circle(s), but none passed local contrast validation.")
        return False

    def _find_circles(self, image: np.ndarray) -> np.ndarray | None:
        """Detects circles using the Circular Hough Transform."""
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=self.HOUGH_DP, minDist=self.HOUGH_MIN_DIST,
            param1=self.HOUGH_PARAM1, param2=self.HOUGH_PARAM2,
            minRadius=self.MIN_RADIUS, maxRadius=self.MAX_RADIUS
        )
        return np.uint16(np.around(circles[0, :])) if circles is not None else None

    def _validate_with_local_contrast(self, circle: np.ndarray) -> tuple[bool, dict]:
        """
        Validates a circle by comparing its internal brightness to the brightness
        of its immediate outer surroundings.
        """
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

        # Define inner and outer masks
        inner_mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(inner_mask, (x, y), r, 255, -1)

        outer_mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), r + self.OUTER_RING_WIDTH, 255, -1)
        outer_mask = cv2.subtract(outer_mask, inner_mask)

        # Calculate mean intensities, ensuring we don't divide by zero
        inner_pixels = self.image[inner_mask > 0]
        outer_pixels = self.image[outer_mask > 0]

        if len(inner_pixels) == 0 or len(outer_pixels) == 0:
            return False, {}

        inner_mean = np.mean(inner_pixels)
        outer_mean = np.mean(outer_pixels)
        
        debug_data = {
            "inner_mean": inner_mean, "outer_mean": outer_mean,
            "inner_mask": inner_mask, "outer_mask": outer_mask
        }

        # Check if the inner area is sufficiently brighter than the outer ring
        if inner_mean > outer_mean * self.LOCAL_CONTRAST_FACTOR:
            self.center = (x, y)
            self.radius = r
            return True, debug_data
            
        return False, debug_data

    def _save_debug_image(self, circle: np.ndarray, debug_data: dict):
        """Saves a diagnostic image when validation fails."""
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        
        debug_img = self.output_image.copy()
        
        # Draw the inner and outer masks
        debug_img[debug_data['inner_mask'] > 0] = debug_img[debug_data['inner_mask'] > 0] * 0.5 + np.array([0, 128, 0], dtype=np.uint8) # Green tint for inner
        debug_img[debug_data['outer_mask'] > 0] = debug_img[debug_data['outer_mask'] > 0] * 0.5 + np.array([0, 0, 128], dtype=np.uint8) # Red tint for outer
        
        # Draw the rejected circle
        cv2.circle(debug_img, (x, y), r, (0, 255, 255), 2) # Yellow circle

        # Add text with debug info
        text1 = f"Inner Mean: {debug_data['inner_mean']:.2f}"
        text2 = f"Outer Mean: {debug_data['outer_mean']:.2f}"
        text3 = f"Threshold: {debug_data['outer_mean'] * self.LOCAL_CONTRAST_FACTOR:.2f}"
        cv2.putText(debug_img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_img, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        debug_path = self.image_path.parent / f"{self.image_path.stem}_DEBUG.png"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"  - INFO: Saved debug image to {debug_path}")

    def _create_precise_mask(self):
        """Generates a final segmentation mask using the validated circle as an ROI."""
        roi_mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(roi_mask, self.center, self.radius, 255, -1)
        roi_pixels = cv2.bitwise_and(self.image, self.image, mask=roi_mask)
        _, self.segmented_mask = cv2.threshold(
            roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    def _analyze_final_segment(self):
        """Computes statistics on the final segmented fiber area."""
        fiber_pixels = self.image[self.segmented_mask > 0]
        if len(fiber_pixels) == 0: return

        contours, _ = cv2.findContours(self.segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return
            
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * (area / (perimeter**2)) if perimeter > 0 else 0
        (x, y), effective_radius = cv2.minEnclosingCircle(c)

        self.results = {
            "Image": self.image_path.name, "Center_X_px": int(x), "Center_Y_px": int(y),
            "Effective_Radius_px": round(effective_radius, 2), "Area_px2": area,
            "Perimeter_px": round(perimeter, 2), "Circularity": round(circularity, 4),
            "Mean_Grayscale": round(np.mean(fiber_pixels), 2),
            "Std_Dev_Grayscale": round(np.std(fiber_pixels), 2),
        }

    def save_outputs(self, output_dir: Path):
        """Saves all analysis outputs."""
        if not self.results:
            print(f"  - Skipping save for {self.image_path.name} due to analysis error.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        contours, _ = cv2.findContours(self.segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.output_image, contours, -1, (255, 0, 0), 2)
        center_coords = (self.results['Center_X_px'], self.results['Center_Y_px'])
        radius_val = int(self.results['Effective_Radius_px'])
        cv2.circle(self.output_image, center_coords, radius_val, (0, 255, 0), 2)
        cv2.drawMarker(self.output_image, center_coords, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

        cv2.imwrite(str(output_dir / f"{self.image_path.stem}_analysis.png"), self.output_image)
        cv2.imwrite(str(output_dir / f"{self.image_path.stem}_mask.png"), self.segmented_mask)

        results_df = pd.DataFrame([self.results])
        results_csv_path = output_dir / "analysis_summary.csv"
        results_df.to_csv(results_csv_path, mode='a', header=not results_csv_path.exists(), index=False)
        print(f"  -> Outputs saved to '{output_dir.resolve()}'")

def main():
    """Main function to run the interactive analyzer."""
    print("="*80)
    print(" Data-Tuned Hybrid Fiber Analyzer v3.2 (Local Contrast) ".center(80))
    print("="*80)

    debug_choice = input("Enable DEBUG mode? (Saves diagnostic images on failure) [y/N]: ").strip().lower()
    DEBUG_MODE = True if debug_choice == 'y' else False
    if DEBUG_MODE:
        print("-> DEBUG mode is ON.")

    output_dir_str = input("Enter output directory (default: 'tuned_analysis_results'): ").strip()
    output_dir = Path(output_dir_str) if output_dir_str else Path("tuned_analysis_results")
    
    while True:
        try:
            path_str = input("\nEnter a path to an image OR a directory (or 'exit'): ").strip().strip('"')
            if path_str.lower() in ['exit', 'quit', 'q']: break
            
            input_path = Path(path_str)
            if not input_path.exists():
                print(f"Error: Path '{input_path}' does not exist.")
                continue

            image_paths = []
            if input_path.is_dir():
                print(f"Processing images in directory: {input_path}")
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif'):
                    image_paths.extend(input_path.glob(ext))
            elif input_path.is_file():
                image_paths.append(input_path)
            
            if not image_paths:
                print("Error: No valid image files found.")
                continue

            for img_path in image_paths:
                analyzer = TunedFiberAnalyzer(img_path, debug_mode=DEBUG_MODE)
                if analyzer.run_full_pipeline():
                    analyzer.save_outputs(output_dir)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting program.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
        
        print("\n" + "-"*80)

    print("\n" + "="*80)
    print(" Analysis complete. Goodbye! ".center(80))
    print("="*80)

if __name__ == "__main__":
    main()
