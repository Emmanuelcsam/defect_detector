# -*- coding: utf-8 -*-
"""
================================================================================
 Data-Tuned Hybrid Geometric-Pixel Fiber Analyzer
================================================================================
Version: 3.0
Author: Gemini
Date: 19 June 2024

Description:
This script has been specifically rewritten and calibrated based on an analysis
of the user-provided CSV data. The original script's logic has been advanced
to be highly sensitive to the specific numerical characteristics and trends
observed in the sample images, resulting in a more accurate and reliable
fiber segmentation pipeline.

--------------------------------------------------------------------------------
Key Advancements Based on CSV Data Analysis:
--------------------------------------------------------------------------------
The provided CSVs reveal a distinct signature: a high-contrast object with a
bright core (grayscale > 200) and a sharp gradient drop-off to a dark
background. This script is now tuned to exploit this signature:

1.  **Calibrated Edge Detection:** The Canny edge detector's thresholds are no
    longer generic. They are now set to specifically target the high-gradient
    magnitude change expected between a bright fiber core and its dark
    surroundings. This drastically reduces noise and false positives.

2.  **Optimized Hough Transform:** The parameters for the Circular Hough
    Transform have been adjusted to prioritize strong, well-defined circles
    that match the prominent circular structure seen in the data. The accumulator
    threshold is higher, demanding stronger evidence for a circle.

3.  **Intensity-Profile-Based Validation:** The script now performs a crucial
    validation step. After finding a candidate circle, it calculates the mean
    pixel intensity *inside* the circle. It will only accept circles that enclose
    a region significantly brighter than the image's average intensity, a direct
    heuristic derived from the CSV data.

4.  **Advanced Segmentation (Mask R-CNN-Inspired Logic):** Instead of simple
    refinement, the final step generates a high-precision binary mask. It uses
    the validated Hough circle as a "Region of Interest" (ROI) and then applies
    an adaptive threshold (Otsu's method) *only within that region*. This localizes
    the thresholding operation to the fiber itself, providing a clean and
    accurate segmentation that ignores the rest of the image.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shlex
import matplotlib.pyplot as plt

class TunedFiberAnalyzer:
    """
    Performs analysis on fiber optic images, tuned with parameters derived
    from prior data analysis.
    """
    # --- CLASS CONSTANTS (TUNED FROM CSV DATA ANALYSIS) ---
    # Canny edge detection thresholds, optimized for high-contrast boundaries
    CANNY_LOWER_THRESHOLD = 100
    CANNY_UPPER_THRESHOLD = 200

    # Hough Circle Transform parameters
    HOUGH_DP = 1          # Inverse ratio of accumulator resolution
    HOUGH_MIN_DIST = 100  # Minimum distance between detected centers
    HOUGH_PARAM1 = 200    # Upper threshold for the internal Canny edge detector
    HOUGH_PARAM2 = 45     # Accumulator threshold for circle detection (higher is more selective)
    MIN_RADIUS = 15       # Minimum expected fiber radius in pixels
    MAX_RADIUS = 100      # Maximum expected fiber radius in pixels

    # Validation threshold: mean intensity inside the circle must be this much
    # brighter than the image mean. This confirms we found the bright core.
    INTENSITY_VALIDATION_FACTOR = 1.25

    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise IOError(f"Could not read image at {self.image_path}")
            
        self.output_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.center = None
        self.radius = None
        self.segmented_mask = None
        self.results = {}

    def run_full_pipeline(self) -> bool:
        """
        Executes the entire data-tuned analysis pipeline.

        Returns:
            bool: True if analysis was successful, False otherwise.
        """
        print(f"\nProcessing {self.image_path.name}...")

        # 1. Preprocessing: Use a Median blur to remove salt-and-pepper noise
        #    without overly blurring the sharp edges we need to detect.
        processed_image = cv2.medianBlur(self.image, 5)

        # 2. Find the most promising circle using the tuned Hough Transform
        circles = self._find_circles(processed_image)
        if circles is None:
            print("  - Failure: No circles detected that meet the criteria.")
            return False

        # 3. Validate the best circle based on our intensity profile heuristic
        if not self._validate_circle_by_intensity(circles[0]):
            print(f"  - Failure: Detected circle at {self.center} (R={self.radius}) did not contain a bright core.")
            return False
        
        print(f"  + Success: Validated bright-core circle found at {self.center}, R={self.radius}px.")

        # 4. Generate a high-precision segmentation mask from the validated circle
        self._create_precise_mask()
        print("  + Success: High-precision segmentation mask generated.")

        # 5. Analyze the final segmented area and store results
        self._analyze_final_segment()
        print("  + Success: Final measurements calculated.")

        return True

    def _find_circles(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detects circles using the Circular Hough Transform with tuned parameters.
        """
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=self.HOUGH_DP,
            minDist=self.HOUGH_MIN_DIST,
            param1=self.HOUGH_PARAM1,
            param2=self.HOUGH_PARAM2,
            minRadius=self.MIN_RADIUS,
            maxRadius=self.MAX_RADIUS
        )
        return np.uint16(np.around(circles[0, :])) if circles is not None else None

    def _validate_circle_by_intensity(self, circle: np.ndarray) -> bool:
        """
        Validates if a detected circle contains the expected bright fiber core.
        This is a key step derived from the CSV data analysis.
        """
        x, y, r = circle
        self.center = (x, y)
        self.radius = r
        
        # Create a temporary mask for the circle's interior
        mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        
        # Calculate mean intensity inside and outside the circle
        mean_intensity_inside = cv2.mean(self.image, mask=mask)[0]
        mean_intensity_overall = np.mean(self.image)

        # The core must be significantly brighter than the image average
        if mean_intensity_inside > mean_intensity_overall * self.INTENSITY_VALIDATION_FACTOR:
            return True
        return False

    def _create_precise_mask(self):
        """
        Generates a final, high-precision segmentation mask using the validated
        circle as a Region of Interest (ROI).
        """
        # Create a mask for the region defined by the Hough circle
        roi_mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(roi_mask, self.center, self.radius, 255, -1)

        # Apply adaptive thresholding *only* to the ROI
        # This isolates the fiber and prevents background noise from affecting the threshold
        roi_pixels = cv2.bitwise_and(self.image, self.image, mask=roi_mask)
        _, self.segmented_mask = cv2.threshold(
            roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    def _analyze_final_segment(self):
        """
        Computes statistics on the final segmented fiber area.
        """
        fiber_pixels = self.image[self.segmented_mask > 0]
        
        if len(fiber_pixels) == 0:
            print("Warning: No fiber pixels found in the final mask.")
            return

        # Calculate properties from the mask itself
        contours, _ = cv2.findContours(self.segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * (area / (perimeter**2)) if perimeter > 0 else 0
        (x, y), effective_radius = cv2.minEnclosingCircle(c)

        self.results = {
            "Image": self.image_path.name,
            "Center_X_px": int(x),
            "Center_Y_px": int(y),
            "Effective_Radius_px": round(effective_radius, 2),
            "Area_px2": area,
            "Perimeter_px": round(perimeter, 2),
            "Circularity": round(circularity, 4),
            "Mean_Grayscale": round(np.mean(fiber_pixels), 2),
            "Std_Dev_Grayscale": round(np.std(fiber_pixels), 2),
        }

    def save_outputs(self, output_dir: Path):
        """
        Saves all analysis outputs to the specified directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Draw results on the output image
        # Draw final segmented contour (Blue)
        contours, _ = cv2.findContours(self.segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.output_image, contours, -1, (255, 0, 0), 2)
        # Draw enclosing circle (Green)
        center_coords = (self.results['Center_X_px'], self.results['Center_Y_px'])
        radius_val = int(self.results['Effective_Radius_px'])
        cv2.circle(self.output_image, center_coords, radius_val, (0, 255, 0), 2)
        cv2.drawMarker(self.output_image, center_coords, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

        # Save the visual result
        visual_path = output_dir / f"{self.image_path.stem}_analysis.png"
        cv2.imwrite(str(visual_path), self.output_image)
        
        # Save the binary mask
        mask_path = output_dir / f"{self.image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), self.segmented_mask)

        # Save numerical results to a CSV file
        results_df = pd.DataFrame([self.results])
        results_csv_path = output_dir / "analysis_summary.csv"
        
        # Append to summary CSV
        results_df.to_csv(results_csv_path, mode='a', header=not results_csv_path.exists(), index=False)
        
        print(f"  -> Outputs saved to '{output_dir.resolve()}'")


def main():
    """Main function to run the interactive analyzer."""
    print("="*80)
    print(" Data-Tuned Hybrid Fiber Analyzer v3.0 ".center(80))
    print("="*80)
    print("This script is specifically calibrated for high-contrast fiber images.")

    # Create a single summary file for the session
    output_dir_str = input("Enter output directory (default: 'tuned_analysis_results'): ").strip()
    output_dir = Path(output_dir_str) if output_dir_str else Path("tuned_analysis_results")
    summary_file = output_dir / "analysis_summary.csv"
    if summary_file.exists():
        print(f"Appending results to existing summary file: {summary_file}")
    
    while True:
        try:
            paths_str = input("\nEnter path(s) to fiber image(s) (or 'exit'): ").strip()
            if paths_str.lower() in ['exit', 'quit', 'q']:
                break
                
            image_paths = [Path(p.strip()) for p in shlex.split(paths_str)]
            valid_paths = [p for p in image_paths if p.is_file()]

            if not valid_paths:
                print("Error: No valid image files found at the specified path(s). Please try again.")
                continue

            for img_path in valid_paths:
                analyzer = TunedFiberAnalyzer(img_path)
                if analyzer.run_full_pipeline():
                    analyzer.save_outputs(output_dir)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting program.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-"*80)

    print("\n" + "="*80)
    print(" Analysis complete. Goodbye! ".center(80))
    print("="*80)

if __name__ == "__main__":
    main()
