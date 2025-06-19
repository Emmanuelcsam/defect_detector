# -*- coding: utf-8 -*-
"""
================================================================================
 Data-Tuned Hybrid Geometric-Pixel Fiber Analyzer
================================================================================
Version: 3.1 (Robust Edition)
Author: Gemini
Date: 19 June 2024

Description:
This version incorporates feedback from real-world testing to improve robustness
and user experience. The detection parameters have been relaxed to handle a
wider variety of images, and the script now supports processing entire
directories of images.

--------------------------------------------------------------------------------
Key Improvements in this version:
--------------------------------------------------------------------------------
1.  **More Forgiving Detection Parameters:** The thresholds for the Hough Circle
    Transform have been relaxed. Specifically, the accumulator threshold
    (HOUGH_PARAM2) has been lowered, making the detector less strict and more
    likely to find circles in images with minor imperfections.

2.  **Directory Processing:** The script now accepts a path to a directory.
    It will automatically find and process all common image files (.jpg, .jpeg,
    .png) within that directory, greatly improving workflow efficiency.

3.  **Improved Preprocessing:** A Gaussian blur has been added to the
    preprocessing pipeline. This helps to smooth the image and reduce noise,
    leading to more stable and reliable circle detection.

4.  **Enhanced Path Handling & User Feedback:** The command-line input logic
    is now more robust and provides clearer error messages if a path is neither
    a valid file nor a directory.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shlex
import glob

class TunedFiberAnalyzer:
    """
    Performs analysis on fiber optic images, with robust parameters.
    """
    # --- CLASS CONSTANTS (RE-TUNED FOR ROBUSTNESS) ---
    # Preprocessing
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    MEDIAN_BLUR_KERNEL = 5

    # Hough Circle Transform parameters - Made more lenient
    HOUGH_DP = 1
    HOUGH_MIN_DIST = 100
    HOUGH_PARAM1 = 150    # Upper threshold for the internal Canny.
    HOUGH_PARAM2 = 30     # Accumulator threshold. Lowered from 45 to be less strict.
    MIN_RADIUS = 15
    MAX_RADIUS = 120      # Increased max radius slightly for more flexibility

    # Validation threshold
    INTENSITY_VALIDATION_FACTOR = 1.2

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
        """
        print(f"\nProcessing {self.image_path.name}...")

        # 1. Preprocessing: Gaussian + Median blur for robust noise reduction
        blurred_image = cv2.GaussianBlur(self.image, self.GAUSSIAN_BLUR_KERNEL, 0)
        processed_image = cv2.medianBlur(blurred_image, self.MEDIAN_BLUR_KERNEL)

        # 2. Find promising circles with the more lenient Hough Transform
        circles = self._find_circles(processed_image)
        if circles is None:
            print("  - Failure: No circles detected. Try adjusting Hough parameters if this persists.")
            return False

        # 3. Validate the best circle based on our intensity profile heuristic
        #    We iterate through found circles in case the "best" one is not the fiber
        for circle in circles:
            if self._validate_circle_by_intensity(circle):
                print(f"  + Success: Validated bright-core circle found at {self.center}, R={self.radius}px.")
                # 4. Generate a high-precision segmentation mask from the validated circle
                self._create_precise_mask()
                print("  + Success: High-precision segmentation mask generated.")

                # 5. Analyze the final segmented area and store results
                self._analyze_final_segment()
                print("  + Success: Final measurements calculated.")
                return True
        
        print(f"  - Failure: Found {len(circles)} circle(s), but none contained a valid bright core.")
        return False


    def _find_circles(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detects circles using the Circular Hough Transform with robust parameters.
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
        """
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        
        # Create a temporary mask for the circle's interior
        mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Ensure there are pixels in the mask to avoid division by zero
        if np.sum(mask) == 0:
            return False

        mean_intensity_inside = cv2.mean(self.image, mask=mask)[0]
        mean_intensity_overall = np.mean(self.image)

        if mean_intensity_inside > mean_intensity_overall * self.INTENSITY_VALIDATION_FACTOR:
            # If valid, set the instance's center and radius
            self.center = (x, y)
            self.radius = r
            return True
        return False

    def _create_precise_mask(self):
        """
        Generates a final, high-precision segmentation mask using the validated
        circle as a Region of Interest (ROI).
        """
        roi_mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(roi_mask, self.center, self.radius, 255, -1)

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
            self.results = {"Error": "No pixels in final mask"}
            return

        contours, _ = cv2.findContours(self.segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.results = {"Error": "No contours found in final mask"}
            return
            
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
        """Saves all analysis outputs to the specified directory."""
        if not self.results or "Error" in self.results:
            print(f"  - Skipping save for {self.image_path.name} due to analysis error.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Draw final segmented contour (Blue) and enclosing circle (Green)
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
    print(" Data-Tuned Hybrid Fiber Analyzer v3.1 (Robust Edition) ".center(80))
    print("="*80)

    output_dir_str = input("Enter output directory (default: 'tuned_analysis_results'): ").strip()
    output_dir = Path(output_dir_str) if output_dir_str else Path("tuned_analysis_results")
    
    while True:
        try:
            path_str = input("\nEnter a path to an image OR a directory of images (or 'exit'): ").strip()
            if path_str.lower() in ['exit', 'quit', 'q']: break
            
            # Remove quotes that might be wrapped around the path
            if path_str.startswith('"') and path_str.endswith('"'):
                path_str = path_str[1:-1]
            
            input_path = Path(path_str)
            image_paths = []

            if not input_path.exists():
                print(f"Error: The path '{input_path}' does not exist. Please check the path and try again.")
                continue

            if input_path.is_dir():
                print(f"Processing all images in directory: {input_path}")
                for ext in ('*.jpg', '*.jpeg', '*.png'):
                    image_paths.extend(input_path.glob(ext))
            elif input_path.is_file():
                image_paths.append(input_path)
            
            if not image_paths:
                print("Error: No valid image files found at the specified path. Please provide a path to a .jpg, .jpeg, or .png file, or a directory containing them.")
                continue

            for img_path in image_paths:
                analyzer = TunedFiberAnalyzer(img_path)
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
