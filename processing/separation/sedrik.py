import cv2
import numpy as np
import os
from scipy.signal import find_peaks

def adaptive_segment_image(image_path, peak_prominence=1000, output_dir="adaptive_segmented_regions"):
    """
    Automatically segments an image by finding the most prominent intensity peaks
    in its histogram and creating separate images for each region.

    Args:
        image_path (str): The full path to the input grayscale image.
        peak_prominence (int, optional): A parameter that controls the sensitivity
                                         of peak detection. A higher value will
                                         find only the most significant peaks.
                                         Defaults to 1000.
        output_dir (str, optional): The directory where output images will be saved.
                                    Defaults to "adaptive_segmented_regions".

    Returns:
        None. Images are saved to the specified directory.
    """
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Error: Could not read the image from '{image_path}'. Check the file format.")
        return

    histogram = cv2.calcHist([original_image], [0], None, [256], [0, 256])
    histogram = histogram.flatten()

    peaks, properties = find_peaks(histogram, prominence=peak_prominence, width=(None, None), rel_height=1.0)

    if len(peaks) == 0:
        print(f"No significant intensity peaks found with prominence={peak_prominence}. Try lowering the value.")
        return

    # --- FIX APPLIED HERE ---
    # We convert the peak boundaries to standard Python integers immediately after detection.
    # The list comprehension `[int(x) for x in ...]` ensures each value is a standard int.
    left_bases = [int(x) for x in properties['left_ips']]
    right_bases = [int(x) for x in properties['right_ips']]
    
    intensity_ranges = list(zip(left_bases, right_bases))

    print(f"Found {len(peaks)} significant intensity regions with prominence > {peak_prominence}.")
    print("Detected Ranges:", intensity_ranges)
    
    os.makedirs(output_dir, exist_ok=True)

    for i, (min_val, max_val) in enumerate(intensity_ranges):
        peak_intensity = peaks[i]
        print(f"Processing region {i+1}: Centered at intensity {peak_intensity} (Range: {min_val}-{max_val})...")

        # Now, min_val and max_val are standard integers, which cv2.inRange will accept.
        mask = cv2.inRange(original_image, min_val, max_val)
        
        segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)

        output_filename = f"region_{i+1}_peak_{peak_intensity}_range_{min_val}-{max_val}.png"
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, segmented_image)
        print(f"-> Saved segmented image to '{output_path}'")

    print("\nProcessing complete.")


if __name__ == '__main__':
    input_image_file = r"C:\Users\Saem1001\Documents\GitHub\IPPS\processing\output\img (210)_intensity_map.png"
    prominence_threshold = 1000
    adaptive_segment_image(input_image_file, peak_prominence=prominence_threshold)