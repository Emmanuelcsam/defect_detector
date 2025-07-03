import cv2
import numpy as np
import os

def reimagine_image(image_path, output_folder="reimagined_images"):
    """
    Takes an image path and applies a wide range of OpenCV functions,
    saving each result to a specified folder.

    Args:
        image_path (str): The path to the input image.
        output_folder (str): The name of the folder to save the output images.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: '{output_folder}'")

    # Read the image
    img = cv2.imread(image_path)
    # Check if the image was loaded correctly
    if img is None:
        print(f"Error: Failed to load image from '{image_path}'. The file may be corrupt or not a supported image format.")
        return
        
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def save_image(name, image):
        # A helper function to save images to the output folder
        cv2.imwrite(os.path.join(output_folder, f"{name}.jpg"), image)

    print("Starting image transformations...")

    # --- Thresholding ---
    ret, thresh_binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    save_image("threshold_binary", thresh_binary)
    ret, thresh_binary_inv = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    save_image("threshold_binary_inv", thresh_binary_inv)
    ret, thresh_trunc = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TRUNC)
    save_image("threshold_trunc", thresh_trunc)
    ret, thresh_tozero = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TOZERO)
    save_image("threshold_tozero", thresh_tozero)
    ret, thresh_tozero_inv = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TOZERO_INV)
    save_image("threshold_tozero_inv", thresh_tozero_inv)
    adaptive_thresh_mean = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    save_image("adaptive_threshold_mean", adaptive_thresh_mean)
    adaptive_thresh_gaussian = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    save_image("adaptive_threshold_gaussian", adaptive_thresh_gaussian)
    ret, otsu_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_image("otsu_threshold", otsu_thresh)

    # --- Masking (using a circular mask) ---
    mask = np.zeros(img.shape[:2], dtype="uint8")
    (cX, cY) = (img.shape[1] // 2, img.shape[0] // 2)
    radius = int(min(cX, cY) * 0.8) # Use 80% of the smallest dimension for the radius
    cv2.circle(mask, (cX, cY), radius, 255, -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    save_image("masked_circle", masked_img)

    # --- Recolorizing ---
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    save_image("recolor_hsv", hsv_img)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    save_image("recolor_lab", lab_img)
    # Apply various colormaps to the grayscale image
    colormaps_to_apply = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET, cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN, cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_COOL, cv2.COLORMAP_HSV, cv2.COLORMAP_PINK, cv2.COLORMAP_HOT]
    colormap_names = ["AUTUMN", "BONE", "JET", "WINTER", "RAINBOW", "OCEAN", "SUMMER", "SPRING", "COOL", "HSV", "PINK", "HOT"]
    for name, colormap in zip(colormap_names, colormaps_to_apply):
        colormap_img = cv2.applyColorMap(gray_img, colormap)
        save_image(f"recolor_colormap_{name.lower()}", colormap_img)

    # --- Preprocessing ---
    # Blurring
    blurred = cv2.blur(img, (15, 15))
    save_image("preprocessing_blur", blurred)
    gaussian_blurred = cv2.GaussianBlur(img, (15, 15), 0)
    save_image("preprocessing_gaussian_blur", gaussian_blurred)
    median_blurred = cv2.medianBlur(img, 15)
    save_image("preprocessing_median_blur", median_blurred)
    bilateral_filtered = cv2.bilateralFilter(img, 15, 75, 75)
    save_image("preprocessing_bilateral_filter", bilateral_filtered)
    # Morphological
    kernel = np.ones((5,5),np.uint8)
    eroded = cv2.erode(thresh_binary, kernel, iterations = 1)
    save_image("preprocessing_erode", eroded)
    dilated = cv2.dilate(thresh_binary, kernel, iterations = 1)
    save_image("preprocessing_dilate", dilated)
    opening = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, kernel)
    save_image("preprocessing_opening", opening)
    closing = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)
    save_image("preprocessing_closing", closing)
    gradient = cv2.morphologyEx(thresh_binary, cv2.MORPH_GRADIENT, kernel)
    save_image("preprocessing_gradient", gradient)
    # Gradients
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    save_image("preprocessing_laplacian", np.uint8(np.absolute(laplacian)))
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    save_image("preprocessing_sobel_x", np.uint8(np.absolute(sobel_x)))
    save_image("preprocessing_sobel_y", np.uint8(np.absolute(sobel_y)))
    # Edge Detection
    canny_edges = cv2.Canny(gray_img, 100, 200)
    save_image("preprocessing_canny_edges", canny_edges)
    # Denoising
    denoised_color = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    save_image("preprocessing_denoised_color", denoised_color)
    # Histogram Equalization
    equalized_hist = cv2.equalizeHist(gray_img)
    save_image("preprocessing_equalized_hist", equalized_hist)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray_img)
    save_image("preprocessing_clahe", clahe_img)

    # --- Retexturing (Resizing Only) ---
    # Geometric Transformations
    (h, w) = img.shape[:2]
    resized_inter_nearest = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
    save_image("retexturing_resize_nearest", resized_inter_nearest)
    resized_inter_cubic = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    save_image("retexturing_resize_cubic", resized_inter_cubic)
    
    # --- Pixel Intensity and Binary Manipulation ---
    brighter = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
    save_image("intensity_brighter", brighter)
    darker = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)
    save_image("intensity_darker", darker)
    higher_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    save_image("intensity_higher_contrast", higher_contrast)
    lower_contrast = cv2.convertScaleAbs(img, alpha=0.7, beta=0)
    save_image("intensity_lower_contrast", lower_contrast)
    
    # Create a simple second image for bitwise operations
    img2 = np.zeros(img.shape, dtype="uint8")
    cv2.rectangle(img2, (w//4, h//4), (w*3//4, h*3//4), (255, 255, 255), -1)
    
    bitwise_and_op = cv2.bitwise_and(img, img2)
    save_image("binary_bitwise_and", bitwise_and_op)
    bitwise_or_op = cv2.bitwise_or(img, img2)
    save_image("binary_bitwise_or", bitwise_or_op)
    bitwise_xor_op = cv2.bitwise_xor(img, img2)
    save_image("binary_bitwise_xor", bitwise_xor_op)
    bitwise_not_op = cv2.bitwise_not(img)
    save_image("binary_bitwise_not", bitwise_not_op)

    print(f"\nProcessing complete!")
    print(f"All {len(os.listdir(output_folder))} reimagined images have been saved in the '{output_folder}' directory.")
    
    # Return list of created files
    created_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')]
    return created_files

if __name__ == '__main__':
    # Ask the user for the image path
    image_to_process = input("Please enter the path to the image you want to reimagine: ")
    
    # Call the function with the user-provided path
    reimagine_image(image_to_process)