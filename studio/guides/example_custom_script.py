"""
Fiber Optic Core Analysis Script
This example shows how to create a complete image processing function for the GUI
"""
import cv2
import numpy as np

def process_image(image: np.ndarray, 
                  analysis_mode: str = "intensity",
                  core_threshold: int = 180,
                  highlight_core: bool = True,
                  overlay_alpha: float = 0.5) -> np.ndarray:
    """
    Analyze fiber optic images with multiple visualization modes.
    
    This function demonstrates:
    - Multiple parameter types for automatic UI generation
    - Proper grayscale/color handling
    - Different analysis modes
    - Error handling
    
    Args:
        image: Input fiber optic image
        analysis_mode: Type of analysis ("intensity", "gradient", "profile")
        core_threshold: Brightness threshold for core detection (0-255)
        highlight_core: Whether to highlight the detected core region
        overlay_alpha: Transparency of overlay (0.0-1.0)
        
    Returns:
        Processed image with analysis visualization
    """
    try:
        # Validate parameters
        overlay_alpha = max(0.0, min(1.0, overlay_alpha))
        core_threshold = max(0, min(255, core_threshold))
        
        # Ensure we have a grayscale image for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Keep color for output
            result = image.copy()
        else:
            gray = image.copy()
            # Convert to color for visualization
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Get image dimensions
        h, w = gray.shape
        center = (w // 2, h // 2)
        
        # Perform analysis based on mode
        if analysis_mode.lower() == "intensity":
            # Create intensity heatmap
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            result = cv2.addWeighted(result, 1-overlay_alpha, heatmap, overlay_alpha, 0)
            
        elif analysis_mode.lower() == "gradient":
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x**2 + grad_y**2)
            gradient = np.clip(gradient, 0, 255).astype(np.uint8)
            
            # Visualize gradient
            gradient_color = cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
            result = cv2.addWeighted(result, 1-overlay_alpha, gradient_color, overlay_alpha, 0)
            
        elif analysis_mode.lower() == "profile":
            # Create radial profile visualization
            # Calculate distance from center for each pixel
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            
            # Normalize distances
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            dist_normalized = (dist_from_center / max_dist * 255).astype(np.uint8)
            
            # Create radial gradient visualization
            radial_color = cv2.applyColorMap(dist_normalized, cv2.COLORMAP_PLASMA)
            result = cv2.addWeighted(result, 1-overlay_alpha, radial_color, overlay_alpha, 0)
            
        # Highlight core region if requested
        if highlight_core:
            # Detect bright core region
            _, core_mask = cv2.threshold(gray, core_threshold, 255, cv2.THRESH_BINARY)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find core contour
            contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (assumed to be the core)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Draw core boundary
                cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
                
                # Calculate and display core properties
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)
                    
                    # Add text annotation
                    area = cv2.contourArea(largest_contour)
                    text = f"Core Area: {int(area)} px"
                    cv2.putText(result, text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add mode indicator
        mode_text = f"Mode: {analysis_mode.capitalize()}"
        cv2.putText(result, mode_text, (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
        
    except Exception as e:
        print(f"Error in fiber analysis: {e}")
        # Return original image on error
        if len(image.shape) == 3:
            return image
        else:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


# Optional: Test code for standalone execution
if __name__ == "__main__":
    # This section only runs when the script is executed directly
    # It won't run when imported by the GUI
    
    print("Testing fiber analysis script...")
    
    # Create a synthetic test image (circular gradient)
    test_size = 300
    center = test_size // 2
    y, x = np.ogrid[:test_size, :test_size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Create fiber-like pattern
    max_dist = center
    intensity = 255 * np.exp(-(dist**2) / (2 * (max_dist/3)**2))
    test_image = intensity.astype(np.uint8)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Test different modes
    modes = ["intensity", "gradient", "profile"]
    
    for mode in modes:
        result = process_image(test_image, analysis_mode=mode)
        cv2.imshow(f"Fiber Analysis - {mode}", result)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Test with a real image if available
    real_image = cv2.imread("fiber_image.jpg")
    if real_image is not None:
        result = process_image(real_image, analysis_mode="intensity", 
                             core_threshold=200, highlight_core=True)
        cv2.imshow("Real Fiber Analysis", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
