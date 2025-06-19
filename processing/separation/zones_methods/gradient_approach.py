import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize
from skimage.feature import canny
from skimage.morphology import remove_small_objects
import warnings
warnings.filterwarnings('ignore')


class FiberOpticSegmenter:
    """
    Universal fiber optic endface segmentation that doesn't rely on Hough circles.
    Uses multiple robust methods to ensure accurate segmentation across different images.
    """
    
    def __init__(self, image_path, output_dir='output_universal'):
        self.image_path = image_path
        self.output_dir = output_dir
        self.original = None
        self.gray = None
        self.enhanced = None
        self.center = None
        self.core_radius = None
        self.cladding_radius = None
        
    def load_and_preprocess(self):
        """Load image and apply preprocessing"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
            
        self.original = cv2.imread(self.image_path)
        if self.original is None:
            raise ValueError(f"Could not read image: {self.image_path}")
            
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.enhanced = clahe.apply(self.gray)
        
        print(f"Loaded image: {self.image_path}")
        print(f"Dimensions: {self.gray.shape}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def find_center_multi_method(self):
        """Find center using multiple methods and combine results"""
        centers = []
        weights = []
        
        # Method 1: Brightness-weighted centroid
        center1, conf1 = self._brightness_centroid()
        if center1 is not None:
            centers.append(center1)
            weights.append(conf1)
            
        # Method 2: Circular gradient optimization
        center2, conf2 = self._gradient_center()
        if center2 is not None:
            centers.append(center2)
            weights.append(conf2)
            
        # Method 3: Morphological center
        center3, conf3 = self._morphological_center()
        if center3 is not None:
            centers.append(center3)
            weights.append(conf3)
            
        # Method 4: Edge-based center estimation
        center4, conf4 = self._edge_based_center()
        if center4 is not None:
            centers.append(center4)
            weights.append(conf4)
        
        if not centers:
            # Fallback to image center
            h, w = self.gray.shape
            self.center = (w//2, h//2)
            print("Warning: Using image center as fallback")
        else:
            # Weighted average of all detected centers
            centers = np.array(centers)
            weights = np.array(weights)
            weights = weights / weights.sum()
            self.center = np.average(centers, axis=0, weights=weights).astype(int)
            
        print(f"Final center: {self.center}")
        
    def _brightness_centroid(self):
        """Find center using brightness-weighted centroid"""
        # Use top percentile of brightness
        threshold = np.percentile(self.enhanced, 85)
        bright_mask = self.enhanced > threshold
        
        # Remove small objects
        bright_mask = remove_small_objects(bright_mask, min_size=100)
        
        if np.sum(bright_mask) == 0:
            return None, 0
            
        # Calculate centroid
        y_coords, x_coords = np.where(bright_mask)
        weights = self.enhanced[bright_mask]
        
        cx = np.average(x_coords, weights=weights)
        cy = np.average(y_coords, weights=weights)
        
        # Confidence based on compactness of bright region
        if len(x_coords) > 0:
            spread = np.std(np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2))
            confidence = 1.0 / (1.0 + spread/100)
        else:
            confidence = 0
            
        return (int(cx), int(cy)), confidence
    
    def _gradient_center(self):
        """Find center by optimizing radial gradient alignment"""
        # Calculate gradients
        grad_x = cv2.Sobel(self.enhanced, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(self.enhanced, cv2.CV_64F, 0, 1, ksize=5)
        
        # Initial guess from brightness
        h, w = self.gray.shape
        initial_guess = [w//2, h//2]
        
        bright_center = self._brightness_centroid()
        if bright_center[0] is not None:
            initial_guess = list(bright_center[0])
        
        def objective(center):
            cx, cy = center
            y_grid, x_grid = np.ogrid[:h, :w]
            
            # Radial vectors from center
            dx = x_grid - cx
            dy = y_grid - cy
            r = np.sqrt(dx**2 + dy**2)
            
            # Normalize radial vectors
            mask = r > 5  # Avoid division by zero near center
            dx_norm = np.zeros_like(dx, dtype=float)
            dy_norm = np.zeros_like(dy, dtype=float)
            dx_norm[mask] = dx[mask] / r[mask]
            dy_norm[mask] = dy[mask] / r[mask]
            
            # Radial component of gradient
            radial_grad = grad_x * dx_norm + grad_y * dy_norm
            
            # We want to maximize radial gradient alignment
            return -np.sum(np.abs(radial_grad))
        
        # Optimize
        result = minimize(objective, initial_guess, method='Powell',
                         options={'maxiter': 50})
        
        if result.success:
            cx, cy = result.x
            # Ensure within image bounds
            cx = np.clip(cx, 0, w-1)
            cy = np.clip(cy, 0, h-1)
            confidence = 0.8
            return (int(cx), int(cy)), confidence
        
        return None, 0
    
    def _morphological_center(self):
        """Find center using morphological operations"""
        # Threshold to get fiber region
        _, binary = cv2.threshold(self.enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find largest contour
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get center of largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, 0
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Confidence based on circularity
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            confidence = circularity  # More circular = higher confidence
        else:
            confidence = 0
            
        return (cx, cy), confidence
    
    def _edge_based_center(self):
        """Find center using edge points and RANSAC-like approach"""
        # Detect edges
        edges = canny(self.enhanced, sigma=2.0, low_threshold=0.1, high_threshold=0.3)
        edge_points = np.column_stack(np.where(edges))[:, ::-1]  # Convert to (x, y)
        
        if len(edge_points) < 10:
            return None, 0
        
        # Sample edge points
        n_samples = min(1000, len(edge_points))
        indices = np.random.choice(len(edge_points), n_samples, replace=False)
        sampled_points = edge_points[indices]
        
        best_center = None
        best_score = -np.inf
        
        # Try multiple random combinations
        for _ in range(50):
            # Select 3 random points
            if len(sampled_points) < 3:
                continue
                
            idx = np.random.choice(len(sampled_points), 3, replace=False)
            p1, p2, p3 = sampled_points[idx]
            
            # Calculate circumcenter
            d = 2 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            if abs(d) < 1e-6:
                continue
                
            cx = ((p1[0]**2+p1[1]**2)*(p2[1]-p3[1]) + 
                  (p2[0]**2+p2[1]**2)*(p3[1]-p1[1]) + 
                  (p3[0]**2+p3[1]**2)*(p1[1]-p2[1])) / d
            cy = ((p1[0]**2+p1[1]**2)*(p3[0]-p2[0]) + 
                  (p2[0]**2+p2[1]**2)*(p1[0]-p3[0]) + 
                  (p3[0]**2+p3[1]**2)*(p2[0]-p1[0])) / d
            
            # Check if center is within image bounds
            h, w = self.gray.shape
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            
            # Score based on radial histogram
            distances = np.linalg.norm(sampled_points - [cx, cy], axis=1)
            hist, _ = np.histogram(distances, bins=50)
            
            # Good center should have strong peaks in histogram
            score = np.max(hist) + 0.5 * np.sum(hist > np.mean(hist))
            
            if score > best_score:
                best_score = score
                best_center = (int(cx), int(cy))
        
        if best_center is not None:
            confidence = min(1.0, best_score / 100)
            return best_center, confidence
            
        return None, 0
    
    def find_radii_adaptive(self):
        """Find core and cladding radii using adaptive radial profiling"""
        if self.center is None:
            raise ValueError("Center must be found first")
            
        cx, cy = self.center
        h, w = self.gray.shape
        
        # Maximum possible radius
        max_radius = min(cx, cy, w-cx, h-cy)
        
        # Create radial profiles for multiple features
        profiles = self._create_radial_profiles(max_radius)
        
        # Find boundaries using combined profile analysis
        boundaries = self._analyze_profiles(profiles, max_radius)
        
        if len(boundaries) >= 2:
            self.core_radius = boundaries[0]
            self.cladding_radius = boundaries[1]
        else:
            # Fallback estimation
            self.core_radius = int(max_radius * 0.2)
            self.cladding_radius = int(max_radius * 0.6)
            
        print(f"Core radius: {self.core_radius}")
        print(f"Cladding radius: {self.cladding_radius}")
    
    def _create_radial_profiles(self, max_radius):
        """Create multiple radial profiles for robust boundary detection"""
        cx, cy = self.center
        profiles = {
            'intensity': np.zeros(max_radius),
            'gradient': np.zeros(max_radius),
            'variance': np.zeros(max_radius),
            'counts': np.zeros(max_radius)
        }
        
        # Pre-compute features
        grad_mag = np.sqrt(cv2.Sobel(self.enhanced, cv2.CV_64F, 1, 0)**2 + 
                          cv2.Sobel(self.enhanced, cv2.CV_64F, 0, 1)**2)
        
        # Build radial profiles
        h, w = self.gray.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        r_map = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2).astype(int)
        
        for r in range(max_radius):
            mask = r_map == r
            if np.any(mask):
                pixels = self.enhanced[mask]
                profiles['intensity'][r] = np.mean(pixels)
                profiles['gradient'][r] = np.mean(grad_mag[mask])
                profiles['variance'][r] = np.var(pixels) if len(pixels) > 1 else 0
                profiles['counts'][r] = len(pixels)
        
        # Smooth profiles
        for key in ['intensity', 'gradient', 'variance']:
            if len(profiles[key]) > 5:
                profiles[key] = savgol_filter(profiles[key], 
                                             window_length=min(11, len(profiles[key])//2*2-1), 
                                             polyorder=3)
        
        return profiles
    
    def _analyze_profiles(self, profiles, max_radius):
        """Analyze profiles to find boundaries"""
        boundaries = []
        
        # Method 1: Gradient peaks
        grad_profile = profiles['gradient']
        if len(grad_profile) > 10:
            # Normalize
            grad_norm = (grad_profile - grad_profile.min()) / (grad_profile.max() - grad_profile.min() + 1e-6)
            
            # Find peaks
            peaks, properties = find_peaks(grad_norm, 
                                         height=0.3,
                                         distance=max_radius//10,
                                         prominence=0.2)
            
            if len(peaks) > 0:
                # Sort by prominence
                prominences = properties['prominences']
                sorted_idx = np.argsort(prominences)[::-1]
                boundaries.extend(peaks[sorted_idx[:2]])
        
        # Method 2: Intensity derivatives
        intensity_profile = profiles['intensity']
        if len(intensity_profile) > 10:
            # Second derivative
            d2_intensity = np.gradient(np.gradient(intensity_profile))
            d2_smooth = savgol_filter(d2_intensity, 
                                    window_length=min(11, len(d2_intensity)//2*2-1),
                                    polyorder=3)
            
            # Find zero crossings with negative to positive transition
            zero_crossings = []
            for i in range(1, len(d2_smooth)-1):
                if d2_smooth[i-1] < 0 and d2_smooth[i+1] > 0:
                    zero_crossings.append(i)
            
            # Add significant crossings
            if zero_crossings:
                boundaries.extend(zero_crossings[:2])
        
        # Method 3: Variance changes
        var_profile = profiles['variance']
        if len(var_profile) > 10:
            # Look for significant changes in variance
            var_grad = np.abs(np.gradient(var_profile))
            var_peaks, _ = find_peaks(var_grad, height=np.mean(var_grad))
            if len(var_peaks) > 0:
                boundaries.extend(var_peaks[:2])
        
        # Remove duplicates and sort
        boundaries = list(set(boundaries))
        boundaries.sort()
        
        # Ensure we have at least 2 boundaries
        if len(boundaries) < 2:
            # Use default positions based on typical fiber geometry
            boundaries = [int(max_radius * 0.15), int(max_radius * 0.5)]
        
        return boundaries[:2]  # Return first two boundaries
    
    def segment_regions(self):
        """Create masks and segment the image into regions"""
        h, w = self.gray.shape
        
        # Create circular masks
        mask_core = np.zeros((h, w), dtype=np.uint8)
        mask_cladding = np.zeros((h, w), dtype=np.uint8)
        mask_ferrule = np.zeros((h, w), dtype=np.uint8)
        
        # Core mask
        cv2.circle(mask_core, self.center, self.core_radius, 255, -1)
        
        # Cladding mask (annulus)
        cv2.circle(mask_cladding, self.center, self.cladding_radius, 255, -1)
        mask_cladding = cv2.subtract(mask_cladding, mask_core)
        
        # Ferrule mask (outside cladding)
        mask_ferrule = np.ones((h, w), dtype=np.uint8) * 255
        cv2.circle(mask_ferrule, self.center, self.cladding_radius, 0, -1)
        
        # Apply masks to original image
        region_core = cv2.bitwise_and(self.original, self.original, mask=mask_core)
        region_cladding = cv2.bitwise_and(self.original, self.original, mask=mask_cladding)
        region_ferrule = cv2.bitwise_and(self.original, self.original, mask=mask_ferrule)
        
        return {
            'masks': {
                'core': mask_core,
                'cladding': mask_cladding,
                'ferrule': mask_ferrule
            },
            'regions': {
                'core': region_core,
                'cladding': region_cladding,
                'ferrule': region_ferrule
            }
        }
    
    def visualize_results(self, results):
        """Create visualization of the segmentation results"""
        # Create annotated image
        annotated = self.original.copy()
        cv2.circle(annotated, self.center, 3, (0, 255, 255), -1)  # Center point
        cv2.circle(annotated, self.center, self.core_radius, (0, 255, 0), 2)  # Core boundary
        cv2.circle(annotated, self.center, self.cladding_radius, (0, 0, 255), 2)  # Cladding boundary
        
        # Add labels
        cv2.putText(annotated, "Core", 
                   (self.center[0] - 20, self.center[1] - self.core_radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, "Cladding", 
                   (self.center[0] - 35, self.center[1] - self.cladding_radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Create composite visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Detected Boundaries")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(self.enhanced, cmap='gray')
        axes[0, 2].set_title("Enhanced Image")
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(cv2.cvtColor(results['regions']['core'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Core Region")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(results['regions']['cladding'], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Cladding Region")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv2.cvtColor(results['regions']['ferrule'], cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("Ferrule Region")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save results
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        plt.savefig(os.path.join(self.output_dir, f"{base_name}_visualization.png"), dpi=150)
        cv2.imwrite(os.path.join(self.output_dir, f"{base_name}_annotated.png"), annotated)
        cv2.imwrite(os.path.join(self.output_dir, f"{base_name}_core.png"), results['regions']['core'])
        cv2.imwrite(os.path.join(self.output_dir, f"{base_name}_cladding.png"), results['regions']['cladding'])
        cv2.imwrite(os.path.join(self.output_dir, f"{base_name}_ferrule.png"), results['regions']['ferrule'])
        
        plt.close()
        
    def process(self):
        """Main processing pipeline"""
        print("\n=== Universal Fiber Optic Segmentation ===")
        
        # Step 1: Load and preprocess
        print("\n1. Loading and preprocessing...")
        self.load_and_preprocess()
        
        # Step 2: Find center using multiple methods
        print("\n2. Finding fiber center using multiple methods...")
        self.find_center_multi_method()
        
        # Step 3: Find radii using adaptive profiling
        print("\n3. Finding core and cladding boundaries...")
        self.find_radii_adaptive()
        
        # Step 4: Segment regions
        print("\n4. Segmenting regions...")
        results = self.segment_regions()
        
        # Step 5: Visualize and save results
        print("\n5. Creating visualizations...")
        self.visualize_results(results)
        
        print(f"\nResults saved to: {self.output_dir}")
        
        # Return results for further analysis if needed
        return {
            'center': self.center,
            'core_radius': self.core_radius,
            'cladding_radius': self.cladding_radius,
            'masks': results['masks'],
            'regions': results['regions']
        }


def main():
    """Main function with user interaction"""
    print("=== Universal Fiber Optic Endface Segmentation Tool ===")
    print("\nThis tool segments fiber optic endface images without relying on Hough circles.")
    print("It uses multiple robust methods to ensure accurate segmentation.\n")
    
    # Get image path from user
    image_path = input("Enter the full path to your fiber optic image: ").strip().strip('"').strip("'")
    
    # Get output directory (optional)
    output_dir = input("Enter output directory (press Enter for 'output_universal'): ").strip()
    if not output_dir:
        output_dir = 'output_universal'
    
    try:
        # Create segmenter and process
        segmenter = FiberOpticSegmenter(image_path, output_dir)
        results = segmenter.process()
        
        print("\n=== Segmentation Complete ===")
        print(f"Center: {results['center']}")
        print(f"Core radius: {results['core_radius']} pixels")
        print(f"Cladding radius: {results['cladding_radius']} pixels")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your image path and try again.")


if __name__ == "__main__":
    main()
