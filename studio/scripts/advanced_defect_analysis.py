import cv2
import numpy as np
from scipy import stats, ndimage
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import feature, morphology, filters
from skimage.segmentation import watershed
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

class AdvancedDefectAnalyzer:
    """Advanced defect analysis using multiple sophisticated methods"""
    
    def __init__(self, region_image, region_name="region"):
        self.original = region_image
        self.region_name = region_name
        self.gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY) if len(region_image.shape) == 3 else region_image
        self.mask = self.gray > 0  # Non-black pixels
        self.results = {}
        
    def statistical_outlier_detection(self):
        """Multiple statistical methods for outlier detection"""
        pixels = self.gray[self.mask]
        results = {}
        
        # 1. Z-score method
        z_scores = np.abs(stats.zscore(pixels))
        z_outliers = z_scores > 3
        results['z_score'] = {
            'outliers': z_outliers,
            'count': np.sum(z_outliers),
            'percentage': (np.sum(z_outliers) / len(pixels)) * 100
        }
        
        # 2. Modified Z-score using MAD
        median = np.median(pixels)
        mad = np.median(np.abs(pixels - median))
        modified_z_scores = 0.6745 * (pixels - median) / mad
        mad_outliers = np.abs(modified_z_scores) > 3.5
        results['mad'] = {
            'outliers': mad_outliers,
            'count': np.sum(mad_outliers),
            'percentage': (np.sum(mad_outliers) / len(pixels)) * 100
        }
        
        # 3. IQR method
        Q1 = np.percentile(pixels, 25)
        Q3 = np.percentile(pixels, 75)
        IQR = Q3 - Q1
        iqr_outliers = (pixels < (Q1 - 1.5 * IQR)) | (pixels > (Q3 + 1.5 * IQR))
        results['iqr'] = {
            'outliers': iqr_outliers,
            'count': np.sum(iqr_outliers),
            'percentage': (np.sum(iqr_outliers) / len(pixels)) * 100
        }
        
        # 4. Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        pixel_reshape = pixels.reshape(-1, 1)
        iso_predictions = iso_forest.fit_predict(pixel_reshape)
        iso_outliers = iso_predictions == -1
        results['isolation_forest'] = {
            'outliers': iso_outliers,
            'count': np.sum(iso_outliers),
            'percentage': (np.sum(iso_outliers) / len(pixels)) * 100
        }
        
        self.results['statistical'] = results
        return results
    
    def local_adaptive_thresholding(self, window_size=15):
        """Advanced local thresholding methods"""
        results = {}
        
        # 1. Niblack's method
        mean = cv2.blur(self.gray, (window_size, window_size))
        mean_sq = cv2.blur(self.gray**2, (window_size, window_size))
        std = np.sqrt(mean_sq - mean**2)
        k = -0.2
        niblack_threshold = mean + k * std
        niblack_defects = (self.gray < niblack_threshold) & self.mask
        results['niblack'] = niblack_defects
        
        # 2. Sauvola's method
        k = 0.5
        R = 128
        sauvola_threshold = mean * (1 + k * ((std / R) - 1))
        sauvola_defects = (self.gray < sauvola_threshold) & self.mask
        results['sauvola'] = sauvola_defects
        
        # 3. Local contrast
        local_min = cv2.erode(self.gray, np.ones((window_size, window_size)))
        local_max = cv2.dilate(self.gray, np.ones((window_size, window_size)))
        local_contrast = (local_max - local_min) / (local_max + local_min + 1e-7)
        contrast_defects = (local_contrast > 0.3) & self.mask
        results['contrast'] = contrast_defects
        
        self.results['adaptive'] = results
        return results
    
    def texture_analysis(self):
        """Texture-based defect detection using LBP and GLCM"""
        results = {}
        
        # 1. Local Binary Patterns
        radius = 1
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(self.gray, n_points, radius, method='uniform')
        
        # Calculate LBP histogram for normal regions
        normal_lbp = lbp[self.mask].flatten()
        hist, _ = np.histogram(normal_lbp, bins=np.arange(0, n_points + 3), density=True)
        
        # Sliding window LBP analysis
        window_size = 15
        lbp_anomaly_map = np.zeros_like(self.gray, dtype=float)
        
        for i in range(window_size//2, self.gray.shape[0] - window_size//2):
            for j in range(window_size//2, self.gray.shape[1] - window_size//2):
                if self.mask[i, j]:
                    window = lbp[i-window_size//2:i+window_size//2+1, 
                                j-window_size//2:j+window_size//2+1]
                    window_hist, _ = np.histogram(window.flatten(), 
                                                 bins=np.arange(0, n_points + 3), 
                                                 density=True)
                    # Chi-square distance
                    chi_sq = np.sum((window_hist - hist)**2 / (window_hist + hist + 1e-7))
                    lbp_anomaly_map[i, j] = chi_sq
        
        # Threshold anomaly map
        lbp_threshold = np.percentile(lbp_anomaly_map[self.mask], 95)
        lbp_defects = (lbp_anomaly_map > lbp_threshold) & self.mask
        results['lbp'] = lbp_defects
        
        # 2. Haralick features (simplified)
        # Calculate local variance as texture measure
        local_var = ndimage.generic_filter(self.gray, np.var, size=7)
        var_threshold = np.percentile(local_var[self.mask], 95)
        texture_defects = (local_var > var_threshold) & self.mask
        results['variance'] = texture_defects
        
        self.results['texture'] = results
        return results
    
    def morphological_analysis(self):
        """Morphological operations for defect detection"""
        results = {}
        
        # 1. Top-hat transform (bright defects)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat = cv2.morphologyEx(self.gray, cv2.MORPH_TOPHAT, kernel)
        tophat_threshold = np.percentile(tophat[self.mask], 95)
        tophat_defects = (tophat > tophat_threshold) & self.mask
        results['tophat'] = tophat_defects
        
        # 2. Bottom-hat transform (dark defects)
        blackhat = cv2.morphologyEx(self.gray, cv2.MORPH_BLACKHAT, kernel)
        blackhat_threshold = np.percentile(blackhat[self.mask], 95)
        blackhat_defects = (blackhat > blackhat_threshold) & self.mask
        results['blackhat'] = blackhat_defects
        
        # 3. Morphological gradient
        gradient = cv2.morphologyEx(self.gray, cv2.MORPH_GRADIENT, kernel)
        gradient_threshold = np.percentile(gradient[self.mask], 95)
        gradient_defects = (gradient > gradient_threshold) & self.mask
        results['gradient'] = gradient_defects
        
        self.results['morphological'] = results
        return results
    
    def frequency_domain_analysis(self):
        """Frequency domain defect detection"""
        results = {}
        
        # 1. FFT-based analysis
        f_transform = np.fft.fft2(self.gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # High-pass filter to detect high-frequency defects
        rows, cols = self.gray.shape
        crow, ccol = rows//2, cols//2
        
        # Create high-pass filter
        mask_hp = np.ones((rows, cols), np.uint8)
        r = 30  # Filter radius
        center = (crow, ccol)
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r**2
        mask_hp[mask_area] = 0
        
        # Apply filter
        f_shift_hp = f_shift * mask_hp
        f_ishift = np.fft.ifftshift(f_shift_hp)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Threshold high-frequency components
        hp_threshold = np.percentile(img_back[self.mask], 95)
        fft_defects = (img_back > hp_threshold) & self.mask
        results['fft_highpass'] = fft_defects
        
        # 2. Wavelet-based edge detection
        edges = filters.sobel(self.gray)
        edge_threshold = np.percentile(edges[self.mask], 95)
        edge_defects = (edges > edge_threshold) & self.mask
        results['edges'] = edge_defects
        
        self.results['frequency'] = results
        return results
    
    def spatial_clustering(self):
        """Spatial clustering of anomalous regions"""
        # Combine various defect masks
        all_defects = np.zeros_like(self.mask, dtype=bool)
        
        if 'statistical' in self.results:
            all_defects |= self.create_defect_mask_from_statistical()
        if 'adaptive' in self.results:
            for method_defects in self.results['adaptive'].values():
                all_defects |= method_defects
        
        # Get coordinates of defect pixels
        defect_coords = np.column_stack(np.where(all_defects))
        
        if len(defect_coords) > 10:  # Need minimum points for DBSCAN
            # DBSCAN clustering
            clustering = DBSCAN(eps=5, min_samples=5).fit(defect_coords)
            labels = clustering.labels_
            
            # Create labeled defect image
            labeled_defects = np.zeros_like(self.gray)
            for i, (y, x) in enumerate(defect_coords):
                if labels[i] != -1:  # Not noise
                    labeled_defects[y, x] = labels[i] + 1
            
            self.results['clustering'] = {
                'labels': labeled_defects,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'noise_points': np.sum(labels == -1)
            }
        
        return self.results.get('clustering', None)
    
    def create_defect_mask_from_statistical(self):
        """Helper to create defect mask from statistical results"""
        defect_mask = np.zeros(self.gray.shape, dtype=bool)
        pixels = self.gray[self.mask]
        
        for method, data in self.results['statistical'].items():
            outliers = data['outliers']
            temp_mask = np.zeros_like(defect_mask)
            temp_mask[self.mask] = outliers
            defect_mask |= temp_mask
        
        return defect_mask
    
    def comprehensive_analysis(self):
        """Run all analysis methods"""
        print(f"\n{'='*60}")
        print(f"ADVANCED DEFECT ANALYSIS - {self.region_name.upper()}")
        print(f"{'='*60}\n")
        
        # Run all analyses
        print("1. Statistical Outlier Detection...")
        stat_results = self.statistical_outlier_detection()
        for method, data in stat_results.items():
            print(f"   - {method}: {data['count']} defects ({data['percentage']:.2f}%)")
        
        print("\n2. Local Adaptive Thresholding...")
        adaptive_results = self.local_adaptive_thresholding()
        for method, defects in adaptive_results.items():
            count = np.sum(defects)
            percentage = (count / np.sum(self.mask)) * 100
            print(f"   - {method}: {count} defects ({percentage:.2f}%)")
        
        print("\n3. Texture Analysis...")
        texture_results = self.texture_analysis()
        for method, defects in texture_results.items():
            count = np.sum(defects)
            percentage = (count / np.sum(self.mask)) * 100
            print(f"   - {method}: {count} defects ({percentage:.2f}%)")
        
        print("\n4. Morphological Analysis...")
        morph_results = self.morphological_analysis()
        for method, defects in morph_results.items():
            count = np.sum(defects)
            percentage = (count / np.sum(self.mask)) * 100
            print(f"   - {method}: {count} defects ({percentage:.2f}%)")
        
        print("\n5. Frequency Domain Analysis...")
        freq_results = self.frequency_domain_analysis()
        for method, defects in freq_results.items():
            count = np.sum(defects)
            percentage = (count / np.sum(self.mask)) * 100
            print(f"   - {method}: {count} defects ({percentage:.2f}%)")
        
        print("\n6. Spatial Clustering...")
        cluster_results = self.spatial_clustering()
        if cluster_results:
            print(f"   - Found {cluster_results['n_clusters']} defect clusters")
            print(f"   - Noise points: {cluster_results['noise_points']}")
        
        return self.results
    
    def visualize_results(self, save_prefix="advanced_defects"):
        """Create comprehensive visualization of all results"""
        n_methods = 5  # Number of method categories
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Advanced Defect Analysis - {self.region_name}', fontsize=16)
        
        # Original
        ax = axes[0, 0]
        ax.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB) if len(self.original.shape) == 3 else self.original, cmap='gray')
        ax.set_title('Original')
        ax.axis('off')
        
        # Statistical outliers (combined)
        ax = axes[0, 1]
        stat_combined = self.create_defect_mask_from_statistical()
        ax.imshow(stat_combined, cmap='hot')
        ax.set_title('Statistical Outliers (Combined)')
        ax.axis('off')
        
        # Adaptive thresholding (Niblack)
        ax = axes[0, 2]
        if 'adaptive' in self.results:
            ax.imshow(self.results['adaptive']['niblack'], cmap='hot')
            ax.set_title('Niblack Adaptive Threshold')
        ax.axis('off')
        
        # Texture (LBP)
        ax = axes[1, 0]
        if 'texture' in self.results:
            ax.imshow(self.results['texture']['lbp'], cmap='hot')
            ax.set_title('LBP Texture Anomalies')
        ax.axis('off')
        
        # Morphological (Top-hat)
        ax = axes[1, 1]
        if 'morphological' in self.results:
            ax.imshow(self.results['morphological']['tophat'], cmap='hot')
            ax.set_title('Top-hat (Bright Defects)')
        ax.axis('off')
        
        # Frequency (Edges)
        ax = axes[1, 2]
        if 'frequency' in self.results:
            ax.imshow(self.results['frequency']['edges'], cmap='hot')
            ax.set_title('High-Frequency Defects')
        ax.axis('off')
        
        # Clustering result
        ax = axes[2, 0]
        if 'clustering' in self.results:
            ax.imshow(self.results['clustering']['labels'], cmap='tab20')
            ax.set_title(f"Defect Clusters ({self.results['clustering']['n_clusters']})")
        ax.axis('off')
        
        # Combined defects with original overlay
        ax = axes[2, 1]
        overlay = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB) if len(self.original.shape) == 3 else cv2.cvtColor(self.gray, cv2.COLOR_GRAY2RGB)
        combined_mask = np.zeros_like(self.mask)
        for method_results in self.results.values():
            if isinstance(method_results, dict):
                for defects in method_results.values():
                    if isinstance(defects, np.ndarray) and defects.dtype == bool:
                        combined_mask |= defects
        overlay[combined_mask] = [255, 0, 0]  # Red for defects
        ax.imshow(overlay)
        ax.set_title('All Defects Combined')
        ax.axis('off')
        
        # Summary statistics
        ax = axes[2, 2]
        ax.axis('off')
        summary_text = f"Total pixels: {np.sum(self.mask)}\n"
        summary_text += f"Combined defects: {np.sum(combined_mask)}\n"
        summary_text += f"Defect rate: {(np.sum(combined_mask)/np.sum(self.mask)*100):.2f}%"
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        ax.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_comprehensive.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved as {save_prefix}_comprehensive.png")

def main():
    """Example usage"""
    # Load a region image (assumes you've run one of the segmentation scripts)
    region_image = cv2.imread("core.png")
    
    if region_image is None:
        print("Please run core.py first to generate the core region image.")
        return
    
    # Create analyzer
    analyzer = AdvancedDefectAnalyzer(region_image, "core")
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_analysis()
    
    # Visualize results
    analyzer.visualize_results("core_advanced_defects")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()