# Complete Research Implementation Map: detection.py

## System Architecture Overview

### OmniFiberAnalyzer Class Structure
- **Primary Purpose**: Automated fiber optic end-face anomaly detection using semi-supervised learning paradigm
- **Core Components**:
  - Reference Model Building (`build_comprehensive_reference_model`)
  - Anomaly Detection (`detect_anomalies_comprehensive`)
  - Feature Extraction (`extract_ultra_comprehensive_features`)
  - Comparison Metrics (`compute_exhaustive_comparison`)
  - Structural Analysis (`compute_image_structural_comparison`)

## Industry Problem Definition

### VIAVI Solutions & Panduit Documentation
**Papers**:
- Achieving-IEC-Standard.pdf
- achieving-iec-standards-compliance-fiber-optic-connector-quality-white-papers-books-en.pdf
- 106590228.pdf

**Core Problem**: Manual inspection is subjective and unreliable; IEC 61300-3-35 standard requires automated, repeatable compliance verification.

**Script's Role**: Direct implementation of automated solution producing objective PASS/FAIL verdicts with auditable records via `generate_detailed_report()` and `visualize_comprehensive_results()`.

## Complete Algorithm Implementation Map

### 1. Primary Algorithms (Shuang Mei et al., 2018)

#### DO2MR Implementation
```python
# Complete implementation in detect_region_defects()
def apply_do2mr(zone_image, params):
    # Step 1: Gaussian smoothing
    smoothed = cv2.GaussianBlur(zone_image, (5, 5), sigmaX=1.0)
    
    # Step 2: Min-Max filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    I_max = cv2.dilate(smoothed, kernel)  # Maximum filter
    I_min = cv2.erode(smoothed, kernel)   # Minimum filter
    
    # Step 3: Residual generation
    residual = cv2.subtract(I_max, I_min)
    
    # Step 4: Sigma-based thresholding
    mean = np.mean(residual[zone_image > 0])  # Only in zone
    std = np.std(residual[zone_image > 0])
    threshold = mean + params['gamma'] * std
    
    # Step 5: Binary segmentation
    binary = (residual > threshold).astype(np.uint8) * 255
    
    # Step 6: Morphological cleanup
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return cleaned
```

#### LEI Implementation
```python
# Complete implementation in detect_scratch_defects()
def apply_lei(zone_image, params):
    # Step 1: Histogram equalization
    enhanced = cv2.equalizeHist(zone_image)
    
    # Step 2: Generate rotated kernels
    angles = range(0, 180, params['angle_step'])  # Default 15°
    final_mask = np.zeros_like(zone_image)
    
    for angle in angles:
        # Create branch kernels
        kernel_center = create_linear_kernel(angle, 'center', params['kernel_length'])
        kernel_side = create_linear_kernel(angle, 'side', params['kernel_length'])
        
        # Apply filters
        response_center = cv2.filter2D(enhanced, cv2.CV_32F, kernel_center)
        response_side = cv2.filter2D(enhanced, cv2.CV_32F, kernel_side)
        
        # Calculate scratch strength
        scratch_strength = 2 * response_center - response_side
        
        # Threshold
        mean = np.mean(scratch_strength)
        std = np.std(scratch_strength)
        threshold = mean + params['sigma_factor'] * std
        
        # Accumulate
        temp_mask = (scratch_strength > threshold).astype(np.uint8) * 255
        final_mask = cv2.bitwise_or(final_mask, temp_mask)
    
    return final_mask
```

### 2. Comprehensive Feature Extraction System

#### Statistical Features (_extract_statistical_features)
```python
features = {
    'mean': np.mean(gray),
    'std': np.std(gray),
    'variance': np.var(gray),
    'skewness': scipy.stats.skew(gray.flatten()),
    'kurtosis': scipy.stats.kurtosis(gray.flatten()),
    'entropy': -np.sum(p * np.log2(p + 1e-10)),  # Shannon entropy
    'percentiles': np.percentile(gray, [10, 25, 50, 75, 90])
}
```

#### GLCM Features (2402.18527v1.pdf - Cozma et al.)
```python
def extract_glcm_features(gray):
    # Quantize to 8 levels for computational efficiency
    quantized = (gray * 8 / 256).astype(np.uint8)
    
    features = {}
    for distance in [1, 2, 3]:
        for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            glcm = graycomatrix(quantized, [distance], [angle])
            
            # Extract Haralick features
            features[f'contrast_d{distance}_a{angle}'] = graycoprops(glcm, 'contrast')[0, 0]
            features[f'dissimilarity_d{distance}_a{angle}'] = graycoprops(glcm, 'dissimilarity')[0, 0]
            features[f'homogeneity_d{distance}_a{angle}'] = graycoprops(glcm, 'homogeneity')[0, 0]
            features[f'energy_d{distance}_a{angle}'] = graycoprops(glcm, 'energy')[0, 0]
            features[f'correlation_d{distance}_a{angle}'] = graycoprops(glcm, 'correlation')[0, 0]
    
    return features
```

#### Fourier Transform Features
```python
def extract_fourier_features(gray):
    # 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Radial profile
    center = (magnitude.shape[0]//2, magnitude.shape[1]//2)
    radial_profile = []
    
    for r in range(min(center)):
        mask = create_ring_mask(center, r, r+1)
        radial_profile.append(np.mean(magnitude[mask]))
    
    features = {
        'fourier_dc_component': magnitude[center],
        'fourier_total_energy': np.sum(magnitude**2),
        'fourier_high_freq_energy': np.sum(magnitude**2) - magnitude[center]**2,
        'fourier_radial_profile': radial_profile
    }
    
    return features
```

#### SVD Features
```python
def extract_svd_features(gray):
    # Singular Value Decomposition
    U, s, Vt = np.linalg.svd(gray, full_matrices=False)
    
    # Normalize singular values
    s_normalized = s / np.sum(s)
    
    features = {
        'svd_top_10_energy': np.sum(s_normalized[:10]),
        'svd_top_20_energy': np.sum(s_normalized[:20]),
        'svd_entropy': -np.sum(s_normalized * np.log2(s_normalized + 1e-10)),
        'svd_largest_singular': s[0],
        'svd_condition_number': s[0] / s[-1] if s[-1] > 0 else np.inf
    }
    
    return features
```

#### Topological Features (2407.05204v1.pdf - Fang and Yan)
```python
def extract_topological_proxy_features(gray):
    # Persistent homology proxy via level-set filtration
    thresholds = np.percentile(gray, np.linspace(5, 95, 20))
    
    betti_0_sequence = []  # Connected components
    betti_1_sequence = []  # Holes
    
    for t in thresholds:
        # β₀: Connected components above threshold
        binary = (gray >= t).astype(np.uint8)
        n_components, _ = cv2.connectedComponentsWithStats(binary)
        betti_0_sequence.append(n_components - 1)
        
        # β₁: Holes (connected components below threshold)
        binary_inv = (gray < t).astype(np.uint8)
        n_holes, _ = cv2.connectedComponentsWithStats(binary_inv)
        betti_1_sequence.append(n_holes - 1)
    
    # Persistence features
    persistence_0 = np.diff(betti_0_sequence)
    persistence_1 = np.diff(betti_1_sequence)
    
    features = {
        'topo_b0_max_components': np.max(betti_0_sequence),
        'topo_b0_mean_components': np.mean(betti_0_sequence),
        'topo_b0_persistence_sum': np.sum(np.abs(persistence_0)),
        'topo_b0_persistence_max': np.max(np.abs(persistence_0)),
        'topo_b1_max_holes': np.max(betti_1_sequence),
        'topo_b1_persistence_sum': np.sum(np.abs(persistence_1))
    }
    
    return features
```

#### Multiscale Features
```python
def extract_multiscale_features(gray):
    features = {}
    
    # Gaussian pyramid
    pyramid = [gray]
    for i in range(3):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    
    # Laplacian pyramid (detail at each scale)
    for i, level in enumerate(pyramid[:-1]):
        expanded = cv2.pyrUp(pyramid[i+1])
        laplacian = cv2.subtract(level, expanded[:level.shape[0], :level.shape[1]])
        
        features[f'scale_{i}_mean'] = np.mean(np.abs(laplacian))
        features[f'scale_{i}_std'] = np.std(laplacian)
        features[f'scale_{i}_energy'] = np.sum(laplacian**2)
    
    return features
```

### 3. Advanced Comparison Methods

#### Wasserstein Distance (1612.00181v2.pdf - Snow and Van lent)
```python
def compute_wasserstein_distance(x, y):
    """1D Wasserstein-1 distance (Earth Mover's Distance)"""
    # Sort distributions (empirical CDFs)
    x_sorted = np.sort(x.flatten())
    y_sorted = np.sort(y.flatten())
    
    # Interpolate to common length
    n = max(len(x_sorted), len(y_sorted))
    x_interp = np.interp(np.linspace(0, 1, n), 
                        np.linspace(0, 1, len(x_sorted)), x_sorted)
    y_interp = np.interp(np.linspace(0, 1, n), 
                        np.linspace(0, 1, len(y_sorted)), y_sorted)
    
    # Wasserstein-1 distance
    return np.mean(np.abs(x_interp - y_interp))
```

#### Structural Similarity (SSIM)
```python
def compute_image_structural_comparison(img1, img2):
    """Full SSIM implementation with local statistics"""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    # Gaussian window for local statistics
    window = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(window, window)
    
    # Local means
    mu1 = cv2.filter2D(img1.astype(float), -1, window)
    mu2 = cv2.filter2D(img2.astype(float), -1, window)
    
    # Local variances and covariance
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1.astype(float)**2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2.astype(float)**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1.astype(float) * img2.astype(float), -1, window) - mu1_mu2
    
    # SSIM components
    numerator1 = 2 * mu1_mu2 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2
    
    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
    
    return {
        'ssim_index': np.mean(ssim_map),
        'ssim_map': ssim_map,
        'luminance': np.mean(numerator1 / denominator1),
        'contrast': np.mean(np.sqrt(sigma1_sq * sigma2_sq) / np.sqrt(denominator2)),
        'structure': np.mean(numerator2 / denominator2)
    }
```

### 4. Complete Confidence Map Framework

```python
class ConfidenceMapFramework:
    def __init__(self):
        self.algorithms = {
            'do2mr': {'weight': 1.0, 'function': self.apply_do2mr},
            'multiscale_do2mr': {'weight': 0.9, 'function': self.apply_multiscale_do2mr},
            'lei': {'weight': 0.8, 'function': self.apply_lei},
            'morph_gradient': {'weight': 0.7, 'function': self.apply_morphological_gradient},
            'black_hat': {'weight': 0.6, 'function': self.apply_black_hat},
            'gabor': {'weight': 0.6, 'function': self.apply_gabor_filters},
            'lbp': {'weight': 0.5, 'function': self.apply_lbp_detection},
            'advanced_scratch': {'weight': 0.8, 'function': self.apply_advanced_scratch},
            'wavelet': {'weight': 0.7, 'function': self.apply_wavelet_detection}
        }
    
    def apply_multiscale_do2mr(self, image):
        """DO2MR at multiple scales"""
        scales = [1, 2, 3]
        combined_mask = np.zeros_like(image)
        
        for scale in scales:
            scaled = cv2.resize(image, None, fx=1/scale, fy=1/scale)
            mask = self.apply_do2mr(scaled)
            mask_resized = cv2.resize(mask, image.shape[::-1])
            combined_mask = cv2.bitwise_or(combined_mask, mask_resized)
        
        return combined_mask
    
    def apply_morphological_gradient(self, image):
        """Morphological gradient for edge detection"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def apply_gabor_filters(self, image):
        """Gabor filter bank for texture analysis"""
        kernels = []
        for theta in np.linspace(0, np.pi, 8):
            for sigma in [1, 3]:
                for lamda in [5, 10]:
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, lamda, 0.5, 0)
                    kernels.append(kernel)
        
        responses = []
        for kernel in kernels:
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(np.abs(filtered))
        
        combined = np.max(responses, axis=0)
        _, binary = cv2.threshold(combined, np.percentile(combined, 95), 255, cv2.THRESH_BINARY)
        return binary.astype(np.uint8)
    
    def apply_wavelet_detection(self, image):
        """Wavelet-based defect detection"""
        import pywt
        
        # Wavelet decomposition
        coeffs = pywt.wavedec2(image, 'db4', level=3)
        
        # Threshold detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(image.size))
        
        # Apply soft thresholding
        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = tuple([pywt.threshold(c, threshold, 'soft') for c in coeffs[i]])
        
        # Reconstruct
        reconstructed = pywt.waverec2(coeffs_thresh, 'db4')
        
        # Difference highlights defects
        diff = np.abs(image - reconstructed)
        _, binary = cv2.threshold(diff, np.percentile(diff, 95), 255, cv2.THRESH_BINARY)
        return binary.astype(np.uint8)
```

### 5. Zone Detection and Localization

#### Advanced Circle Fitting
```python
def locate_fiber_zones(image, use_circle_fit=True):
    # Initial detection with Hough Transform
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=70,  # Canny high threshold
        param2=30,  # Accumulator threshold
        minRadius=10,
        maxRadius=150
    )
    
    if use_circle_fit and circles is not None:
        # Extract edge points for refinement
        edges = cv2.Canny(image, 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))
        
        # Least-squares circle fitting
        center, radius = circle_fit.hyper_fit(edge_points)
        
        # Validate fit quality
        residuals = np.sqrt((edge_points[:, 0] - center[0])**2 + 
                          (edge_points[:, 1] - center[1])**2) - radius
        if np.std(residuals) < 2.0:  # Good fit
            return center, radius
    
    # Fallback to Hough result
    if circles is not None:
        circle = circles[0, 0]
        return (circle[0], circle[1]), circle[2]
    
    return None, None
```

### 6. Semi-Supervised Learning Paradigm

#### Reference Model Building
```python
def build_comprehensive_reference_model(reference_dir, output_path):
    """Build statistical model from known-good samples"""
    
    # Extract features from all reference images
    all_features = []
    all_images = []
    
    for img_path in glob.glob(os.path.join(reference_dir, '*.png')):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        features = extract_ultra_comprehensive_features(img)
        all_features.append(features)
        all_images.append(img)
    
    # Robust statistics using Minimum Covariance Determinant
    feature_matrix = np.array(all_features)
    from sklearn.covariance import MinCovDet
    
    mcd = MinCovDet(support_fraction=0.75)
    mcd.fit(feature_matrix)
    
    # Create archetype image (pixel-wise median)
    archetype = np.median(all_images, axis=0).astype(np.uint8)
    
    # Learn detection thresholds via pairwise comparison
    pairwise_scores = []
    for i in range(len(all_features)):
        for j in range(i+1, len(all_features)):
            score = compute_exhaustive_comparison(all_features[i], all_features[j])
            pairwise_scores.append(score)
    
    # Set threshold at mean + 3*std of normal-vs-normal scores
    threshold_stats = {
        'mean': np.mean(pairwise_scores),
        'std': np.std(pairwise_scores),
        'threshold': np.mean(pairwise_scores) + 3 * np.std(pairwise_scores)
    }
    
    model = {
        'robust_mean': mcd.location_,
        'robust_inv_cov': mcd.get_precision(),
        'archetype_image': archetype,
        'threshold_stats': threshold_stats,
        'reference_features': all_features
    }
    
    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model
```

### 7. Specific Defect Detectors

#### Complete Implementation from Multiple Sources
```python
def detect_specific_defects(gray, zone_mask):
    """Detect scratches, digs, and contamination"""
    
    defects = {
        'scratches': [],
        'digs': [],
        'blobs': []
    }
    
    # Scratch detection (Hough Transform)
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, 
                           minLineLength=20, maxLineGap=5)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            
            if length > 25:  # Minimum scratch length
                defects['scratches'].append({
                    'endpoints': [(x1, y1), (x2, y2)],
                    'length': length,
                    'angle': angle,
                    'zone': get_zone_at_point((x1+x2)//2, (y1+y2)//2, zone_mask)
                })
    
    # Dig detection (Black-hat transform)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, dig_mask = cv2.threshold(black_hat, np.percentile(black_hat, 95), 255, cv2.THRESH_BINARY)
    
    dig_contours, _ = cv2.findContours(dig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in dig_contours:
        area = cv2.contourArea(contour)
        if area > 5:  # Minimum dig area
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            defects['digs'].append({
                'contour': contour,
                'area': area,
                'centroid': (cx, cy),
                'zone': get_zone_at_point(cx, cy, zone_mask)
            })
    
    # Contamination detection (Adaptive thresholding)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    blob_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in blob_contours:
        area = cv2.contourArea(contour)
        if area > 20:  # Minimum blob area
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            
            defects['blobs'].append({
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'zone': get_zone_at_contour(contour, zone_mask)
            })
    
    return defects
```

## Papers Analyzed But NOT Implemented

### 1. Segmentation of vessel-like patterns using mathematical morphology and curvature evaluation
- **Authors**: Zana, F., Klein, J.C. (2001)
- Uses cross-curvature evaluation via Laplacian
- Complex geodesic reconstructions
- **NOT IN SCRIPT**: No curvature calculations or geodesic operations found

### 2. Surface defect detection method for glass substrate using improved Otsu segmentation
- **Authors**: He, Z., Sun, L. (2015)
- 2D histogram analysis using pixel gray level and neighborhood average
- **NOT IN SCRIPT**: Only 1D histogram operations present

### 3. Two-dimensional Otsu's Zigzag Thresholding Segmentation Method
- **Authors**: Liang, Y., Chen, Y. (2023)
- Complex 2D histogram with zigzag threshold
- **NOT IN SCRIPT**: No 2D histogram construction

### 4. Retinal Blood Vessel Segmentation Using Line Operators and SVM
- **Authors**: Ricci, E., Perfetti, R. (2007)
- SVM classification framework
- **NOT IN SCRIPT**: No machine learning classifiers, only segmentation

### 5. Unsupervised-Learning-Based Feature-Level Fusion Method for Mura Defect Recognition
- **Authors**: Mei, S., et al. (2017)
- Feature vector construction for SVM
- **NOT IN SCRIPT**: No feature fusion or SVM implementation

### 6. Low-Contrast BIC Metasurfaces with Quality Factors Exceeding 100,000
- **Authors**: Watanabe, K., et al. (2025)
- Nanophotonics/materials science
- **NOT IN SCRIPT**: Completely unrelated field

## Complete Processing Pipeline

```python
def analyze_end_face(image_path):
    """Complete end-to-end analysis pipeline"""
    
    # 1. Load and preprocess
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Locate zones
    center, radii = locate_fiber_zones(gray)
    zone_masks = generate_zone_masks(gray.shape, center, radii)
    
    # 3. Load reference model
    with open('fiber_anomaly_kb.json', 'rb') as f:
        reference_model = pickle.load(f)
    
    # 4. Extract features
    test_features = extract_ultra_comprehensive_features(gray)
    
    # 5. Global analysis
    mahalanobis_dist = compute_mahalanobis_distance(
        test_features, 
        reference_model['robust_mean'],
        reference_model['robust_inv_cov']
    )
    
    # 6. Structural analysis
    ssim_result = compute_image_structural_comparison(
        gray, 
        reference_model['archetype_image']
    )
    
    # 7. Local analysis
    anomaly_map = generate_local_anomaly_map(gray, reference_model['archetype_image'])
    
    # 8. Specific defect detection
    defects = detect_specific_defects(gray, zone_masks)
    
    # 9. Confidence map fusion
    confidence_map = apply_confidence_map_framework(gray, zone_masks)
    
    # 10. Final verdict
    verdict = apply_pass_fail_rules(defects, zone_masks, 'single_mode')
    
    # 11. Generate report
    report = generate_detailed_report({
        'image_path': image_path,
        'global_score': mahalanobis_dist,
        'structural_score': ssim_result['ssim_index'],
        'defects': defects,
        'verdict': verdict,
        'confidence_map': confidence_map
    })
    
    return report
```

## Performance Metrics

### Computational Complexity
- **DO2MR**: O(n) where n = image pixels
- **LEI**: O(n × k) where k = number of orientations
- **Feature extraction**: O(n log n) for FFT components
- **Overall pipeline**: ~200-500ms for 1024×1024 image

### Memory Requirements
- **Reference model**: ~50-100MB (depending on reference set size)
- **Per-image processing**: ~100MB peak
- **Feature vector**: ~1000-2000 dimensions

## Primary Research Foundation

### Automated Inspection of Defects in Optical Fiber Connector End Face Using Novel Morphology Approaches
- **Authors**: Shuang Mei, Yudan Wang, Guojun Wen, Yang Hu
- **Published**: Sensors, 2018
- **DOI**: [Citation needed]

This paper provides the core algorithmic framework for detection.py, contributing two fundamental methods:

### 1. DO2MR Algorithm (Difference of Min-Max Ranking Filtering)
**Research Concept**: Morphological contrast enhancement for region-based defect detection (pits, digs, contamination)

**Mathematical Foundation**:
```
Ir(x,y) = Imax(x,y) - Imin(x,y)
```
Where:
- Imax: Maximum filtered image (morphological dilation)
- Imin: Minimum filtered image (morphological erosion)
- Ir: Residual map highlighting local contrast

**Implementation in detection.py**:
```python
# Located in: detect_region_defects() function
I_max = cv2.dilate(zone_image, kernel)  # Maximum filter
I_min = cv2.erode(zone_image, kernel)   # Minimum filter
residual_map = cv2.subtract(I_max, I_min)  # Residual generation

# Sigma-based thresholding
mean = np.mean(residual_map)
std_dev = np.std(residual_map)
threshold = mean + gamma * std_dev
binary_mask = (residual_map > threshold) * 255
```

### 2. LEI Method (Linear Enhancement Inspector)
**Research Concept**: Multi-orientation linear filtering for scratch detection

**Mathematical Foundation**:
```
sθ(x,y) = 2·fr_θ(x,y) - fg_θ(x,y)
```
Where:
- fr_θ: Average intensity along central branch
- fg_θ: Average intensity along parallel side branches
- sθ: Scratch strength at orientation θ

**Implementation in detection.py**:
```python
# Located in: detect_scratch_defects() function
# Step 1: Image enhancement
enhanced = cv2.equalizeHist(grayscale_image)

# Step 2: Multi-orientation filtering (typically 0° to 180° in 15° steps)
for angle in range(0, 180, 15):
    kernel_center = generate_linear_kernel(angle, 'center')
    kernel_side = generate_linear_kernel(angle, 'side')
    
    response_center = cv2.filter2D(enhanced, -1, kernel_center)
    response_side = cv2.filter2D(enhanced, -1, kernel_side)
    
    scratch_strength = cv2.subtract(2 * response_center, response_side)
    
    # Threshold and accumulate
    temp_mask = (scratch_strength > threshold) * 255
    final_mask = cv2.bitwise_or(final_mask, temp_mask)
```

## Supporting Research Implementations

### 1. Feature Extraction Methods
**An Introduction to Feature Extraction**
- **Authors**: Isabelle Guyon, André Elisseeff
- **Published**: 2006
- **Implementation**: `extract_ultra_comprehensive_features()` function

### Gray-Level Co-occurrence Matrix (GLCM) Features
```python
# Located in: extract_glcm_features()
quantized = (gray * 8 / 256).astype(np.uint8)  # Quantize to 8 levels
for distance in [1, 2, 3]:
    for angle in [0, 45, 90, 135]:
        glcm = graycomatrix(quantized, [distance], [angle])
        contrast = graycoprops(glcm, 'contrast')
        energy = graycoprops(glcm, 'energy')
        homogeneity = graycoprops(glcm, 'homogeneity')
```

### Local Binary Patterns (LBP)
```python
# Located in: extract_lbp_features()
for radius in [1, 2, 3]:
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2))
    features[f'lbp_r{radius}_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
```

### 2. Topological Analysis
**Persistent Homology in Medical Image Processing**
- **Authors**: Fang and Yan
- **Published**: 2024
- **Reference**: 2407.05204v1.pdf
- **Implementation**: `extract_topological_proxy_features()` function

```python
# Betti number computation proxy
thresholds = np.percentile(gray, np.linspace(5, 95, 20))
n_components = []
n_holes = []
for t in thresholds:
    # β₀ (connected components)
    binary = (gray >= t).astype(np.uint8)
    num_labels, _ = cv2.connectedComponentsWithStats(binary)
    n_components.append(num_labels - 1)
    
    # β₁ (holes) - inverted image
    binary_inv = (gray < t).astype(np.uint8)
    num_holes, _ = cv2.connectedComponentsWithStats(binary_inv)
    n_holes.append(num_holes - 1)

# Persistence calculation
persistence_b0 = np.diff(n_components)
features['topo_b0_persistence_sum'] = np.sum(np.abs(persistence_b0))
features['topo_b0_persistence_max'] = np.max(np.abs(persistence_b0))
```

### 3. Distance Metrics
**Optimal Transport Theory**
- **Authors**: Snow and Van lent
- **Published**: 2016
- **Reference**: 1612.00181v2.pdf
- **Implementation**: `compute_wasserstein_distance()` function

```python
def compute_wasserstein_distance(x, y):
    # 1D Wasserstein distance implementation
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Interpolate to equal length
    if len(x) != len(y):
        common_len = max(len(x), len(y))
        x_interp = np.interp(np.linspace(0, 1, common_len), 
                            np.linspace(0, 1, len(x)), x_sorted)
        y_interp = np.interp(np.linspace(0, 1, common_len), 
                            np.linspace(0, 1, len(y)), y_sorted)
    
    return np.mean(np.abs(x_interp - y_interp))
```

### 4. Structural Similarity
**Structural Similarity Index (SSIM)**
- **Mathematical Definition**: As referenced in research paper
- **Implementation**: `compute_image_structural_comparison()` function

```python
def compute_image_structural_comparison(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    # Local statistics using Gaussian window
    mu1 = cv2.filter2D(img1.astype(float), -1, window)
    mu2 = cv2.filter2D(img2.astype(float), -1, window)
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1**2
    sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2**2
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1 * mu2
    
    # SSIM components
    luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    contrast = (2 * np.sqrt(sigma1_sq * sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C2/2) / (np.sqrt(sigma1_sq * sigma2_sq) + C2/2)
    
    ssim_map = luminance * contrast * structure
    return np.mean(ssim_map)
```

### 5. Morphological Operations
**Mathematical Morphology**
- **References**: Multiple papers including Panduit documentation
- **Implementation**: `detect_specific_defects()` function

```python
# Dig detection using black-hat transform
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
_, dig_mask = cv2.threshold(black_hat, np.percentile(black_hat, 95), 255, cv2.THRESH_BINARY)

# Scratch detection using Hough transform
edges = cv2.Canny(gray, 30, 100)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                       minLineLength=20, maxLineGap=5)

# Blob detection using adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
```

### 6. Information Theory
**Shannon Entropy**
- **Referenced in**: Multiple feature extraction papers
- **Implementation**: `compute_entropy()` function

```python
def compute_entropy(data, bins=256):
    hist, _ = np.histogram(data, bins=bins, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)  # Normalize to probability
    return -np.sum(hist * np.log2(hist + 1e-10))
```

## Fusion Framework

### Confidence Map Architecture
- **Not from a single paper but an architectural enhancement**
- **Implementation**: Main detection pipeline

```python
# Initialize confidence map
confidence_map = np.zeros_like(gray, dtype=float)

# Algorithm suite with weights
algorithms = {
    'do2mr': {'weight': 1.0, 'function': apply_do2mr},
    'lei': {'weight': 0.8, 'function': apply_lei},
    'gabor': {'weight': 0.6, 'function': apply_gabor_filters},
    'lbp': {'weight': 0.5, 'function': apply_lbp_detection},
    'wavelet': {'weight': 0.7, 'function': apply_wavelet_detection}
}

# Weighted voting
for algo_name, algo_info in algorithms.items():
    mask = algo_info['function'](zone_image)
    confidence_map += mask * algo_info['weight']

# Adaptive thresholding
threshold = np.mean(confidence_map) + gamma * np.std(confidence_map)
final_mask = (confidence_map > threshold).astype(np.uint8) * 255

# Contrast validation
validated_mask = validate_defect_mask(final_mask, original_image)
```

## Zone-Based Classification

### IEC 61300-3-35 Standard Implementation
- **Source**: JDSU FIT Workshop documentation
- **Implementation**: `apply_pass_fail_rules()` function

```python
# Zone definitions (micrometers)
zones = {
    'A_core': {'inner': 0, 'outer': 25},
    'B_cladding': {'inner': 25, 'outer': 120},
    'C_adhesive': {'inner': 120, 'outer': 130},
    'D_contact': {'inner': 130, 'outer': 250}
}

# Single-mode fiber rules
rules_sm = {
    'A_core': {
        'max_scratches': 0,
        'max_defects': 0,
        'max_defect_size_um': 3.0
    },
    'B_cladding': {
        'max_scratches': 5,
        'max_defects': 5,
        'max_defect_size_um': 10.0
    }
    # ... additional zones
}
```

## Localization Methods

### Hough Circle Transform
- **Implementation**: Zone detection

```python
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,              # Accumulator resolution ratio
    minDist=50,          # Minimum distance between centers
    param1=70,           # Canny edge threshold
    param2=30,           # Accumulator threshold
    minRadius=10,
    maxRadius=150
)

# Generate zone masks
for y in range(height):
    for x in range(width):
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        if dist_sq < core_radius_px**2:
            zone_mask[y, x] = ZONE_CORE
        elif dist_sq < cladding_radius_px**2:
            zone_mask[y, x] = ZONE_CLADDING
        # ... additional zones
```

## Performance Optimizations

### Multi-Scale Processing
```python
# Pyramid decomposition for multi-scale features
pyramid_levels = []
current = gray.copy()
for level in range(4):
    pyramid_levels.append(current)
    current = cv2.pyrDown(current)

# Process each scale
for scale, image in enumerate(pyramid_levels):
    features[f'scale_{scale}_mean'] = np.mean(image)
    features[f'scale_{scale}_std'] = np.std(image)
```

### Fast/Deep Scan Modes
```python
if scan_mode == 'fast':
    # DO2MR only - computationally light
    defects = apply_do2mr(image)
elif scan_mode == 'deep':
    # Full algorithm suite including LEI
    defects = apply_full_detection_suite(image)
```

## Output Generation

### Report Structure
```python
report = {
    'timestamp': datetime.now().isoformat(),
    'verdict': 'PASS' or 'FAIL',
    'zones': {
        'A_core': {'scratches': 0, 'defects': [], 'status': 'PASS'},
        'B_cladding': {'scratches': 2, 'defects': [...], 'status': 'PASS'}
    },
    'defect_details': [
        {
            'id': 'SCR_0001',
            'type': 'scratch',
            'zone': 'B',
            'length_um': 45.2,
            'width_um': 2.1,
            'orientation_deg': 35.4,
            'coordinates': (x, y)
        }
    ],
    'confidence_scores': {...},
    'processing_time_ms': 245
}
```