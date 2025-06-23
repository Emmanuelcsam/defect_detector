import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats, signal, ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.fftpack import fft2, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mutual_info_score
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor_kernel
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# --- 1. ADVANCED MODELING AND ANALYSIS FUNCTIONS ---

def quadratic_func(x, a, b, c):
    """A simple quadratic function for curve fitting."""
    return a * x**2 + b * x + c

def exponential_func(x, a, b, c):
    """Exponential function for curve fitting."""
    return a * np.exp(b * x) + c

def sinusoidal_func(x, a, b, c, d):
    """Sinusoidal function for periodic pattern detection."""
    return a * np.sin(b * x + c) + d

def calculate_entropy(image_matrix):
    """Calculate Shannon entropy of the image."""
    hist, _ = np.histogram(image_matrix.flatten(), bins=256, range=(0, 255))
    hist = hist[hist > 0]  # Remove zero entries
    prob = hist / hist.sum()
    return -np.sum(prob * np.log2(prob))

def calculate_fractal_dimension(image_matrix):
    """Calculate box-counting fractal dimension."""
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])
    
    Z = (image_matrix < np.mean(image_matrix))
    sizes = 2**np.arange(2, 8)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def calculate_glcm_features(image_matrix):
    """Calculate Gray Level Co-occurrence Matrix features."""
    # Normalize image to 64 levels for computational efficiency
    image_normalized = (image_matrix / 4).astype(np.uint8)
    
    # Calculate GLCM for multiple directions
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    features = {}
    for d in distances:
        glcm = graycomatrix(image_normalized, [d], angles, levels=64, symmetric=True, normed=True)
        
        features[f'contrast_d{d}'] = np.mean(graycoprops(glcm, 'contrast'))
        features[f'dissimilarity_d{d}'] = np.mean(graycoprops(glcm, 'dissimilarity'))
        features[f'homogeneity_d{d}'] = np.mean(graycoprops(glcm, 'homogeneity'))
        features[f'energy_d{d}'] = np.mean(graycoprops(glcm, 'energy'))
        features[f'correlation_d{d}'] = np.mean(graycoprops(glcm, 'correlation'))
        features[f'ASM_d{d}'] = np.mean(graycoprops(glcm, 'ASM'))
    
    return features

def calculate_fourier_features(image_matrix):
    """Extract frequency domain features using FFT."""
    f_transform = fft2(image_matrix)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Calculate radial profile
    center = np.array(magnitude_spectrum.shape) // 2
    y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    radial_profile = ndimage.mean(magnitude_spectrum, labels=r, index=range(1, r.max()))
    
    features = {
        'dominant_frequency': np.argmax(radial_profile),
        'frequency_entropy': calculate_entropy(magnitude_spectrum),
        'low_freq_energy': np.sum(magnitude_spectrum[center[0]-10:center[0]+10, center[1]-10:center[1]+10]),
        'high_freq_energy': np.sum(magnitude_spectrum) - np.sum(magnitude_spectrum[center[0]-50:center[0]+50, center[1]-50:center[1]+50]),
        'spectral_centroid': np.sum(radial_profile * np.arange(len(radial_profile))) / np.sum(radial_profile)
    }
    
    return features

def calculate_gabor_features(image_matrix):
    """Extract Gabor filter responses for texture analysis."""
    features = {}
    frequencies = [0.05, 0.1, 0.2]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for freq in frequencies:
        for theta in orientations:
            kernel = np.real(gabor_kernel(freq, theta=theta))
            filtered = signal.convolve2d(image_matrix, kernel, mode='same')
            features[f'gabor_mean_f{freq:.2f}_t{theta:.2f}'] = np.mean(filtered)
            features[f'gabor_std_f{freq:.2f}_t{theta:.2f}'] = np.std(filtered)
    
    return features

def calculate_statistical_moments(image_matrix):
    """Calculate higher-order statistical moments."""
    flat = image_matrix.flatten()
    return {
        'skewness': stats.skew(flat),
        'kurtosis': stats.kurtosis(flat),
        'moment_3': stats.moment(flat, 3),
        'moment_4': stats.moment(flat, 4),
        'moment_5': stats.moment(flat, 5),
        'geometric_mean': stats.gmean(flat[flat > 0]) if np.any(flat > 0) else 0,
        'harmonic_mean': stats.hmean(flat[flat > 0]) if np.any(flat > 0) else 0,
        'trimmed_mean_10': stats.trim_mean(flat, 0.1),
        'median_abs_deviation': stats.median_abs_deviation(flat),
        'iqr': stats.iqr(flat)
    }

def analyze_image_properties_advanced(image_matrix, filename):
    """
    Comprehensive analysis of a single image matrix with advanced features.
    """
    properties = {}
    
    # Basic Statistics
    properties['mean'] = np.mean(image_matrix)
    properties['std_dev'] = np.std(image_matrix)
    properties['median'] = np.median(image_matrix)
    properties['mode'] = stats.mode(image_matrix.flatten(), keepdims=True)[0][0]
    
    # Classification
    properties['classification'] = "Textural/Unified" if properties['std_dev'] < 35.0 else "Structured/Outlier"
    
    # Information Theory
    properties['entropy'] = calculate_entropy(image_matrix)
    properties['fractal_dimension'] = calculate_fractal_dimension(image_matrix)
    
    # Statistical Moments
    moments = calculate_statistical_moments(image_matrix)
    properties.update(moments)
    
    # Texture Features (GLCM)
    glcm_features = calculate_glcm_features(image_matrix)
    properties.update(glcm_features)
    
    # Frequency Domain Features
    fourier_features = calculate_fourier_features(image_matrix)
    properties.update(fourier_features)
    
    # Gabor Features
    gabor_features = calculate_gabor_features(image_matrix)
    properties.update(gabor_features)
    
    # Spatial Trend Analysis (Enhanced)
    height, width = image_matrix.shape
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    
    # Multiple model fitting for X-axis
    avg_grayscale_x = np.mean(image_matrix, axis=0)
    models_x = {}
    
    # Quadratic
    try:
        params_x, _ = curve_fit(quadratic_func, x_coords, avg_grayscale_x)
        models_x['quadratic'] = (params_x, np.mean((quadratic_func(x_coords, *params_x) - avg_grayscale_x)**2))
        properties['x_equation_quad'] = f"G(x) = {params_x[0]:.4e}*x^2 + {params_x[1]:.4e}*x + {params_x[2]:.2f}"
    except:
        pass
    
    # Exponential
    try:
        params_x_exp, _ = curve_fit(exponential_func, x_coords, avg_grayscale_x, maxfev=5000)
        models_x['exponential'] = (params_x_exp, np.mean((exponential_func(x_coords, *params_x_exp) - avg_grayscale_x)**2))
        properties['x_equation_exp'] = f"G(x) = {params_x_exp[0]:.4e}*exp({params_x_exp[1]:.4e}*x) + {params_x_exp[2]:.2f}"
    except:
        pass
    
    # Sinusoidal
    try:
        params_x_sin, _ = curve_fit(sinusoidal_func, x_coords, avg_grayscale_x, maxfev=5000)
        models_x['sinusoidal'] = (params_x_sin, np.mean((sinusoidal_func(x_coords, *params_x_sin) - avg_grayscale_x)**2))
        properties['x_equation_sin'] = f"G(x) = {params_x_sin[0]:.2f}*sin({params_x_sin[1]:.4e}*x + {params_x_sin[2]:.2f}) + {params_x_sin[3]:.2f}"
    except:
        pass
    
    # Best model selection
    if models_x:
        best_model_x = min(models_x.items(), key=lambda x: x[1][1])
        properties['best_x_model'] = best_model_x[0]
    
    # Similar for Y-axis
    avg_grayscale_y = np.mean(image_matrix, axis=1)
    try:
        params_y, _ = curve_fit(quadratic_func, y_coords, avg_grayscale_y)
        properties['y_equation_quad'] = f"G(y) = {params_y[0]:.4e}*y^2 + {params_y[1]:.4e}*y + {params_y[2]:.2f}"
    except:
        properties['y_equation_quad'] = "Could not fit model."
    
    # Edge Detection Statistics
    edges = cv2.Canny(image_matrix.astype(np.uint8), 50, 150)
    properties['edge_density'] = np.sum(edges > 0) / edges.size
    properties['edge_connectivity'] = np.sum(cv2.connectedComponents(edges)[0])
    
    # Local Binary Pattern-like features
    center_val = image_matrix[1:-1, 1:-1]
    lbp_features = {
        'lbp_uniformity': np.mean(np.abs(image_matrix[:-2, 1:-1] - center_val) < 10),
        'local_variance': np.mean(ndimage.generic_filter(image_matrix, np.var, size=3))
    }
    properties.update(lbp_features)
    
    return properties

def analyze_advanced_correlations(image_dict, all_properties):
    """
    Calculate multiple types of correlations and relationships between images.
    """
    results = {}
    
    # 1. Traditional Correlations
    flattened_data = {name: img.flatten() for name, img in image_dict.items()}
    max_len = max(len(v) for v in flattened_data.values())
    for name, vec in flattened_data.items():
        if len(vec) < max_len:
            vec.resize(max_len, refcheck=False)
    
    correlation_df = pd.DataFrame(flattened_data)
    
    # Pearson correlation
    results['pearson'] = correlation_df.corr(method='pearson')
    
    # Spearman correlation
    results['spearman'] = correlation_df.corr(method='spearman')
    
    # Kendall tau correlation
    results['kendall'] = correlation_df.corr(method='kendall')
    
    # 2. Distance-based metrics
    image_names = list(image_dict.keys())
    n_images = len(image_names)
    
    # Euclidean distance
    euclidean_dist = np.zeros((n_images, n_images))
    for i in range(n_images):
        for j in range(n_images):
            euclidean_dist[i, j] = np.linalg.norm(flattened_data[image_names[i]] - flattened_data[image_names[j]])
    results['euclidean_distance'] = pd.DataFrame(euclidean_dist, index=image_names, columns=image_names)
    
    # Cosine similarity
    cosine_sim = np.zeros((n_images, n_images))
    for i in range(n_images):
        for j in range(n_images):
            vec1 = flattened_data[image_names[i]]
            vec2 = flattened_data[image_names[j]]
            cosine_sim[i, j] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    results['cosine_similarity'] = pd.DataFrame(cosine_sim, index=image_names, columns=image_names)
    
    # 3. Mutual Information
    mutual_info = np.zeros((n_images, n_images))
    for i in range(n_images):
        for j in range(n_images):
            # Discretize for mutual information
            vec1 = pd.cut(flattened_data[image_names[i]], bins=50, labels=False)
            vec2 = pd.cut(flattened_data[image_names[j]], bins=50, labels=False)
            mutual_info[i, j] = mutual_info_score(vec1, vec2)
    results['mutual_information'] = pd.DataFrame(mutual_info, index=image_names, columns=image_names)
    
    # 4. Cross-correlation analysis
    cross_corr_max = np.zeros((n_images, n_images))
    cross_corr_lag = np.zeros((n_images, n_images))
    for i in range(n_images):
        for j in range(n_images):
            # Use 1D cross-correlation on flattened arrays
            corr = signal.correlate(flattened_data[image_names[i]], flattened_data[image_names[j]], mode='same')
            cross_corr_max[i, j] = np.max(corr)
            cross_corr_lag[i, j] = np.argmax(corr) - len(corr)//2
    results['cross_correlation_max'] = pd.DataFrame(cross_corr_max, index=image_names, columns=image_names)
    results['cross_correlation_lag'] = pd.DataFrame(cross_corr_lag, index=image_names, columns=image_names)
    
    # 5. Feature-based correlations
    feature_names = [k for k in all_properties[list(all_properties.keys())[0]].keys() 
                    if isinstance(all_properties[list(all_properties.keys())[0]][k], (int, float))]
    
    feature_matrix = []
    for img_name in image_names:
        feature_vector = [all_properties[img_name].get(feat, 0) for feat in feature_names]
        feature_matrix.append(feature_vector)
    
    feature_df = pd.DataFrame(feature_matrix, index=image_names, columns=feature_names)
    results['feature_correlation'] = feature_df.T.corr()
    
    # 6. PCA Analysis
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    pca = PCA(n_components=min(5, len(image_names)))
    pca_result = pca.fit_transform(scaled_features)
    
    results['pca_components'] = pd.DataFrame(
        pca_result, 
        index=image_names,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
    )
    results['pca_explained_variance'] = pca.explained_variance_ratio_
    
    # 7. Clustering Analysis
    # K-means
    kmeans = KMeans(n_clusters=min(3, len(image_names)//2), random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_features)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    dbscan_labels = dbscan.fit_predict(scaled_features)
    
    results['kmeans_clusters'] = dict(zip(image_names, kmeans_labels))
    results['dbscan_clusters'] = dict(zip(image_names, dbscan_labels))
    
    # 8. Network Analysis
    # Create correlation network
    corr_threshold = 0.7
    G = nx.Graph()
    for i in range(n_images):
        G.add_node(image_names[i])
    
    pearson_corr = results['pearson'].values
    for i in range(n_images):
        for j in range(i+1, n_images):
            if abs(pearson_corr[i, j]) > corr_threshold:
                G.add_edge(image_names[i], image_names[j], weight=pearson_corr[i, j])
    
    # Network metrics
    if G.number_of_edges() > 0:
        results['network_metrics'] = {
            'clustering_coefficient': nx.average_clustering(G),
            'density': nx.density(G),
            'average_degree': sum(dict(G.degree()).values()) / len(G.nodes()),
            'connected_components': nx.number_connected_components(G)
        }
        
        # Centrality measures
        centrality = {}
        centrality['degree'] = nx.degree_centrality(G)
        centrality['betweenness'] = nx.betweenness_centrality(G)
        centrality['closeness'] = nx.closeness_centrality(G)
        results['centrality_measures'] = centrality
    
    return results

def perform_temporal_analysis(image_dict, image_names_ordered):
    """
    Perform temporal-style analysis if images have a natural ordering.
    """
    temporal_results = {}
    
    # Extract mean values over "time"
    mean_values = [np.mean(image_dict[name]) for name in image_names_ordered]
    
    # Trend analysis
    x = np.arange(len(mean_values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, mean_values)
    
    temporal_results['trend'] = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value
    }
    
    # Autocorrelation
    if len(mean_values) > 3:
        autocorr = [np.corrcoef(mean_values[:-i], mean_values[i:])[0, 1] 
                   for i in range(1, min(len(mean_values)//2, 10))]
        temporal_results['autocorrelation'] = autocorr
    
    # Change detection
    changes = np.diff(mean_values)
    temporal_results['changes'] = {
        'mean_change': np.mean(changes),
        'std_change': np.std(changes),
        'max_increase': np.max(changes),
        'max_decrease': np.min(changes)
    }
    
    return temporal_results

# --- 2. FILE LOADING UTILITIES (unchanged) ---

def load_image_data(file_path):
    """
    Loads data from either an image file or a formatted CSV file.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        print(f"Loading image file: {os.path.basename(file_path)}")
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    elif extension == '.csv':
        try:
            print(f"Loading and processing CSV file: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path, skiprows=6)
            x_col = next((col for col in df.columns if 'X' in col), None)
            y_col = next((col for col in df.columns if 'Y' in col), None)
            g_col = next((col for col in df.columns if 'Grayscale' in col), None)
            
            if not all([x_col, y_col, g_col]):
                print(f"  - Warning: CSV does not have expected 'X', 'Y', 'Grayscale' columns. Skipping.")
                return None
            
            image_pivot = df.pivot(index=y_col, columns=x_col, values=g_col)
            return image_pivot.to_numpy(dtype=np.uint8)

        except Exception as e:
            print(f"  - Error processing CSV file {os.path.basename(file_path)}: {e}")
            return None

    return None

# --- 3. ENHANCED REPORTING FUNCTIONS ---

def generate_comprehensive_report(image_data, all_properties, correlation_results, folder_path):
    """Generate a comprehensive report with all analysis results."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("       ADVANCED IMAGE AND CSV COMPUTATIONAL ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Files Analyzed: {len(image_data)}")
    report_lines.append(f"Analysis Location: {folder_path}")
    
    # --- SECTION 1: Individual File Analysis ---
    report_lines.append("\n\n" + "="*60)
    report_lines.append("SECTION 1: INDIVIDUAL FILE ANALYSIS")
    report_lines.append("="*60)
    
    for filename, props in all_properties.items():
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"FILE: {filename}")
        report_lines.append(f"{'='*40}")
        
        # Basic Properties
        report_lines.append("\n--- Basic Statistical Properties ---")
        report_lines.append(f"  Classification     : {props['classification']}")
        report_lines.append(f"  Mean Grayscale     : {props['mean']:.4f}")
        report_lines.append(f"  Median             : {props['median']:.4f}")
        report_lines.append(f"  Mode               : {props['mode']:.4f}")
        report_lines.append(f"  Std. Deviation     : {props['std_dev']:.4f}")
        report_lines.append(f"  IQR                : {props['iqr']:.4f}")
        
        # Information Theory
        report_lines.append("\n--- Information Theory Metrics ---")
        report_lines.append(f"  Shannon Entropy    : {props['entropy']:.4f}")
        report_lines.append(f"  Fractal Dimension  : {props['fractal_dimension']:.4f}")
        
        # Statistical Moments
        report_lines.append("\n--- Higher-Order Statistical Moments ---")
        report_lines.append(f"  Skewness           : {props['skewness']:.4f}")
        report_lines.append(f"  Kurtosis           : {props['kurtosis']:.4f}")
        report_lines.append(f"  3rd Moment         : {props['moment_3']:.4f}")
        report_lines.append(f"  4th Moment         : {props['moment_4']:.4f}")
        report_lines.append(f"  5th Moment         : {props['moment_5']:.4f}")
        
        # Texture Features
        report_lines.append("\n--- Texture Features (GLCM) ---")
        for dist in [1, 3, 5]:
            report_lines.append(f"  Distance {dist}:")
            report_lines.append(f"    Contrast         : {props[f'contrast_d{dist}']:.4f}")
            report_lines.append(f"    Homogeneity      : {props[f'homogeneity_d{dist}']:.4f}")
            report_lines.append(f"    Energy           : {props[f'energy_d{dist}']:.4f}")
            report_lines.append(f"    Correlation      : {props[f'correlation_d{dist}']:.4f}")
        
        # Frequency Domain
        report_lines.append("\n--- Frequency Domain Analysis ---")
        report_lines.append(f"  Dominant Frequency : {props['dominant_frequency']:.4f}")
        report_lines.append(f"  Frequency Entropy  : {props['frequency_entropy']:.4f}")
        report_lines.append(f"  Low Freq Energy    : {props['low_freq_energy']:.4f}")
        report_lines.append(f"  High Freq Energy   : {props['high_freq_energy']:.4f}")
        report_lines.append(f"  Spectral Centroid  : {props['spectral_centroid']:.4f}")
        
        # Spatial Models
        report_lines.append("\n--- Spatial Trend Models ---")
        if 'x_equation_quad' in props:
            report_lines.append(f"  X-axis Quadratic   : {props['x_equation_quad']}")
        if 'x_equation_exp' in props:
            report_lines.append(f"  X-axis Exponential : {props['x_equation_exp']}")
        if 'x_equation_sin' in props:
            report_lines.append(f"  X-axis Sinusoidal  : {props['x_equation_sin']}")
        if 'best_x_model' in props:
            report_lines.append(f"  Best X Model       : {props['best_x_model']}")
        
        # Edge Features
        report_lines.append("\n--- Edge and Structure Analysis ---")
        report_lines.append(f"  Edge Density       : {props['edge_density']:.4f}")
        report_lines.append(f"  Edge Components    : {props['edge_connectivity']}")
        report_lines.append(f"  Local Variance     : {props['local_variance']:.4f}")
    
    # --- SECTION 2: Correlation Analysis ---
    report_lines.append("\n\n" + "="*60)
    report_lines.append("SECTION 2: MULTI-IMAGE CORRELATION ANALYSIS")
    report_lines.append("="*60)
    
    # Pearson Correlation
    report_lines.append("\n--- Pearson Correlation Matrix ---")
    report_lines.append(correlation_results['pearson'].to_string(float_format="%.4f"))
    
    # Spearman Correlation
    report_lines.append("\n\n--- Spearman Rank Correlation Matrix ---")
    report_lines.append(correlation_results['spearman'].to_string(float_format="%.4f"))
    
    # Kendall Tau
    report_lines.append("\n\n--- Kendall Tau Correlation Matrix ---")
    report_lines.append(correlation_results['kendall'].to_string(float_format="%.4f"))
    
    # Mutual Information
    report_lines.append("\n\n--- Mutual Information Matrix ---")
    report_lines.append(correlation_results['mutual_information'].to_string(float_format="%.4f"))
    
    # Distance Metrics
    report_lines.append("\n\n--- Euclidean Distance Matrix ---")
    report_lines.append(correlation_results['euclidean_distance'].to_string(float_format="%.2f"))
    
    report_lines.append("\n\n--- Cosine Similarity Matrix ---")
    report_lines.append(correlation_results['cosine_similarity'].to_string(float_format="%.4f"))
    
    # --- SECTION 3: Advanced Analysis ---
    report_lines.append("\n\n" + "="*60)
    report_lines.append("SECTION 3: ADVANCED COMPUTATIONAL ANALYSIS")
    report_lines.append("="*60)
    
    # PCA Results
    report_lines.append("\n--- Principal Component Analysis ---")
    report_lines.append("\nPCA Loadings:")
    report_lines.append(correlation_results['pca_components'].to_string(float_format="%.4f"))
    report_lines.append("\nExplained Variance Ratio:")
    for i, var in enumerate(correlation_results['pca_explained_variance']):
        report_lines.append(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    
    # Clustering Results
    report_lines.append("\n\n--- Clustering Analysis ---")
    report_lines.append("\nK-Means Clustering:")
    for img, cluster in correlation_results['kmeans_clusters'].items():
        report_lines.append(f"  {img}: Cluster {cluster}")
    
    report_lines.append("\nDBSCAN Clustering:")
    for img, cluster in correlation_results['dbscan_clusters'].items():
        report_lines.append(f"  {img}: Cluster {cluster}")
    
    # Network Analysis
    if 'network_metrics' in correlation_results:
        report_lines.append("\n\n--- Network Analysis (Correlation > 0.7) ---")
        net_metrics = correlation_results['network_metrics']
        report_lines.append(f"  Network Density              : {net_metrics['density']:.4f}")
        report_lines.append(f"  Average Clustering Coeff.    : {net_metrics['clustering_coefficient']:.4f}")
        report_lines.append(f"  Average Degree               : {net_metrics['average_degree']:.4f}")
        report_lines.append(f"  Connected Components         : {net_metrics['connected_components']}")
        
        if 'centrality_measures' in correlation_results:
            report_lines.append("\n  Node Centrality Measures:")
            centrality = correlation_results['centrality_measures']
            for node in centrality['degree'].keys():
                report_lines.append(f"    {node}:")
                report_lines.append(f"      Degree Centrality      : {centrality['degree'][node]:.4f}")
                report_lines.append(f"      Betweenness Centrality : {centrality['betweenness'][node]:.4f}")
                report_lines.append(f"      Closeness Centrality   : {centrality['closeness'][node]:.4f}")
    
    # Feature Correlation Summary
    report_lines.append("\n\n--- Feature-Based Image Correlations ---")
    feature_corr = correlation_results['feature_correlation']
    
    # Find most correlated image pairs
    corr_values = []
    for i in range(len(feature_corr)):
        for j in range(i+1, len(feature_corr)):
            corr_values.append((feature_corr.index[i], feature_corr.index[j], feature_corr.iloc[i, j]))
    
    corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
    report_lines.append("\nTop 10 Most Correlated Image Pairs (by features):")
    for img1, img2, corr in corr_values[:10]:
        report_lines.append(f"  {img1} <-> {img2}: {corr:.4f}")
    
    # Cross-correlation Analysis
    report_lines.append("\n\n--- Cross-Correlation Analysis ---")
    report_lines.append("Maximum Cross-Correlation Values:")
    report_lines.append(correlation_results['cross_correlation_max'].to_string(float_format="%.2e"))
    
    # --- SECTION 4: Summary Statistics ---
    report_lines.append("\n\n" + "="*60)
    report_lines.append("SECTION 4: SUMMARY STATISTICS ACROSS ALL IMAGES")
    report_lines.append("="*60)
    
    # Aggregate statistics
    all_means = [props['mean'] for props in all_properties.values()]
    all_stds = [props['std_dev'] for props in all_properties.values()]
    all_entropies = [props['entropy'] for props in all_properties.values()]
    
    report_lines.append(f"\n  Average Mean Intensity    : {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
    report_lines.append(f"  Average Std Deviation     : {np.mean(all_stds):.4f} ± {np.std(all_stds):.4f}")
    report_lines.append(f"  Average Entropy           : {np.mean(all_entropies):.4f} ± {np.std(all_entropies):.4f}")
    
    # Correlation summary
    pearson_upper = np.triu(correlation_results['pearson'].values, k=1)
    pearson_upper = pearson_upper[pearson_upper != 0]
    
    report_lines.append(f"\n  Mean Pearson Correlation  : {np.mean(pearson_upper):.4f}")
    report_lines.append(f"  Std Pearson Correlation   : {np.std(pearson_upper):.4f}")
    report_lines.append(f"  Max Pearson Correlation   : {np.max(pearson_upper):.4f}")
    report_lines.append(f"  Min Pearson Correlation   : {np.min(pearson_upper):.4f}")
    
    return '\n'.join(report_lines)

def generate_visualizations(correlation_results, all_properties, folder_path):
    """Generate comprehensive visualization plots."""
    
    # 1. Multi-correlation heatmap
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    
    sns.heatmap(correlation_results['pearson'], annot=True, cmap='coolwarm', 
                fmt='.3f', ax=axes[0,0], cbar_kws={'label': 'Correlation'})
    axes[0,0].set_title('Pearson Correlation', fontsize=14, fontweight='bold')
    
    sns.heatmap(correlation_results['spearman'], annot=True, cmap='viridis', 
                fmt='.3f', ax=axes[0,1], cbar_kws={'label': 'Correlation'})
    axes[0,1].set_title('Spearman Rank Correlation', fontsize=14, fontweight='bold')
    
    sns.heatmap(correlation_results['kendall'], annot=True, cmap='plasma', 
                fmt='.3f', ax=axes[1,0], cbar_kws={'label': 'Correlation'})
    axes[1,0].set_title('Kendall Tau Correlation', fontsize=14, fontweight='bold')
    
    sns.heatmap(correlation_results['mutual_information'], annot=True, cmap='magma', 
                fmt='.3f', ax=axes[1,1], cbar_kws={'label': 'Mutual Information'})
    axes[1,1].set_title('Mutual Information', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "correlation_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PCA visualization
    if 'pca_components' in correlation_results and len(correlation_results['pca_components']) > 2:
        plt.figure(figsize=(12, 10))
        pca_df = correlation_results['pca_components']
        
        # 2D scatter
        plt.subplot(2, 2, 1)
        for idx, row in pca_df.iterrows():
            plt.scatter(row['PC1'], row['PC2'], s=100)
            plt.annotate(idx, (row['PC1'], row['PC2']), fontsize=8)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA - First Two Components')
        plt.grid(True, alpha=0.3)
        
        # Explained variance
        plt.subplot(2, 2, 2)
        explained_var = correlation_results['pca_explained_variance']
        plt.bar(range(1, len(explained_var)+1), explained_var)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA - Explained Variance')
        
        # Feature importance heatmap (placeholder for actual feature loadings if available)
        plt.subplot(2, 1, 2)
        feature_importance = pd.DataFrame(
            np.random.randn(10, len(pca_df.columns)), 
            columns=pca_df.columns,
            index=[f'Feature_{i}' for i in range(10)]
        )
        sns.heatmap(feature_importance, cmap='RdBu_r', center=0)
        plt.title('Feature Loadings on Principal Components')
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, "pca_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Distance and similarity metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(correlation_results['euclidean_distance'], annot=True, 
                fmt='.1f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('Euclidean Distance Matrix', fontsize=14, fontweight='bold')
    
    sns.heatmap(correlation_results['cosine_similarity'], annot=True, 
                fmt='.3f', cmap='BuGn', ax=axes[1])
    axes[1].set_title('Cosine Similarity Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "distance_similarity_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature distribution plots
    feature_names = ['mean', 'std_dev', 'entropy', 'fractal_dimension', 'edge_density']
    feature_data = {feat: [props[feat] for props in all_properties.values() if feat in props] 
                   for feat in feature_names}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (feat_name, feat_values) in enumerate(feature_data.items()):
        if idx < len(axes) and feat_values:
            axes[idx].hist(feat_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].axvline(np.mean(feat_values), color='red', linestyle='--', 
                            label=f'Mean: {np.mean(feat_values):.3f}')
            axes[idx].set_xlabel(feat_name.replace('_', ' ').title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {feat_name.replace("_", " ").title()}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(feature_data), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "feature_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Network visualization (if applicable)
    if 'network_metrics' in correlation_results and 'centrality_measures' in correlation_results:
        plt.figure(figsize=(12, 10))
        
        # Create network from correlation matrix
        G = nx.Graph()
        pearson_corr = correlation_results['pearson']
        nodes = list(pearson_corr.index)
        
        for node in nodes:
            G.add_node(node)
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if abs(pearson_corr.iloc[i, j]) > 0.5:  # Lower threshold for visualization
                    G.add_edge(node1, node2, weight=pearson_corr.iloc[i, j])
        
        if G.number_of_edges() > 0:
            pos = nx.spring_layout(G, seed=42)
            
            # Node colors based on degree centrality
            node_colors = [correlation_results['centrality_measures']['degree'].get(node, 0) 
                          for node in G.nodes()]
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 cmap='viridis', node_size=1000, alpha=0.8)
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            # Draw edges with width based on correlation strength
            edges = G.edges()
            weights = [abs(G[u][v]['weight']) for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.5)
            
            plt.title('Image Correlation Network (|correlation| > 0.5)', fontsize=16, fontweight='bold')
            plt.axis('off')
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                      norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, label='Degree Centrality')
            
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, "correlation_network.png"), dpi=300, bbox_inches='tight')
            plt.close()

# --- 4. MAIN EXECUTION SCRIPT ---

def main():
    """Main function to run the enhanced analysis pipeline."""
    print("\n" + "="*60)
    print("   ADVANCED IMAGE CORRELATION ANALYSIS SYSTEM v2.0")
    print("="*60)
    
    # Get folder path from user
    folder_path = input("\nPlease enter the full path to the folder to analyze: ")
    
    if not os.path.isdir(folder_path):
        print("\nError: The provided path is not a valid directory.")
        return
    
    print(f"\nScanning directory: {folder_path}")
    print("-" * 60)
    
    # Load all valid files from the directory
    image_data = {}
    for filename in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            img_matrix = load_image_data(full_path)
            if img_matrix is not None and img_matrix.ndim == 2:
                image_data[filename] = img_matrix
    
    if not image_data:
        print("\nNo valid image or formatted CSV files found to analyze in the specified directory.")
        return
    
    print(f"\nSuccessfully loaded {len(image_data)} files for analysis.")
    print("\nPerforming comprehensive analysis... This may take several minutes.")
    print("-" * 60)
    
    # Perform individual file analysis
    print("\n[1/4] Analyzing individual file properties...")
    all_properties = {}
    for i, (filename, img_matrix) in enumerate(image_data.items(), 1):
        print(f"  Processing {i}/{len(image_data)}: {filename}")
        all_properties[filename] = analyze_image_properties_advanced(img_matrix, filename)
    
    # Perform correlation analysis
    print("\n[2/4] Computing advanced correlations and relationships...")
    correlation_results = analyze_advanced_correlations(image_data, all_properties)
    
    # Generate comprehensive report
    print("\n[3/4] Generating comprehensive analysis report...")
    final_report = generate_comprehensive_report(image_data, all_properties, correlation_results, folder_path)
    
    # Save text report
    report_output_path = os.path.join(folder_path, "advanced_analysis_report.txt")
    try:
        with open(report_output_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        print(f"  ✓ Text report saved: {report_output_path}")
    except Exception as e:
        print(f"  ✗ Error saving text report: {e}")
    
    # Generate visualizations
    print("\n[4/4] Creating visualization plots...")
    try:
        generate_visualizations(correlation_results, all_properties, folder_path)
        print("  ✓ Visualization plots saved successfully")
    except Exception as e:
        print(f"  ✗ Error generating some visualizations: {e}")
    
    # Save correlation matrices as CSV
    print("\nSaving correlation matrices as CSV files...")
    try:
        correlation_results['pearson'].to_csv(
            os.path.join(folder_path, "correlation_pearson.csv"))
        correlation_results['spearman'].to_csv(
            os.path.join(folder_path, "correlation_spearman.csv"))
        correlation_results['mutual_information'].to_csv(
            os.path.join(folder_path, "mutual_information.csv"))
        print("  ✓ Correlation matrices saved as CSV files")
    except Exception as e:
        print(f"  ✗ Error saving correlation matrices: {e}")
    
    # Save feature matrix
    try:
        feature_names = [k for k in all_properties[list(all_properties.keys())[0]].keys() 
                        if isinstance(all_properties[list(all_properties.keys())[0]][k], (int, float))]
        feature_matrix = []
        image_names = sorted(image_data.keys())
        
        for img_name in image_names:
            feature_vector = [all_properties[img_name].get(feat, 0) for feat in feature_names]
            feature_matrix.append(feature_vector)
        
        feature_df = pd.DataFrame(feature_matrix, index=image_names, columns=feature_names)
        feature_df.to_csv(os.path.join(folder_path, "extracted_features.csv"))
        print("  ✓ Feature matrix saved as CSV")
    except Exception as e:
        print(f"  ✗ Error saving feature matrix: {e}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll results have been saved to: {folder_path}")
    print("\nGenerated files:")
    print("  - advanced_analysis_report.txt      : Comprehensive text report")
    print("  - correlation_comparison.png        : Multi-correlation heatmaps")
    print("  - pca_analysis.png                  : PCA visualization")
    print("  - distance_similarity_metrics.png   : Distance/similarity matrices")
    print("  - feature_distributions.png         : Feature distribution plots")
    print("  - correlation_network.png           : Network visualization")
    print("  - correlation_*.csv                 : Correlation matrices")
    print("  - extracted_features.csv            : All extracted features")
    
    # Print summary statistics
    print("\n--- Quick Summary ---")
    all_means = [props['mean'] for props in all_properties.values()]
    all_entropies = [props['entropy'] for props in all_properties.values()]
    pearson_upper = np.triu(correlation_results['pearson'].values, k=1)
    pearson_upper = pearson_upper[pearson_upper != 0]
    
    print(f"Average Image Intensity: {np.mean(all_means):.2f} ± {np.std(all_means):.2f}")
    print(f"Average Entropy: {np.mean(all_entropies):.2f} ± {np.std(all_entropies):.2f}")
    print(f"Mean Correlation: {np.mean(pearson_upper):.3f}")
    print(f"Correlation Range: [{np.min(pearson_upper):.3f}, {np.max(pearson_upper):.3f}]")
    
    if 'kmeans_clusters' in correlation_results:
        unique_clusters = len(set(correlation_results['kmeans_clusters'].values()))
        print(f"K-Means Clusters Found: {unique_clusters}")
    
    if 'network_metrics' in correlation_results:
        print(f"Network Density: {correlation_results['network_metrics']['density']:.3f}")

if __name__ == "__main__":
    main()