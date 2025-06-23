#!/usr/bin/env python3
"""
Region Separation Analysis Script
Performs advanced region separation using correlational data and spatial trends
"""

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import networkx as nx
from scipy import stats, signal, ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class RegionSeparator:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.correlation_data = {}
        self.trend_data = {}
        self.features_df = None
        self.separation_results = {}
        
    def load_data(self):
        """Load all correlation matrices and trend data"""
        print("[1/7] Loading correlation matrices and trend data...")
        
        # Load correlation matrices
        corr_files = ['correlation_pearson.csv', 'correlation_spearman.csv', 
                      'mutual_information.csv']
        
        for file in corr_files:
            if (self.base_path / file).exists():
                self.correlation_data[file.replace('.csv', '')] = pd.read_csv(
                    self.base_path / file, index_col=0
                )
        
        # Load extracted features
        if (self.base_path / 'extracted_features.csv').exists():
            self.features_df = pd.read_csv(self.base_path / 'extracted_features.csv', index_col=0)
        
        # Parse trend data
        self.parse_trend_data()
        
    def parse_trend_data(self):
        """Parse spatial trend equations from trenddata.txt"""
        trend_file = self.base_path / 'trenddata.txt'
        if not trend_file.exists():
            return
            
        with open(trend_file, 'r') as f:
            content = f.read()
        
        current_image = None
        for line in content.split('\n'):
            if 'Results for:' in line:
                current_image = line.split('Results for: ')[1].strip(' ---')
                self.trend_data[current_image] = {}
            elif current_image and 'Classification' in line:
                self.trend_data[current_image]['classification'] = line.split(': ')[1].strip()
            elif current_image and 'Mean Grayscale' in line:
                self.trend_data[current_image]['mean_grayscale'] = float(line.split(': ')[1])