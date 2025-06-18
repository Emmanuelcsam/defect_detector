#!/usr/bin/env python3
"""
Advanced Image Processing Pipeline GUI
======================================
A comprehensive UI for creating, managing, and executing image processing pipelines.
Combines the best features from all previous attempts with enhanced error handling.

Key Features:
- Automatic script cleaning to handle Unicode errors
- Dynamic function loading with robust error handling
- Visual pipeline builder with drag-and-drop reordering
- Real-time display of executing script names
- Advanced zoom/pan functionality
- Parameter editing with type detection
- Pipeline save/load functionality
- Asynchronous processing to prevent UI freezing
- **NEW:** Favorites and Recently Used function tracking
- **NEW:** Right-click to open scripts in VS Code
- **NEW:** Hard Refresh button to reset UI state
"""

import sys
import os
import re
import json
import importlib.util
import inspect
import traceback
import subprocess  # Added for opening files in VS Code
from datetime import datetime  # Added for recently used tracking
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
import cv2

# Qt imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QSplitter, QLineEdit, QMessageBox, QScrollArea, QGroupBox,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QDialog, QFormLayout,
    QDialogButtonBox, QTextEdit, QProgressBar, QStatusBar, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMenu  # Added for context menu
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor, QCursor


# --- NEW: User Preferences Manager ---
class UserPreferences:
    """Handles saving and loading user preferences like favorites and recently used."""
    def __init__(self, filename="user_prefs.json"):
        self.filepath = Path(filename)
        self.data = {
            'favorites': [],
            'recently_used': {}
        }
        self.load()

    def load(self):
        """Load preferences from a JSON file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    loaded_data = json.load(f)
                    # Ensure keys exist for robustness
                    self.data['favorites'] = loaded_data.get('favorites', [])
                    self.data['recently_used'] = loaded_data.get('recently_used', {})
            except (json.JSONDecodeError, IOError):
                print("Could not load user preferences, starting fresh.")
                self.data = {'favorites': [], 'recently_used': {}}

    def save(self):
        """Save current preferences to the JSON file."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            print(f"Error saving preferences: {e}")

    def add_favorite(self, func_name: str):
        """Add a function to favorites."""
        if func_name not in self.data['favorites']:
            self.data['favorites'].append(func_name)
            self.save()

    def remove_favorite(self, func_name: str):
        """Remove a function from favorites."""
        if func_name in self.data['favorites']:
            self.data['favorites'].remove(func_name)
            self.save()

    def is_favorite(self, func_name: str) -> bool:
        """Check if a function is a favorite."""
        return func_name in self.data['favorites']

    def add_recently_used(self, func_name: str):
        """Record that a function was recently used by updating its timestamp."""
        self.data['recently_used'][func_name] = datetime.now().isoformat()
        self.save()

    def get_recently_used(self) -> List[str]:
        """Get a list of functions sorted by most recently used."""
        sorted_items = sorted(self.data['recently_used'].items(), key=lambda item: item[1], reverse=True)
        return [item[0] for item in sorted_items]


class ImageViewer(QScrollArea):
    """Enhanced image viewer with smooth zoom and pan"""
    
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setWidget(self.image_label)
        
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.pan_start = None
        self.zoom_point = None
        
        # Enable mouse tracking for zoom
        self.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
        
    def set_image(self, image: np.ndarray):
        """Set the image to display"""
        if image is None:
            self.image_label.clear()
            self.original_pixmap = None
            return
        
        # Convert numpy array to QImage
        if len(image.shape) == 2:  # Grayscale
            height, width = image.shape
            q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        else:  # Color (BGR to RGB)
            height, width, channel = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_image.data, width, height, channel * width, QImage.Format_RGB888)
        
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.update_display()
        
    def update_display(self):
        """Update the displayed image with current scale"""
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                int(self.original_pixmap.width() * self.scale_factor),
                int(self.original_pixmap.height() * self.scale_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming centered on cursor"""
        if not self.original_pixmap:
            return
        
        # Get cursor position relative to the image
        cursor_pos = event.pos()
        
        # Calculate zoom
        zoom_in_factor = 1.25
        zoom_out_factor = 0.8
        old_scale = self.scale_factor
        
        if event.angleDelta().y() > 0:
            self.scale_factor *= zoom_in_factor
        else:
            self.scale_factor *= zoom_out_factor
            
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))
        
        # Update display
        self.update_display()
        
        # Adjust scrollbars to keep cursor position stable
        if old_scale != self.scale_factor:
            scale_delta = self.scale_factor / old_scale
            
            # Calculate the new scroll positions
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            
            new_h_value = int((h_bar.value() + cursor_pos.x()) * scale_delta - cursor_pos.x())
            new_v_value = int((v_bar.value() + cursor_pos.y()) * scale_delta - cursor_pos.y())
            
            h_bar.setValue(new_h_value)
            v_bar.setValue(new_v_value)
            
    def mousePressEvent(self, event):
        """Start panning on middle mouse button"""
        if event.button() == Qt.MiddleButton:
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            
    def mouseReleaseEvent(self, event):
        """Stop panning"""
        if event.button() == Qt.MiddleButton:
            self.pan_start = None
            self.setCursor(Qt.ArrowCursor)
            
    def mouseMoveEvent(self, event):
        """Handle panning"""
        if self.pan_start:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.scale_factor = 1.0
        self.update_display()
        
    def zoom_to_fit(self):
        """Zoom to fit the image in the viewport"""
        if not self.original_pixmap:
            return
            
        viewport_size = self.viewport().size()
        image_size = self.original_pixmap.size()
        
        scale_x = viewport_size.width() / image_size.width()
        scale_y = viewport_size.height() / image_size.height()
        
        self.scale_factor = min(scale_x, scale_y) * 0.95  # 95% to leave some margin
        self.update_display()


class ScriptCleaner:
    """Cleans scripts with Unicode errors and creates compatible wrappers"""
    
    @staticmethod
    def clean_script_content(content: str) -> str:
        """Remove hardcoded paths and fix common issues"""
        # Remove hardcoded paths
        content = re.sub(r'[a-zA-Z]:\\[^"\'\s\n]+', '', content)
        content = re.sub(r'img_path\s*=\s*["\'][^"\']*["\']', '', content)
        content = re.sub(r'image_path\s*=\s*["\'][^"\']*["\']', '', content)
        content = re.sub(r'base_path\s*=\s*[^\n]+', '', content)
        
        return content
        
    @staticmethod
    def create_process_image_wrapper(script_content: str, script_name: str) -> str:
        """Create a process_image function wrapper from script content"""
        # Try to identify the main processing logic
        if 'def process_image' in script_content:
            return script_content  # Already has the right format
            
        # Extract the core processing operations
        operations = []
        op_patterns = {
            'cv2.GaussianBlur': 'gaussian_blur',
            'cv2.Canny': 'edge_detection',
            'cv2.threshold': 'threshold',
            'cv2.equalizeHist': 'histogram_equalization',
            'cv2.morphologyEx': 'morphology',
            'cv2.HoughCircles': 'circle_detection',
            'cv2.medianBlur': 'median_filter',
            'cv2.Sobel': 'sobel_edge',
            'cv2.Laplacian': 'laplacian_edge',
            'cv2.createCLAHE': 'clahe',
            'cv2.cvtColor.*GRAY': 'grayscale',
        }
        
        detected_ops = []
        for pattern, op_name in op_patterns.items():
            if re.search(pattern, script_content):
                detected_ops.append(op_name)
                
        # Create a generic wrapper
        wrapper = f'''"""
Auto-generated wrapper for {script_name}
Detected operations: {', '.join(detected_ops) if detected_ops else 'unknown'}
"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """Process image using {script_name} logic"""
    try:
        # Default implementation - modify based on original script
        result = image.copy()
        
        # Add your processing here based on the original script
        {ScriptCleaner._generate_default_processing(detected_ops)}
        
        return result
    except Exception as e:
        print(f"Error in {script_name}: {{e}}")
        return image
'''
        return wrapper
        
    @staticmethod
    def _generate_default_processing(operations: List[str]) -> str:
        """Generate default processing code based on detected operations"""
        code_lines = []
        
        if 'grayscale' in operations:
            code_lines.append("if len(result.shape) == 3:")
            code_lines.append("    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)")
            
        if 'gaussian_blur' in operations:
            code_lines.append("result = cv2.GaussianBlur(result, (5, 5), 0)")
            
        if 'edge_detection' in operations:
            code_lines.append("result = cv2.Canny(result, 50, 150)")
            
        if 'threshold' in operations:
            code_lines.append("_, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)")
            
        return '\n        '.join(code_lines) if code_lines else "# Implement processing logic here"


class PipelineWorker(QThread):
    """Background thread for executing the image processing pipeline"""
    
    progress = pyqtSignal(int, str)  # (percentage, current_script_name)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    script_executing = pyqtSignal(str)  # Emits the exact filename being executed
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.pipeline = []
        self.functions = {}
        
    def set_data(self, image: np.ndarray, pipeline: List[Dict], functions: Dict[str, Callable]):
        self.image = image.copy()
        self.pipeline = pipeline
        self.functions = functions
        
    def run(self):
        func_name = "Unknown"
        try:
            result = self.image.copy()
            total_steps = len(self.pipeline)
            
            for i, step in enumerate(self.pipeline):
                func_name = step['name']
                params = step['params']
                
                # Emit the exact script filename being executed
                self.script_executing.emit(f"Executing: {func_name}")
                
                # Emit progress with script name
                status_msg = f"Step {i+1}/{total_steps}: Applying {func_name}"
                self.progress.emit(int((i / total_steps) * 100), status_msg)
                
                if func_name in self.functions:
                    func = self.functions[func_name]
                    result = func(result, **params)
                else:
                    raise RuntimeError(f"Function '{func_name}' not found.")
                    
            self.progress.emit(100, "Processing complete!")
            self.finished.emit(result)
            
        except Exception as e:
            error_msg = f"Error in '{func_name}': {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class FunctionLoader:
    """Enhanced function loader with Unicode error handling"""
    
    def __init__(self, directory: str = "scripts"):
        self.dir = Path(directory)
        self.functions: Dict[str, Callable] = {}
        self.function_info: Dict[str, Dict] = {}
        self.script_cleaner = ScriptCleaner()
        
    def scan(self):
        """Scan directory for functions with enhanced error handling"""
        self.functions.clear()
        self.function_info.clear()
        
        if not self.dir.exists():
            self.dir.mkdir(parents=True)
            
        # Create a cleaned scripts directory
        cleaned_dir = self.dir / "cleaned"
        cleaned_dir.mkdir(exist_ok=True)
        
        successful_loads = 0
        failed_loads = []
        
        for file_path in self.dir.glob("*.py"):
            if file_path.name.startswith("_") or file_path.stem == "cleaned":
                continue
                
            try:
                # First, try to load directly
                module = self._load_module_direct(file_path)
                
                if module and hasattr(module, 'process_image'):
                    self._register_function(file_path, module)
                    successful_loads += 1
                else:
                    # Try to clean and create wrapper
                    cleaned_module = self._clean_and_load(file_path, cleaned_dir)
                    if cleaned_module:
                        self._register_function(file_path, cleaned_module)
                        successful_loads += 1
                    else:
                        failed_loads.append(file_path.name)
                        
            except Exception as e:
                # Try to clean and create wrapper
                try:
                    cleaned_module = self._clean_and_load(file_path, cleaned_dir)
                    if cleaned_module:
                        self._register_function(file_path, cleaned_module)
                        successful_loads += 1
                    else:
                        failed_loads.append(file_path.name)
                except:
                    failed_loads.append(file_path.name)
                    print(f"Failed to load {file_path.name}: {e}")
                    
        print(f"Successfully loaded {successful_loads} functions")
        if failed_loads:
            print(f"Failed to load: {', '.join(failed_loads)}")
            
    def _load_module_direct(self, file_path: Path):
        """Try to load module directly"""
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except:
            return None
            
    def _clean_and_load(self, file_path: Path, cleaned_dir: Path):
        """Clean script and create wrapper"""
        try:
            # Read with error handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Clean the content
            cleaned_content = self.script_cleaner.clean_script_content(content)
            
            # Create wrapper if needed
            if 'def process_image' not in cleaned_content:
                cleaned_content = self.script_cleaner.create_process_image_wrapper(
                    cleaned_content, file_path.stem
                )
                
            # Save cleaned version
            cleaned_path = cleaned_dir / file_path.name
            with open(cleaned_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
                
            # Load the cleaned module
            return self._load_module_direct(cleaned_path)
            
        except Exception as e:
            print(f"Error cleaning {file_path.name}: {e}")
            return None
            
    def _register_function(self, file_path: Path, module):
        """Register a function from a module"""
        func = getattr(module, 'process_image')
        func_name = file_path.name  # Use full filename as identifier
        
        self.functions[func_name] = func
        self.function_info[func_name] = {
            'name': func_name,
            'doc': inspect.getdoc(func) or f"Process image using {file_path.stem}",
            'params': self._get_params(func),
            'category': self._categorize_function(file_path.name)
        }
        
    def _get_params(self, func: Callable) -> Dict[str, Dict]:
        """Extract parameter information"""
        params = {}
        sig = inspect.signature(func)
        
        for name, param in sig.parameters.items():
            if name == 'image':
                continue
            params[name] = {
                'type': param.annotation if param.annotation != param.empty else str,
                'default': param.default if param.default != param.empty else None
            }
        return params
        
    def _categorize_function(self, filename: str) -> str:
        """Categorize function based on filename"""
        filename_lower = filename.lower()
        
        categories = {
            'Filtering': ['blur', 'filter', 'median', 'gaussian'],
            'Edge Detection': ['edge', 'canny', 'sobel', 'laplacian', 'gradient'],
            'Thresholding': ['threshold', 'thresh', 'binary', 'otsu'],
            'Morphology': ['morph', 'erode', 'dilate', 'open', 'close'],
            'Enhancement': ['enhance', 'clahe', 'histogram', 'equalize', 'contrast'],
            'Color': ['color', 'gray', 'grayscale', 'hsv', 'rgb'],
            'Detection': ['detect', 'find', 'circle', 'contour', 'hough'],
            'Transform': ['transform', 'rotate', 'scale', 'resize', 'warp'],
            'Analysis': ['analyze', 'measure', 'profile', 'intensity'],
            'Visualization': ['visualize', 'viz', 'display', 'show', 'heatmap', 'colormap'],
            'Masking': ['mask', 'roi', 'region'],
            'I/O': ['load', 'save', 'read', 'write'],
        }
        
        for category, keywords in categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
                
        return "Other"


class MainWindow(QMainWindow):
    """Main application window with enhanced features"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.processed_image = None
        
        # --- NEW: Initialize User Preferences ---
        self.preferences = UserPreferences()
        
        self.function_loader = FunctionLoader()
        self.worker = PipelineWorker()
        
        # Connect worker signals
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.script_executing.connect(self.update_executing_script)
        
        self.init_ui()
        self.load_functions()
        
    def init_ui(self):
        self.setWindowTitle("Advanced Image Processing Pipeline Studio")
        self.setGeometry(50, 50, 1600, 900)
        
        # Apply modern stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 8px;
                border-radius: 4px;
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QPushButton#ProcessBtn {
                background-color: #4CAF50;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton#ProcessBtn:hover {
                background-color: #45a049;
            }
            QPushButton#ResetBtn {
                background-color: #FF5722;
            }
            QPushButton#ResetBtn:hover {
                background-color: #E64A19;
            }
            QPushButton#UndoBtn {
                background-color: #FF9800;
            }
            QPushButton#UndoBtn:hover {
                background-color: #F57C00;
            }
            QPushButton#ToggleBtn {
                background-color: #9C27B0;
            }
            QPushButton#ToggleBtn:hover {
                background-color: #7B1FA2;
            }
            QPushButton#ToggleBtn:checked {
                background-color: #E91E63;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
            QTableWidget::item:selected {
                background-color: #cce8ff;
                color: black;
            }
            QLabel#ExecutingScript {
                font-weight: bold;
                color: #D32F2F;
                padding: 8px;
                background-color: #FFEBEE;
                border: 1px solid #FFCDD2;
                border-radius: 4px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panels
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_center_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([400, 800, 400])
        
        # Status bar with progress
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Executing script label
        self.executing_label = QLabel("")
        self.executing_label.setObjectName("ExecutingScript")
        self.executing_label.setVisible(False)
        self.status_bar.addPermanentWidget(self.executing_label)
        
    def _create_left_panel(self):
        """Create function library panel with new filters and buttons."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        group = QGroupBox("Function Library")
        group_layout = QVBoxLayout(group)

        # --- MODIFIED: Search and Filter Layout ---
        filter_layout = QFormLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter keywords...")
        self.search_input.textChanged.connect(self.filter_functions)
        filter_layout.addRow(QLabel("Search:"), self.search_input)

        self.category_combo = QComboBox()
        self.category_combo.currentTextChanged.connect(self.filter_by_category)
        filter_layout.addRow(QLabel("Category:"), self.category_combo)

        # --- NEW: Special Views Filter ---
        self.special_view_combo = QComboBox()
        self.special_view_combo.addItems(["None", "Favorites", "Recently Used"])
        self.special_view_combo.currentTextChanged.connect(self.filter_functions)
        filter_layout.addRow(QLabel("View:"), self.special_view_combo)
        
        group_layout.addLayout(filter_layout)
        
        # --- MODIFIED: Function table to add favorite column and context menu ---
        self.function_table = QTableWidget()
        self.function_table.setColumnCount(3)
        self.function_table.setHorizontalHeaderLabels(["★", "Script File", "Category"])
        self.function_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.function_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.function_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.function_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.function_table.itemSelectionChanged.connect(self.show_function_details)
        self.function_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.function_table.customContextMenuRequested.connect(self.open_function_menu)
        group_layout.addWidget(self.function_table)
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        self.details_text.setReadOnly(True)
        group_layout.addWidget(self.details_text)
        
        # --- MODIFIED: Buttons layout with Hard Refresh ---
        button_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.load_functions)
        button_layout.addWidget(refresh_btn)

        hard_refresh_btn = QPushButton("Hard Refresh")
        hard_refresh_btn.setStyleSheet("background-color: #f44336;")
        hard_refresh_btn.setToolTip("Reloads all functions and resets the UI state, clearing potential cache issues.")
        hard_refresh_btn.clicked.connect(self.hard_reset_ui)
        button_layout.addWidget(hard_refresh_btn)
        group_layout.addLayout(button_layout)

        layout.addWidget(group)
        return panel

    def _create_center_panel(self):
        """Create image viewer panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        toolbar.addWidget(load_btn)
        
        save_btn = QPushButton("Save Result")
        save_btn.clicked.connect(self.save_image)
        toolbar.addWidget(save_btn)
        
        toolbar.addWidget(QLabel(" | "))
        
        # Reset button
        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.setObjectName("ResetBtn")
        self.reset_btn.clicked.connect(self.reset_to_original)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setToolTip("Reset to original image (Ctrl+R)")
        self.reset_btn.setShortcut("Ctrl+R")
        toolbar.addWidget(self.reset_btn)
        
        # Toggle original view button
        self.toggle_original_btn = QPushButton("View Original")
        self.toggle_original_btn.setObjectName("ToggleBtn")
        self.toggle_original_btn.setCheckable(True)
        self.toggle_original_btn.toggled.connect(self.toggle_original_view)
        self.toggle_original_btn.setEnabled(False)
        self.toggle_original_btn.setToolTip("Toggle original/processed view (Space)")
        self.toggle_original_btn.setShortcut("Space")
        toolbar.addWidget(self.toggle_original_btn)
        
        toolbar.addStretch()
        
        # Zoom controls
        toolbar.addWidget(QLabel("Zoom:"))
        
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setMaximumWidth(30)
        zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar.addWidget(zoom_out_btn)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(60)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        toolbar.addWidget(self.zoom_label)
        
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setMaximumWidth(30)
        zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar.addWidget(zoom_in_btn)
        
        zoom_fit_btn = QPushButton("Fit")
        zoom_fit_btn.clicked.connect(self.zoom_fit)
        toolbar.addWidget(zoom_fit_btn)
        
        zoom_reset_btn = QPushButton("100%")
        zoom_reset_btn.clicked.connect(self.zoom_reset)
        toolbar.addWidget(zoom_reset_btn)
        
        layout.addLayout(toolbar)
        
        # Image viewer
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer)
        
        # Image info
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setStyleSheet(
            "padding: 8px; background-color: #e8e8e8; border-radius: 4px;"
        )
        layout.addWidget(self.image_info_label)
        
        return panel
        
    def _create_right_panel(self):
        """Create pipeline panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Pipeline group
        group = QGroupBox("Processing Pipeline")
        group_layout = QVBoxLayout(group)
        
        # Pipeline list
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QListWidget.InternalMove)
        self.pipeline_list.itemDoubleClicked.connect(self.edit_pipeline_params)
        group_layout.addWidget(self.pipeline_list)
        
        # Pipeline controls
        controls = QHBoxLayout()
        
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_from_pipeline)
        controls.addWidget(remove_btn)
        
        self.undo_btn = QPushButton("Undo Last")
        self.undo_btn.setObjectName("UndoBtn")
        self.undo_btn.clicked.connect(self.undo_last_step)
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("Undo last processing step (Ctrl+Z)")
        self.undo_btn.setShortcut("Ctrl+Z")
        controls.addWidget(self.undo_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_pipeline)
        controls.addWidget(clear_btn)
        
        group_layout.addLayout(controls)
        
        # Save/Load pipeline
        io_layout = QHBoxLayout()
        
        save_pipe_btn = QPushButton("Save Pipeline")
        save_pipe_btn.clicked.connect(self.save_pipeline)
        io_layout.addWidget(save_pipe_btn)
        
        load_pipe_btn = QPushButton("Load Pipeline")
        load_pipe_btn.clicked.connect(self.load_pipeline)
        io_layout.addWidget(load_pipe_btn)
        
        group_layout.addLayout(io_layout)
        
        # Process button
        self.process_btn = QPushButton("PROCESS IMAGE")
        self.process_btn.setObjectName("ProcessBtn")
        self.process_btn.clicked.connect(self.process_image)
        group_layout.addWidget(self.process_btn)
        
        layout.addWidget(group)
        
        # Currently executing script display
        exec_group = QGroupBox("Execution Status")
        exec_layout = QVBoxLayout(exec_group)
        
        self.current_script_label = QLabel("Ready")
        self.current_script_label.setWordWrap(True)
        self.current_script_label.setStyleSheet(
            "padding: 10px; background-color: #f0f0f0; border-radius: 4px;"
        )
        exec_layout.addWidget(self.current_script_label)
        
        layout.addWidget(exec_group)
        
        return panel
    
    # --- NEW: Function to open a context menu in the library ---
    def open_function_menu(self, position):
        """Create and show a context menu for function table items."""
        selected_items = self.function_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        func_name_item = self.function_table.item(row, 1)
        if not func_name_item: return
        
        func_name = func_name_item.data(Qt.UserRole)

        menu = QMenu()
        add_action = menu.addAction("Add to Pipeline →")
        open_action = menu.addAction("Open in VS Code")
        menu.addSeparator()
        
        is_fav = self.preferences.is_favorite(func_name)
        fav_text = "Remove from Favorites ★" if is_fav else "Add to Favorites ☆"
        favorite_action = menu.addAction(fav_text)

        action = menu.exec_(self.function_table.mapToGlobal(position))

        if action == add_action:
            self.add_to_pipeline(func_name)
        elif action == open_action:
            self.open_script_in_vscode(func_name)
        elif action == favorite_action:
            self.toggle_favorite(func_name)

    # --- NEW: Function to add/remove a function from favorites ---
    def toggle_favorite(self, func_name):
        """Toggles the favorite status of a function."""
        if self.preferences.is_favorite(func_name):
            self.preferences.remove_favorite(func_name)
            self.status_bar.showMessage(f"'{func_name}' removed from favorites.", 3000)
        else:
            self.preferences.add_favorite(func_name)
            self.status_bar.showMessage(f"'{func_name}' added to favorites.", 3000)
        
        # Refresh the table to show the new star status
        self.filter_functions()

    # --- NEW: Function to open a script file in VS Code ---
    def open_script_in_vscode(self, func_name):
        """Opens the specified script file in Visual Studio Code."""
        file_path = self.function_loader.dir / func_name
        if not file_path.exists():
            file_path = self.function_loader.dir / "cleaned" / func_name

        if file_path.exists():
            try:
                subprocess.Popen(['code', str(file_path)])
                self.status_bar.showMessage(f"Opening {func_name} in VS Code...", 3000)
            except FileNotFoundError:
                QMessageBox.warning(self, "Error", "Could not find 'code' command. Is VS Code installed and in your system's PATH?")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open file: {e}")
        else:
            QMessageBox.warning(self, "File Not Found", f"Could not find the script file: {file_path}")

    # --- NEW: Function for the Hard Refresh button ---
    def hard_reset_ui(self):
        """Resets the entire UI and reloads everything from scratch."""
        reply = QMessageBox.question(self, 'Hard Refresh',
                                     "This will clear the current pipeline, reset the image, and reload all functions from disk. This can fix UI state issues.\n\nAre you sure you want to continue?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.status_bar.showMessage("Performing hard refresh...", 5000)
            
            self.current_image = None
            self.processed_image = None
            self.image_viewer.set_image(None)
            self.image_info_label.setText("No image loaded")
            self.reset_btn.setEnabled(False)
            self.toggle_original_btn.setEnabled(False)
            
            self.clear_pipeline()
            
            self.current_script_label.setText("Ready")
            self.progress_bar.setVisible(False)
            self.executing_label.setVisible(False)

            self.function_loader = FunctionLoader()
            self.load_functions()

            self.status_bar.showMessage("Hard refresh complete. Please load an image.", 3000)
        
    def load_functions(self):
        """Load all available functions"""
        self.function_loader.scan()
        self.populate_function_table()
        self.update_categories()
        self.status_bar.showMessage(
            f"Loaded {len(self.function_loader.functions)} functions", 3000
        )
        
    def populate_function_table(self, filter_text="", category="All"):
        """Populate the function table with filtering and sorting."""
        self.function_table.setRowCount(0)
        
        # Determine the source list of functions based on special view
        special_view = self.special_view_combo.currentText()
        all_funcs = sorted(self.function_loader.function_info.items())
        
        func_list = []
        if special_view == "Favorites":
            favs = self.preferences.data['favorites']
            func_list = [item for item in all_funcs if item[0] in favs]
        elif special_view == "Recently Used":
            recent = self.preferences.get_recently_used()
            func_dict = dict(all_funcs)
            func_list = [(name, func_dict[name]) for name in recent if name in func_dict]
        else: # "None"
            func_list = all_funcs

        for func_name, func_info in func_list:
            if filter_text and filter_text.lower() not in func_name.lower():
                continue
            if category != "All" and func_info['category'] != category:
                continue

            row = self.function_table.rowCount()
            self.function_table.insertRow(row)
            
            # Favorite Star Column
            is_fav = self.preferences.is_favorite(func_name)
            fav_char = "★" if is_fav else "☆"
            fav_item = QTableWidgetItem(fav_char)
            fav_item.setTextAlignment(Qt.AlignCenter)
            fav_item.setData(Qt.UserRole, func_name) # Store name for click context
            self.function_table.setItem(row, 0, fav_item)

            # Script Name
            name_item = QTableWidgetItem(func_name)
            name_item.setData(Qt.UserRole, func_name)
            self.function_table.setItem(row, 1, name_item)

            # Category
            category_item = QTableWidgetItem(func_info['category'])
            self.function_table.setItem(row, 2, category_item)

    def update_categories(self):
        """Update category combo box"""
        categories = set(info['category'] for info in self.function_loader.function_info.values())
            
        self.category_combo.clear()
        self.category_combo.addItem("All")
        self.category_combo.addItems(sorted(list(categories)))
            
    def filter_functions(self, text=None):
        """Filter functions by search text and special view."""
        self.populate_function_table(
            filter_text=self.search_input.text(),
            category=self.category_combo.currentText()
        )
        
    def filter_by_category(self, category):
        """Filter functions by category"""
        # When category changes, reset special view to None for clarity
        self.special_view_combo.setCurrentIndex(0)
        self.populate_function_table(
            filter_text=self.search_input.text(),
            category=category
        )
        
    def show_function_details(self):
        """Show details of selected function"""
        current_row = self.function_table.currentRow()
        if current_row < 0:
            self.details_text.clear()
            return
            
        name_item = self.function_table.item(current_row, 1)
        if name_item:
            func_name = name_item.data(Qt.UserRole)
            func_info = self.function_loader.function_info.get(func_name, {})
            
            details = f"<b>File:</b> {func_name}<br>"
            details += f"<b>Category:</b> {func_info.get('category', 'Unknown')}<br>"
            details += f"<b>Description:</b> {func_info.get('doc', 'No description')}<br>"
            
            if func_info.get('params'):
                details += "<br><b>Parameters:</b><br>"
                for param_name, param_info in func_info['params'].items():
                    param_type = param_info['type'].__name__ if hasattr(param_info['type'], '__name__') else str(param_info['type'])
                    details += f"• {param_name} ({param_type})"
                    if param_info['default'] is not None:
                        details += f" = {param_info['default']}"
                    details += "<br>"
                    
            self.details_text.setHtml(details)
            
    def add_to_pipeline(self, func_name):
        """Add selected function to pipeline and track usage."""
        func_info = self.function_loader.function_info[func_name]
        
        # Create pipeline step
        pipeline_step = {
            'name': func_name,
            'params': {
                name: p_info['default'] 
                for name, p_info in func_info['params'].items()
                if p_info['default'] is not None
            }
        }
        
        # Add to list
        item_text = self._format_pipeline_item(pipeline_step)
        list_item = QListWidgetItem(item_text)
        list_item.setData(Qt.UserRole, pipeline_step)
        self.pipeline_list.addItem(list_item)
        
        # --- NEW: Track recently used ---
        self.preferences.add_recently_used(func_name)
        if self.special_view_combo.currentText() == "Recently Used":
            self.filter_functions() # Refresh list if view is active
        
    def _format_pipeline_item(self, step):
        """Format pipeline item text"""
        params_str = ", ".join(f"{k}={v}" for k, v in step['params'].items())
        return f"{step['name']}" + (f" ({params_str})" if params_str else "")
        
    def edit_pipeline_params(self, item):
        """Edit parameters of pipeline item"""
        pipeline_step = item.data(Qt.UserRole)
        func_name = pipeline_step['name']
        func_info = self.function_loader.function_info[func_name]
        
        if not func_info['params']:
            QMessageBox.information(
                self, "No Parameters",
                f"'{func_name}' has no editable parameters."
            )
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Parameters - {func_name}")
        dialog.setMinimumWidth(400)
        form_layout = QFormLayout(dialog)
        widgets = {}
        
        for name, p_info in func_info['params'].items():
            current_val = pipeline_step['params'].get(name, p_info['default'])
            widget = None
            
            if p_info['type'] is bool:
                widget = QCheckBox()
                if current_val: widget.setChecked(True)
            elif p_info['type'] is int:
                widget = QSpinBox(); widget.setRange(-10000, 10000)
                if current_val is not None: widget.setValue(current_val)
            elif p_info['type'] is float:
                widget = QDoubleSpinBox(); widget.setRange(-10000.0, 10000.0); widget.setDecimals(4)
                if current_val is not None: widget.setValue(current_val)
            else:
                widget = QLineEdit()
                if current_val is not None: widget.setText(str(current_val))
                    
            if widget:
                param_label = QLabel(f"{name} ({p_info['type'].__name__}):")
                form_layout.addRow(param_label, widget)
                widgets[name] = widget
                
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        form_layout.addRow(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            for name, widget in widgets.items():
                if isinstance(widget, QCheckBox):
                    pipeline_step['params'][name] = widget.isChecked()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    pipeline_step['params'][name] = widget.value()
                else:
                    text = widget.text()
                    try: value = eval(text)
                    except: value = text
                    pipeline_step['params'][name] = value
                    
            item.setText(self._format_pipeline_item(pipeline_step))
            item.setData(Qt.UserRole, pipeline_step)
            # Invalidate processed image since parameters changed
            self.processed_image = None
            
    def remove_from_pipeline(self):
        """Remove selected item from pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            self.pipeline_list.takeItem(current_row)
            # --- MODIFIED: Invalidate state when pipeline changes ---
            self.processed_image = None
            self.undo_btn.setEnabled(False)
            self.current_script_label.setText("Pipeline modified. Re-process required.")
            self.status_bar.showMessage("Pipeline modified. Please re-process.", 3000)
            
    def clear_pipeline(self):
        """Clear entire pipeline"""
        if self.pipeline_list.count() > 0:
            self.pipeline_list.clear()
            # --- MODIFIED: Invalidate state when pipeline is cleared ---
            self.processed_image = None
            self.undo_btn.setEnabled(False)
            self.current_script_label.setText("Pipeline cleared.")
            self.status_bar.showMessage("Pipeline cleared.", 3000)

    def load_image(self):
        """Load an image file"""
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)")
        
        if path:
            self.current_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if self.current_image is None:
                QMessageBox.critical(self, "Error", f"Failed to load image from {path}")
                return
            
            # Reset everything for the new image
            self.reset_to_original()
            
            self.image_viewer.set_image(self.current_image)
            self.reset_btn.setEnabled(True)
            self.toggle_original_btn.setEnabled(True)
            
            h, w = self.current_image.shape[:2]
            c = self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
            self.image_info_label.setText(
                f"Loaded: {os.path.basename(path)} | Size: {w}×{h} | Channels: {c} | Type: {self.current_image.dtype}"
            )
            self.status_bar.showMessage("Image loaded successfully", 3000)
            
    def save_image(self):
        """Save processed image"""
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to save")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)")
        
        if path:
            try:
                cv2.imwrite(path, self.processed_image)
                self.status_bar.showMessage(f"Image saved to {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
                
    def process_image(self):
        """Process image through pipeline"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
            
        if self.pipeline_list.count() == 0:
            QMessageBox.warning(self, "Warning", "Pipeline is empty")
            return
            
        pipeline_data = [self.pipeline_list.item(i).data(Qt.UserRole) for i in range(self.pipeline_list.count())]
            
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.executing_label.setVisible(True)
        self.current_script_label.setText("Starting pipeline...")
        
        self.worker.set_data(
            self.current_image,
            pipeline_data,
            self.function_loader.functions
        )
        self.worker.start()
        
    def update_progress(self, percentage, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(percentage)
        self.status_bar.showMessage(message)
        self.current_script_label.setText(message)
        
    def update_executing_script(self, script_name):
        """Update the currently executing script display"""
        self.executing_label.setText(f"⚡ {script_name}")
        
    def on_processing_finished(self, result_image):
        """Handle processing completion"""
        self.processed_image = result_image
        self.image_viewer.set_image(result_image)
        
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.executing_label.setVisible(False)
        
        self.undo_btn.setEnabled(True)
        
        if self.toggle_original_btn.isChecked():
            self.toggle_original_btn.setChecked(False)
        
        self.current_script_label.setText("✓ Processing complete!")
        self.status_bar.showMessage("Processing finished successfully!", 5000)
        
        h, w = result_image.shape[:2]
        c = result_image.shape[2] if len(result_image.shape) > 2 else 1
        self.image_info_label.setText(f"Processed | Size: {w}×{h} | Channels: {c} | Type: {result_image.dtype}")
        
    def on_processing_error(self, error_message):
        """Handle processing error"""
        QMessageBox.critical(self, "Processing Error", error_message)
        
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.executing_label.setVisible(False)
        
        self.current_script_label.setText("✗ Processing failed!")
        self.status_bar.showMessage("An error occurred during processing", 5000)
        
    def zoom_in(self):
        """Zoom in the image"""
        if self.image_viewer.scale_factor < 10:
            self.image_viewer.scale_factor *= 1.25
            self.image_viewer.update_display()
            self.update_zoom_label()
            
    def zoom_out(self):
        """Zoom out the image"""
        if self.image_viewer.scale_factor > 0.1:
            self.image_viewer.scale_factor /= 1.25
            self.image_viewer.update_display()
            self.update_zoom_label()
            
    def zoom_fit(self):
        """Fit image to viewport"""
        self.image_viewer.zoom_to_fit()
        self.update_zoom_label()
        
    def zoom_reset(self):
        """Reset zoom to 100%"""
        self.image_viewer.reset_zoom()
        self.update_zoom_label()
        
    def update_zoom_label(self):
        """Update zoom percentage display"""
        zoom_percent = int(self.image_viewer.scale_factor * 100)
        self.zoom_label.setText(f"{zoom_percent}%")
        
    def save_pipeline(self):
        """Save pipeline configuration"""
        if self.pipeline_list.count() == 0:
            QMessageBox.warning(self, "Warning", "Pipeline is empty")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline", "", "JSON Files (*.json);;All Files (*.*)")
        
        if path:
            pipeline_data = [self.pipeline_list.item(i).data(Qt.UserRole) for i in range(self.pipeline_list.count())]
            try:
                with open(path, 'w') as f:
                    json.dump(pipeline_data, f, indent=2)
                self.status_bar.showMessage(f"Pipeline saved to {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save pipeline: {e}")
                
    def load_pipeline(self):
        """Load pipeline configuration"""
        path, _ = QFileDialog.getOpenFileName(self, "Load Pipeline", "", "JSON Files (*.json);;All Files (*.*)")
        
        if path:
            try:
                with open(path, 'r') as f:
                    pipeline_data = json.load(f)
                    
                self.pipeline_list.clear()
                
                for step in pipeline_data:
                    # Validate that the function exists before adding
                    if step.get('name') in self.function_loader.functions:
                        item_text = self._format_pipeline_item(step)
                        list_item = QListWidgetItem(item_text)
                        list_item.setData(Qt.UserRole, step)
                        self.pipeline_list.addItem(list_item)
                    else:
                        print(f"Warning: Function '{step.get('name')}' from pipeline file not found. Skipping.")

                self.status_bar.showMessage(f"Pipeline loaded from {path}", 3000)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load pipeline: {e}")


    def reset_to_original(self):
        """Reset the displayed image to the original loaded image"""
        if self.current_image is not None:
            self.processed_image = None
            self.image_viewer.set_image(self.current_image)
            
            self.undo_btn.setEnabled(False)
            self.toggle_original_btn.setChecked(False)
            
            h, w = self.current_image.shape[:2]
            c = self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
            self.image_info_label.setText(f"Reset to Original | Size: {w}×{h} | Channels: {c} | Type: {self.current_image.dtype}")
            self.current_script_label.setText("Reset to original image")
            self.status_bar.showMessage("Reset to original image", 3000)
    
    def undo_last_step(self):
        """Remove the last step from pipeline and reprocess"""
        if self.pipeline_list.count() > 0:
            self.pipeline_list.takeItem(self.pipeline_list.count() - 1)
            
            if self.pipeline_list.count() == 0:
                self.reset_to_original()
            else:
                self.status_bar.showMessage("Undoing last step and reprocessing...", 1000)
                self.process_image()
        self.undo_btn.setEnabled(False) # Undo can only be done once per process run
    
    def toggle_original_view(self, checked):
        """Toggle between original and processed image view"""
        if self.current_image is None: return
            
        if checked:
            self.image_viewer.set_image(self.current_image)
            self.toggle_original_btn.setText("View Processed")
            if "(Viewing Original)" not in self.image_info_label.text():
                self.image_info_label.setText(self.image_info_label.text() + " (Viewing Original)")
        else:
            self.image_viewer.set_image(self.processed_image if self.processed_image is not None else self.current_image)
            self.toggle_original_btn.setText("View Original")
            self.image_info_label.setText(self.image_info_label.text().replace(" (Viewing Original)", ""))

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    app.setApplicationName("Image Processing Pipeline Studio")
    app.setOrganizationName("OpenCV Practice")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()