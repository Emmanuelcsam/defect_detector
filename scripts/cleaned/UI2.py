#! python3
"""
Modular Image Processing GUI
============================
A dynamic image processing application with pipeline building capabilities.

Author: Assistant
Date: 2025
"""

import sys
import os
import importlib.util
import inspect
import traceback
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QSplitter, QLineEdit, QMessageBox, QScrollArea, QGroupBox,
    QSlider, QSpinBox, QCheckBox, QComboBox, QTextEdit, QProgressBar,
    QStatusBar, QToolBar, QAction, QMenu, QMenuBar, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QMimeData, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent, QDrag, QFont, QIcon


class ImageViewer(QLabel):
    """Custom QLabel for image display with zoom and pan capabilities."""
    
    def __init__(self):
        super().__init__()
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.pan_start = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.setMinimumSize(400, 400)
        self.setScaledContents(False)
        
    def set_image(self, image: np.ndarray):
        """Set the image to display."""
        if image is None:
            self.clear()
            return
            
        # Convert numpy array to QPixmap
        if len(image.shape) == 2:  # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color (BGR to RGB conversion)
            height, width, channels = image.shape
            bytes_per_line = channels * width
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.update_display()
    
    def update_display(self):
        """Update the displayed image with current scale."""
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if self.original_pixmap:
            # Get the zoom factor
            zoom_in_factor = 1.25
            zoom_out_factor = 0.8
            
            if event.angleDelta().y() > 0:
                self.scale_factor *= zoom_in_factor
            else:
                self.scale_factor *= zoom_out_factor
            
            # Limit zoom range
            self.scale_factor = max(0.1, min(self.scale_factor, 10.0))
            self.update_display()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Start panning on middle mouse button."""
        if event.button() == Qt.MiddleButton:
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop panning."""
        if event.button() == Qt.MiddleButton:
            self.pan_start = None
            self.setCursor(Qt.ArrowCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle panning."""
        if self.pan_start:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            
            # Get parent scroll area and adjust scrollbars
            parent = self.parent()
            while parent and not isinstance(parent, QScrollArea):
                parent = parent.parent()
            
            if parent:
                h_bar = parent.horizontalScrollBar()
                v_bar = parent.verticalScrollBar()
                h_bar.setValue(h_bar.value() - delta.x())
                v_bar.setValue(v_bar.value() - delta.y())


class FunctionLoader:
    """Dynamically loads image processing functions from a directory."""
    
    def __init__(self, functions_dir: str = "functions"):
        self.functions_dir = Path(functions_dir)
        self.functions: Dict[str, Callable] = {}
        self.function_info: Dict[str, Dict[str, Any]] = {}
        
    def scan_functions(self) -> Dict[str, Callable]:
        """Scan the functions directory and load all valid processing functions."""
        self.functions.clear()
        self.function_info.clear()
        
        if not self.functions_dir.exists():
            self.functions_dir.mkdir(exist_ok=True)
            # Create a sample function
            self._create_sample_function()
        
        for file_path in self.functions_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
                
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for a process_image function
                if hasattr(module, 'process_image'):
                    func = getattr(module, 'process_image')
                    if callable(func):
                        self.functions[file_path.name] = func
                        
                        # Extract function info
                        self.function_info[file_path.name] = {
                            'name': file_path.stem,
                            'doc': inspect.getdoc(func) or "No description available",
                            'params': self._get_function_params(func)
                        }
                        
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
        return self.functions
    
    def _get_function_params(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter information from function signature."""
        params = {}
        sig = inspect.signature(func)
        
        for name, param in sig.parameters.items():
            if name == 'image':  # Skip the image parameter
                continue
                
            param_info = {
                'type': param.annotation if param.annotation != param.empty else None,
                'default': param.default if param.default != param.empty else None
            }
            params[name] = param_info
            
        return params
    
    def _create_sample_function(self):
        """Create sample functions to get started."""
        # Gaussian Blur
        blur_code = '''"""Apply Gaussian blur to the image."""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to smooth the image.
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
'''
        
        # Edge Detection
        edge_code = '''"""Detect edges using Canny edge detection."""
import cv2
import numpy as np

def process_image(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges in the image using Canny edge detection.
    
    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
    
    Returns:
        Edge map
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, low_threshold, high_threshold)
'''
        
        # Threshold
        threshold_code = '''"""Apply binary threshold to the image."""
import cv2
import numpy as np

def process_image(image: np.ndarray, threshold: int = 127, max_value: int = 255) -> np.ndarray:
    """
    Apply binary threshold to the image.
    
    Args:
        image: Input image
        threshold: Threshold value
        max_value: Maximum value to use with THRESH_BINARY
    
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
    return binary
'''
        
        # Save sample functions
        with open(self.functions_dir / "gaussian_blur.py", "w") as f:
            f.write(blur_code)
        with open(self.functions_dir / "edge_detection.py", "w") as f:
            f.write(edge_code)
        with open(self.functions_dir / "threshold.py", "w") as f:
            f.write(threshold_code)


class PipelineWorker(QThread):
    """Background thread for executing the image processing pipeline."""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.pipeline = []
        self.functions = {}
        
    def set_data(self, image: np.ndarray, pipeline: List[Tuple[str, Dict]], functions: Dict):
        """Set the data for processing."""
        self.image = image.copy()
        self.pipeline = pipeline
        self.functions = functions
        
    def run(self):
        """Execute the pipeline."""
        try:
            result = self.image.copy()
            total_steps = len(self.pipeline)
            
            for i, (func_name, params) in enumerate(self.pipeline):
                self.status.emit(f"Applying {func_name}...")
                
                if func_name in self.functions:
                    func = self.functions[func_name]
                    result = func(result, **params)
                    
                self.progress.emit(int((i + 1) / total_steps * 100))
                
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(f"Pipeline error: {str(e)}\n{traceback.format_exc()}")


class PipelineItem(QListWidgetItem):
    """Custom list item for pipeline functions."""
    
    def __init__(self, func_name: str, params: Dict[str, Any]):
        super().__init__()
        self.func_name = func_name
        self.params = params
        self.update_text()
        
    def update_text(self):
        """Update the display text with parameters."""
        param_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        if param_str:
            self.setText(f"{self.func_name} ({param_str})")
        else:
            self.setText(self.func_name)


class ImageProcessorGUI(QMainWindow):
    """Main GUI application for modular image processing."""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.processed_image = None
        self.function_loader = FunctionLoader()
        self.pipeline_worker = PipelineWorker()
        
        self.init_ui()
        self.load_functions()
        
        # Connect worker signals
        self.pipeline_worker.progress.connect(self.update_progress)
        self.pipeline_worker.status.connect(self.update_status)
        self.pipeline_worker.finished.connect(self.on_processing_finished)
        self.pipeline_worker.error.connect(self.on_processing_error)
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Modular Image Processing GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Function Library
        left_panel = self.create_function_panel()
        splitter.addWidget(left_panel)
        
        # Middle panel - Image Display
        middle_panel = self.create_image_panel()
        splitter.addWidget(middle_panel)
        
        # Right panel - Pipeline Builder
        right_panel = self.create_pipeline_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 600, 300])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                outline: none;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        
    def create_function_panel(self) -> QWidget:
        """Create the function library panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Function Library")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)
        
        # Search bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search functions...")
        self.search_input.textChanged.connect(self.filter_functions)
        layout.addWidget(self.search_input)
        
        # Function list
        self.function_list = QListWidget()
        self.function_list.setDragEnabled(True)
        self.function_list.itemDoubleClicked.connect(self.add_function_to_pipeline)
        layout.addWidget(self.function_list)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Functions")
        refresh_btn.clicked.connect(self.load_functions)
        layout.addWidget(refresh_btn)
        
        # Function info
        info_group = QGroupBox("Function Info")
        info_layout = QVBoxLayout(info_group)
        self.function_info_text = QTextEdit()
        self.function_info_text.setReadOnly(True)
        self.function_info_text.setMaximumHeight(150)
        info_layout.addWidget(self.function_info_text)
        layout.addWidget(info_group)
        
        # Connect selection change
        self.function_list.currentItemChanged.connect(self.show_function_info)
        
        return panel
        
    def create_image_panel(self) -> QWidget:
        """Create the image display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Image Viewer")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        toolbar_layout.addWidget(load_btn)
        
        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(self.save_image)
        toolbar_layout.addWidget(save_btn)
        
        toolbar_layout.addStretch()
        
        # Zoom controls
        zoom_label = QLabel("Zoom:")
        toolbar_layout.addWidget(zoom_label)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        toolbar_layout.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        toolbar_layout.addWidget(self.zoom_label)
        
        reset_zoom_btn = QPushButton("Reset")
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        toolbar_layout.addWidget(reset_zoom_btn)
        
        layout.addLayout(toolbar_layout)
        
        # Image viewer with scroll area
        scroll_area = QScrollArea()
        self.image_viewer = ImageViewer()
        scroll_area.setWidget(self.image_viewer)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Image info
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        layout.addWidget(self.image_info_label)
        
        return panel
        
    def create_pipeline_panel(self) -> QWidget:
        """Create the pipeline builder panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Processing Pipeline")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)
        
        # Pipeline list
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QListWidget.InternalMove)
        self.pipeline_list.itemDoubleClicked.connect(self.edit_pipeline_item)
        layout.addWidget(self.pipeline_list)
        
        # Pipeline controls
        controls_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_pipeline)
        controls_layout.addWidget(clear_btn)
        
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_pipeline_item)
        controls_layout.addWidget(remove_btn)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Process button
        self.process_btn = QPushButton("Process Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(self.process_btn)
        
        # Current status
        status_group = QGroupBox("Processing Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_group)
        
        layout.addStretch()
        
        return panel
        
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Image", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        
        save_action = QAction("Save Image", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Pipeline menu
        pipeline_menu = menubar.addMenu("Pipeline")
        
        save_pipeline_action = QAction("Save Pipeline", self)
        save_pipeline_action.triggered.connect(self.save_pipeline)
        pipeline_menu.addAction(save_pipeline_action)
        
        load_pipeline_action = QAction("Load Pipeline", self)
        load_pipeline_action.triggered.connect(self.load_pipeline)
        pipeline_menu.addAction(load_pipeline_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def load_functions(self):
        """Load all available processing functions."""
        self.function_list.clear()
        functions = self.function_loader.scan_functions()
        
        for func_name in sorted(functions.keys()):
            self.function_list.addItem(func_name)
            
        self.update_status(f"Loaded {len(functions)} functions")
        
    def filter_functions(self, text: str):
        """Filter the function list based on search text."""
        for i in range(self.function_list.count()):
            item = self.function_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())
            
    def show_function_info(self, current, previous):
        """Display information about the selected function."""
        if current:
            func_name = current.text()
            if func_name in self.function_loader.function_info:
                info = self.function_loader.function_info[func_name]
                
                info_text = f"<b>{info['name']}</b><br><br>"
                info_text += f"{info['doc']}<br><br>"
                
                if info['params']:
                    info_text += "<b>Parameters:</b><br>"
                    for param_name, param_info in info['params'].items():
                        param_type = param_info['type'].__name__ if param_info['type'] else "Any"
                        default = param_info['default']
                        info_text += f" {param_name} ({param_type})"
                        if default is not None:
                            info_text += f" = {default}"
                        info_text += "<br>"
                
                self.function_info_text.setHtml(info_text)
            
    def add_function_to_pipeline(self, item):
        """Add a function to the processing pipeline."""
        if item:
            func_name = item.text()
            
            # Get default parameters
            params = {}
            if func_name in self.function_loader.function_info:
                func_info = self.function_loader.function_info[func_name]
                for param_name, param_info in func_info['params'].items():
                    if param_info['default'] is not None:
                        params[param_name] = param_info['default']
            
            # Add to pipeline
            pipeline_item = PipelineItem(func_name, params)
            self.pipeline_list.addItem(pipeline_item)
            
    def edit_pipeline_item(self, item):
        """Edit parameters of a pipeline item."""
        if isinstance(item, PipelineItem):
            # Create parameter dialog
            from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Edit Parameters - {item.func_name}")
            layout = QFormLayout(dialog)
            
            # Create input widgets for each parameter
            widgets = {}
            if item.func_name in self.function_loader.function_info:
                func_info = self.function_loader.function_info[item.func_name]
                
                for param_name, param_info in func_info['params'].items():
                    param_type = param_info['type']
                    current_value = item.params.get(param_name, param_info['default'])
                    
                    if param_type == int or param_type == float:
                        widget = QSpinBox() if param_type == int else QDoubleSpinBox()
                        widget.setRange(-10000, 10000)
                        widget.setValue(current_value if current_value is not None else 0)
                    else:
                        widget = QLineEdit()
                        widget.setText(str(current_value) if current_value is not None else "")
                    
                    widgets[param_name] = widget
                    layout.addRow(param_name, widget)
            
            # Add buttons
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addRow(buttons)
            
            if dialog.exec_() == QDialog.Accepted:
                # Update parameters
                for param_name, widget in widgets.items():
                    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                        item.params[param_name] = widget.value()
                    else:
                        value = widget.text()
                        # Try to convert to appropriate type
                        try:
                            value = eval(value)
                        except:
                            pass
                        item.params[param_name] = value
                
                item.update_text()
                
    def remove_pipeline_item(self):
        """Remove selected item from pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            self.pipeline_list.takeItem(current_row)
            
    def clear_pipeline(self):
        """Clear all items from pipeline."""
        self.pipeline_list.clear()
        
    def load_image(self):
        """Load an image from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if self.current_image is None:
                    raise ValueError("Failed to load image")
                
                self.processed_image = self.current_image.copy()
                self.image_viewer.set_image(self.current_image)
                
                # Update image info
                shape = self.current_image.shape
                if len(shape) == 2:
                    info = f"Grayscale: {shape[1]}x{shape[0]}"
                else:
                    info = f"Color: {shape[1]}x{shape[0]}, {shape[2]} channels"
                self.image_info_label.setText(info)
                
                self.update_status(f"Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load imag")
                
    def save_image(self):
        """Save the processed image."""
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to save")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                self.update_status(f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save imag")
                
    def process_image(self):
        """Process the image through the pipeline."""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
            
        if self.pipeline_list.count() == 0:
            QMessageBox.warning(self, "Warning", "Pipeline is empty")
            return
            
        # Build pipeline
        pipeline = []
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            if isinstance(item, PipelineItem):
                pipeline.append((item.func_name, item.params))
        
        # Start processing
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.pipeline_worker.set_data(
            self.current_image,
            pipeline,
            self.function_loader.functions
        )
        self.pipeline_worker.start()
        
    def on_processing_finished(self, result: np.ndarray):
        """Handle processing completion."""
        self.processed_image = result
        self.image_viewer.set_image(result)
        self.process_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.update_status("Processing complete")
        
    def on_processing_error(self, error_msg: str):
        """Handle processing error."""
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.process_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.update_status("Processing failed")
        
    def update_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        
    def update_status(self, message: str):
        """Update status bar and status label."""
        self.status_bar.showMessage(message)
        self.status_label.setText(message)
        
    def on_zoom_changed(self, value: int):
        """Handle zoom slider change."""
        scale = value / 100.0
        self.image_viewer.scale_factor = scale
        self.image_viewer.update_display()
        self.zoom_label.setText(f"{value}%")
        
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.zoom_slider.setValue(100)
        
    def save_pipeline(self):
        """Save the current pipeline to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "", "Pipeline Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            import json
            pipeline = []
            for i in range(self.pipeline_list.count()):
                item = self.pipeline_list.item(i)
                if isinstance(item, PipelineItem):
                    pipeline.append({
                        'function': item.func_name,
                        'params': item.params
                    })
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(pipeline, f, indent=2)
                self.update_status(f"Pipeline saved: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save pipelin")
                
    def load_pipeline(self):
        """Load a pipeline from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", "", "Pipeline Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            import json
            try:
                with open(file_path, 'r') as f:
                    pipeline = json.load(f)
                
                self.pipeline_list.clear()
                for item_data in pipeline:
                    pipeline_item = PipelineItem(
                        item_data['function'],
                        item_data['params']
                    )
                    self.pipeline_list.addItem(pipeline_item)
                
                self.update_status(f"Pipeline loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load pipelin")
                
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Modular Image Processing GUI",
            "<h3>Modular Image Processing GUI</h3>"
            "<p>A dynamic image processing application with pipeline building capabilities.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Dynamic function loading from Python scripts</li>"
            "<li>Visual pipeline builder with drag & drop</li>"
            "<li>Real-time image preview with zoom</li>"
            "<li>Parameter editing for each function</li>"
            "<li>Pipeline save/load functionality</li>"
            "</ul>"
            "<p>Created with PyQt5 and OpenCV</p>"
        )


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application icon (optional)
    app.setWindowIcon(QIcon())
    
    window = ImageProcessorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
