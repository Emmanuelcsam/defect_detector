#! python3
"""
Advanced Image Processing Pipeline GUI
======================================
A robust, modular UI for creating, managing, and executing image processing pipelines.
This version combines the best features of previous attempts, focusing on stability,
responsiveness, and ease of use.

Key Features:
- Asynchronous processing using QThread to prevent UI freezing.
- Dynamic function loading from a dedicated 'functions' directory.
- A strict, easy-to-follow template for creating compatible function scripts.
- Visual pipeline builder with drag-and-drop reordering.
- Automatic UI generation for editing function parameters based on type hints.
- Image viewer with smooth zoom (centered on the cursor) and pan.
- Save/Load functionality for both processed images and pipeline configurations.

Author: Gemini Assistant
Date: 2025-06-12
"""

import sys
import os
import json
import importlib.util
import inspect
import traceback
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple
import numpy as np
import cv2

# --- PyQt5 Imports ---
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QSplitter, QLineEdit, QMessageBox, QScrollArea, QGroupBox,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QDialog, QFormLayout,
    QDialogButtonBox, QTextEdit, QProgressBar, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor

# --- Custom ImageViewer with Zoom and Pan ---
class ImageViewer(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setWidget(self.image_label)
        self.original_pixmap = None
        self.scale_factor = 1.0

    def set_image(self, image: np.ndarray):
        if image is None:
            self.image_label.clear()
            self.original_pixmap = None
            return

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
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                int(self.original_pixmap.width() * self.scale_factor),
                int(self.original_pixmap.height() * self.scale_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming centered on the cursor."""
        if not self.original_pixmap:
            return

        # Calculate zoom
        zoom_in_factor = 1.25
        zoom_out_factor = 0.8
        old_scale = self.scale_factor
        if event.angleDelta().y() > 0:
            self.scale_factor *= zoom_in_factor
        else:
            self.scale_factor *= zoom_out_factor
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))

        self.update_display()
        
        # Adjust scrollbars to keep mouse position stable
        new_pos = self.scale_factor / old_scale * event.pos()
        delta = new_pos - event.pos()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())


# --- Asynchronous Pipeline Executor ---
class PipelineWorker(QThread):
    progress = pyqtSignal(int, str)  # (percentage, status_message)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

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
        try:
            result = self.image.copy()
            total_steps = len(self.pipeline)
            for i, step in enumerate(self.pipeline):
                func_name = step['name']
                params = step['params']
                
                # Emit progress update with the name of the function being applied
                status_msg = f"Step {i+1}/{total_steps}: Applying {func_name}..."
                self.progress.emit(int((i / total_steps) * 100), status_msg)

                if func_name in self.functions:
                    func = self.functions[func_name]
                    result = func(result, **params)
                else:
                    raise RuntimeError(f"Function '{func_name}' not found.")
            
            self.progress.emit(100, "Processing complete!")
            self.finished.emit(result)

        except Exception as e:
            error_msg = f"Error in '{func_name}': {e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


# --- Function Loader and Inspector ---
class FunctionLoader:
    def __init__(self, directory: str = "functions"):
        self.dir = Path(directory)
        self.functions: Dict[str, Callable] = {}
        self.function_info: Dict[str, Dict] = {}

    def scan(self):
        self.functions.clear()
        self.function_info.clear()

        if not self.dir.exists():
            self.dir.mkdir(parents=True)

        for file_path in self.dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            try:
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, 'process_image'):
                    func = getattr(module, 'process_image')
                    func_name = file_path.name # Use the file name as the unique ID
                    self.functions[func_name] = func
                    
                    self.function_info[func_name] = {
                        'name': func_name,
                        'doc': inspect.getdoc(func) or "No description provided.",
                        'params': self._get_params(func)
                    }
            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")

    def _get_params(self, func: Callable) -> Dict[str, Dict]:
        params = {}
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            if name == 'image': continue # Standard first argument
            params[name] = {
                'type': param.annotation,
                'default': param.default if param.default is not param.empty else None
            }
        return params

# --- Main Application Window ---
class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.processed_image = None
        
        self.function_loader = FunctionLoader()
        self.worker = PipelineWorker()
        
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)

        self.init_ui()
        self.load_functions()

    def init_ui(self):
        self.setWindowTitle("Advanced Image Processing GUI")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { padding: 8px; border-radius: 4px; }
            QPushButton#ProcessBtn { background-color: #4CAF50; color: white; font-weight: bold; }
            QPushButton#ProcessBtn:disabled { background-color: #9E9E9E; }
            QListWidget::item:selected { background-color: #4a8dff; color: white; }
        """)

        # -- Main Layout --
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # -- Panels --
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_center_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([300, 800, 300])

        # -- Status Bar --
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(15)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        group = QGroupBox("Function Library")
        group_layout = QVBoxLayout(group)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search functions...")
        self.search_input.textChanged.connect(self.filter_functions)
        group_layout.addWidget(self.search_input)
        
        self.function_list = QListWidget()
        self.function_list.itemDoubleClicked.connect(self.add_selected_function_to_pipeline)
        group_layout.addWidget(self.function_list)
        
        refresh_btn = QPushButton("Refresh Functions")
        refresh_btn.clicked.connect(self.load_functions)
        group_layout.addWidget(refresh_btn)
        
        layout.addWidget(group)
        return panel

    def _create_center_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Toolbar
        toolbar = QHBoxLayout()
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        toolbar.addWidget(load_btn)
        save_btn = QPushButton("Save Processed Image")
        save_btn.clicked.connect(self.save_image)
        toolbar.addWidget(save_btn)
        layout.addLayout(toolbar)
        
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer)

        self.image_info_label = QLabel("No image loaded.")
        self.image_info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        layout.addWidget(self.image_info_label)
        
        return panel
        
    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        group = QGroupBox("Processing Pipeline")
        group_layout = QVBoxLayout(group)
        
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QListWidget.InternalMove)
        self.pipeline_list.itemDoubleClicked.connect(self.edit_pipeline_item_params)
        group_layout.addWidget(self.pipeline_list)
        
        # Pipeline Controls
        controls_layout = QHBoxLayout()
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_selected_from_pipeline)
        controls_layout.addWidget(remove_btn)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.pipeline_list.clear)
        controls_layout.addWidget(clear_btn)
        group_layout.addLayout(controls_layout)

        # Pipeline Save/Load
        pipeline_io_layout = QHBoxLayout()
        save_pipe_btn = QPushButton("Save Pipeline")
        save_pipe_btn.clicked.connect(self.save_pipeline)
        pipeline_io_layout.addWidget(save_pipe_btn)
        load_pipe_btn = QPushButton("Load Pipeline")
        load_pipe_btn.clicked.connect(self.load_pipeline)
        pipeline_io_layout.addWidget(load_pipe_btn)
        group_layout.addLayout(pipeline_io_layout)
        
        # Process Button
        self.process_btn = QPushButton("Process Image")
        self.process_btn.setObjectName("ProcessBtn")
        self.process_btn.clicked.connect(self.process_image)
        group_layout.addWidget(self.process_btn)
        
        layout.addWidget(group)
        return panel

    # --- Core Functionality ---
    def load_functions(self):
        self.function_list.clear()
        self.function_loader.scan()
        for name in sorted(self.function_loader.functions.keys()):
            self.function_list.addItem(name)
        self.status_bar.showMessage(f"Loaded {len(self.function_loader.functions)} functions.", 3000)

    def filter_functions(self, text):
        for i in range(self.function_list.count()):
            item = self.function_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def add_selected_function_to_pipeline(self, item):
        func_name = item.text()
        func_info = self.function_loader.function_info[func_name]
        
        # Create a dictionary for this step with default params
        pipeline_step = {
            'name': func_name,
            'params': {name: p_info['default'] for name, p_info in func_info['params'].items()}
        }
        
        list_item = QListWidgetItem(self._format_pipeline_item_text(pipeline_step))
        list_item.setData(Qt.UserRole, pipeline_step) # Store the dictionary
        self.pipeline_list.addItem(list_item)

    def edit_pipeline_item_params(self, item):
        pipeline_step = item.data(Qt.UserRole)
        func_name = pipeline_step['name']
        func_info = self.function_loader.function_info[func_name]
        
        if not func_info['params']:
            QMessageBox.information(self, "No Parameters", f"'{func_name}' has no parameters to edit.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Parameters for {func_name}")
        form_layout = QFormLayout(dialog)
        widgets = {}

        for name, p_info in func_info['params'].items():
            current_val = pipeline_step['params'].get(name, p_info['default'])
            widget = None
            if p_info['type'] is bool:
                widget = QCheckBox()
                if current_val: widget.setChecked(True)
            elif p_info['type'] is int:
                widget = QSpinBox()
                widget.setRange(-10000, 10000)
                if current_val is not None: widget.setValue(current_val)
            elif p_info['type'] is float:
                widget = QDoubleSpinBox()
                widget.setRange(-10000.0, 10000.0)
                if current_val is not None: widget.setValue(current_val)
            else: # string or other
                widget = QLineEdit()
                if current_val is not None: widget.setText(str(current_val))
            
            if widget:
                form_layout.addRow(QLabel(f"{name} ({p_info['type'].__name__}):"), widget)
                widgets[name] = widget

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form_layout.addRow(buttons)

        if dialog.exec_() == QDialog.Accepted:
            for name, widget in widgets.items():
                if isinstance(widget, QCheckBox): pipeline_step['params'][name] = widget.isChecked()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)): pipeline_step['params'][name] = widget.value()
                else: pipeline_step['params'][name] = widget.text() # Keep as string for now
            item.setText(self._format_pipeline_item_text(pipeline_step))
            item.setData(Qt.UserRole, pipeline_step)

    def remove_selected_from_pipeline(self):
        selected_items = self.pipeline_list.selectedItems()
        if not selected_items: return
        for item in selected_items:
            self.pipeline_list.takeItem(self.pipeline_list.row(item))

    def _format_pipeline_item_text(self, step):
        params_str = ", ".join(f"{k}={v}" for k, v in step['params'].items() if v is not None)
        return f"{step['name']}" + (f" ({params_str})" if params_str else "")

    def process_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        if self.pipeline_list.count() == 0:
            QMessageBox.warning(self, "Warning", "The processing pipeline is empty.")
            return

        pipeline_data = [self.pipeline_list.item(i).data(Qt.UserRole) for i in range(self.pipeline_list.count())]
        
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.worker.set_data(self.current_image, pipeline_data, self.function_loader.functions)
        self.worker.start()

    def on_processing_finished(self, result_image):
        self.processed_image = result_image
        self.image_viewer.set_image(result_image)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Processing finished successfully!", 5000)

    def on_processing_error(self, error_message):
        QMessageBox.critical(self, "Pipeline Error", error_message)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("An error occurred during processing.", 5000)

    def update_progress(self, percentage, message):
        self.progress_bar.setValue(percentage)
        self.status_bar.showMessage(message)

    # --- I/O Operations ---
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            self.current_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if self.current_image is None:
                QMessageBox.critical(self, "Error", f"Failed to load image from {path}")
                return
            self.image_viewer.set_image(self.current_image)
            h, w = self.current_image.shape[:2]
            c = self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
            self.image_info_label.setText(f"Loaded: {os.path.basename(path)} ({w}x{h}, {c} channels)")
            self.status_bar.showMessage("Image loaded.", 3000)

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Processed Image", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
        if path:
            try:
                cv2.imwrite(path, self.processed_image)
                self.status_bar.showMessage(f"Image saved to {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")

    def save_pipeline(self):
        if self.pipeline_list.count() == 0:
            QMessageBox.warning(self, "Warning", "Pipeline is empty, nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline", "", "JSON Files (*.json)")
        if path:
            pipeline_data = [self.pipeline_list.item(i).data(Qt.UserRole) for i in range(self.pipeline_list.count())]
            try:
                with open(path, 'w') as f:
                    json.dump(pipeline_data, f, indent=2)
                self.status_bar.showMessage(f"Pipeline saved to {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save pipeline: {e}")

    def load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Pipeline", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    pipeline_data = json.load(f)
                self.pipeline_list.clear()
                for step in pipeline_data:
                    list_item = QListWidgetItem(self._format_pipeline_item_text(step))
                    list_item.setData(Qt.UserRole, step)
                    self.pipeline_list.addItem(list_item)
                self.status_bar.showMessage(f"Pipeline loaded from {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load pipeline: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())
