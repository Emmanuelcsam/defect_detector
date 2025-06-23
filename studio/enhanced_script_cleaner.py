#!/usr/bin/env python3
"""
Enhanced Script Cleaner
=======================
Specifically designed to fix Unicode errors and create GUI-compatible wrappers
from existing OpenCV scripts.
"""

import os
import re
import ast
from pathlib import Path
import shutil
import chardet

class EnhancedScriptCleaner:
    def __init__(self, source_dir=".", output_dir="scripts", verbose=True):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Patterns for different types of operations
        self.operation_patterns = {
            'gaussian_blur': [r'cv2\.GaussianBlur', r'gaussian.*blur'],
            'median_blur': [r'cv2\.medianBlur', r'median.*filter'],
            'bilateral_filter': [r'cv2\.bilateralFilter'],
            'canny_edge': [r'cv2\.Canny', r'canny.*edge'],
            'sobel_edge': [r'cv2\.Sobel', r'sobel'],
            'laplacian_edge': [r'cv2\.Laplacian', r'laplacian'],
            'threshold': [r'cv2\.threshold', r'thresh'],
            'adaptive_threshold': [r'cv2\.adaptiveThreshold'],
            'morphology': [r'cv2\.morphologyEx', r'cv2\.erode', r'cv2\.dilate'],
            'circle_detection': [r'cv2\.HoughCircles', r'circle.*detect'],
            'contour': [r'cv2\.findContours', r'contour'],
            'histogram': [r'cv2\.equalizeHist', r'histogram'],
            'clahe': [r'cv2\.createCLAHE', r'clahe'],
            'grayscale': [r'cv2\.cvtColor.*GRAY', r'grayscale', r'gray'],
            'colormap': [r'cv2\.applyColorMap', r'colormap'],
            'filter2D': [r'cv2\.filter2D'],
            'mask': [r'mask', r'bitwise_and'],
        }
        
    def detect_encoding(self, file_path):
        """Detect file encoding to handle Unicode errors"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding']
        except:
            return 'utf-8'
            
    def read_script_safely(self, file_path):
        """Read script with proper encoding handling"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        # Try detected encoding first
        detected_encoding = self.detect_encoding(file_path)
        if detected_encoding:
            encodings.insert(0, detected_encoding)
            
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                    
                # Clean any remaining Unicode issues
                content = content.encode('ascii', 'ignore').decode('ascii')
                return content
            except:
                continue
                
        # Last resort - read as binary and decode
        with open(file_path, 'rb') as f:
            content = f.read()
            # Remove null bytes and decode
            content = content.replace(b'\x00', b'')
            return content.decode('ascii', 'ignore')
            
    def extract_imports(self, content):
        """Extract import statements from the script"""
        imports = set()
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
            r'import\s+(\w+)\s+as\s+\w+'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            imports.update(matches)
            
        # Always include these
        imports.update(['cv2', 'numpy'])
        
        return sorted(imports)
        
    def detect_operations(self, content):
        """Detect which operations are performed in the script"""
        detected = []
        
        for op_name, patterns in self.operation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    detected.append(op_name)
                    break
                    
        return detected
        
    def extract_parameters(self, content):
        """Try to extract parameters used in the script"""
        params = {}
        
        # Common parameter patterns
        param_patterns = {
            'kernel_size': [r'kernel.*size\s*=\s*(\d+)', r'\((\d+),\s*\1\)'],
            'threshold': [r'threshold\s*=\s*(\d+)', r'thresh.*=\s*(\d+)'],
            'sigma': [r'sigma\s*=\s*(\d+\.?\d*)', r'GaussianBlur.*,\s*(\d+\.?\d*)\)'],
            'iterations': [r'iterations\s*=\s*(\d+)'],
            'clip_limit': [r'clipLimit\s*=\s*(\d+\.?\d*)'],
        }
        
        for param_name, patterns in param_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    try:
                        value = float(match.group(1))
                        params[param_name] = int(value) if value.is_integer() else value
                        break
                    except:
                        pass
                        
        return params
        
    def generate_wrapper(self, script_name, content):
        """Generate a clean wrapper for the script"""
        # Clean the script name
        clean_name = script_name.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Extract information
        imports = self.extract_imports(content)
        operations = self.detect_operations(content)
        parameters = self.extract_parameters(content)
        
        # Generate description
        description = f"Processed from {script_name}"
        if operations:
            description += f" - Detected operations: {', '.join(operations[:3])}"
            
        # Generate the wrapper
        wrapper = f'''"""{description}"""
import cv2
import numpy as np
'''
        
        # Add any additional imports (excluding standard ones)
        additional_imports = [imp for imp in imports if imp not in ['cv2', 'numpy', 'os', 'sys']]
        if additional_imports:
            for imp in additional_imports:
                if imp in ['matplotlib', 'PIL', 'skimage']:  # Common image processing libraries
                    wrapper += f"import {imp}\n"
                    
        # Generate the process_image function
        wrapper += "\ndef process_image(image: np.ndarray"
        
        # Add detected parameters
        if parameters:
            for param_name, default_value in parameters.items():
                wrapper += f", {param_name}: float = {default_value}"
                
        wrapper += ") -> np.ndarray:\n"
        wrapper += f'    """\n    {description}\n    \n    Args:\n        image: Input image\n'
        
        if parameters:
            for param_name in parameters:
                wrapper += f'        {param_name}: {param_name.replace("_", " ").capitalize()}\n'
                
        wrapper += '    \n    Returns:\n        Processed image\n    """\n'
        
        # Generate processing code based on detected operations
        wrapper += self._generate_processing_code(operations, parameters)
        
        return wrapper
        
    def _generate_processing_code(self, operations, parameters):
        """Generate the actual processing code based on detected operations"""
        code = "    try:\n        result = image.copy()\n        \n"
        
        # Generate code for each detected operation
        if 'grayscale' in operations:
            code += "        # Convert to grayscale if needed\n"
            code += "        if len(result.shape) == 3:\n"
            code += "            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n        \n"
            
        if 'gaussian_blur' in operations:
            code += "        # Apply Gaussian blur\n"
            if 'kernel_size' in parameters:
                code += "        kernel_size = int(kernel_size)\n"
                code += "        if kernel_size % 2 == 0:\n"
                code += "            kernel_size += 1\n"
                code += "        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), "
                code += f"{parameters.get('sigma', 0)})\n        \n"
            else:
                code += "        result = cv2.GaussianBlur(result, (5, 5), 0)\n        \n"
                
        if 'median_blur' in operations:
            code += "        # Apply median blur\n"
            code += f"        result = cv2.medianBlur(result, {parameters.get('kernel_size', 5)})\n        \n"
            
        if 'canny_edge' in operations:
            code += "        # Apply Canny edge detection\n"
            code += "        if len(result.shape) == 3:\n"
            code += "            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n"
            code += f"        result = cv2.Canny(result, {parameters.get('threshold', 50)}, "
            code += f"{parameters.get('threshold', 50) * 3})\n        \n"
            
        if 'threshold' in operations and 'adaptive_threshold' not in operations:
            code += "        # Apply threshold\n"
            code += "        if len(result.shape) == 3:\n"
            code += "            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n"
            code += f"        _, result = cv2.threshold(result, {parameters.get('threshold', 127)}, 255, cv2.THRESH_BINARY)\n        \n"
            
        if 'adaptive_threshold' in operations:
            code += "        # Apply adaptive threshold\n"
            code += "        if len(result.shape) == 3:\n"
            code += "            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n"
            code += "        result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "
            code += "cv2.THRESH_BINARY, 11, 2)\n        \n"
            
        if 'morphology' in operations:
            code += "        # Apply morphological operation\n"
            code += "        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n"
            code += "        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)\n        \n"
            
        if 'circle_detection' in operations:
            code += "        # Detect circles\n"
            code += "        if len(result.shape) == 2:\n"
            code += "            display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)\n"
            code += "        else:\n"
            code += "            display = result.copy()\n"
            code += "        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY) if len(display.shape) == 3 else display\n"
            code += "        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30)\n"
            code += "        if circles is not None:\n"
            code += "            circles = np.uint16(np.around(circles))\n"
            code += "            for i in circles[0, :]:\n"
            code += "                cv2.circle(display, (i[0], i[1]), i[2], (0, 255, 0), 2)\n"
            code += "        result = display\n        \n"
            
        if 'histogram' in operations:
            code += "        # Apply histogram equalization\n"
            code += "        if len(result.shape) == 3:\n"
            code += "            ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)\n"
            code += "            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])\n"
            code += "            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)\n"
            code += "        else:\n"
            code += "            result = cv2.equalizeHist(result)\n        \n"
            
        if 'clahe' in operations:
            code += "        # Apply CLAHE\n"
            code += f"        clahe = cv2.createCLAHE(clipLimit={parameters.get('clip_limit', 2.0)}, tileGridSize=(8,8))\n"
            code += "        if len(result.shape) == 3:\n"
            code += "            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)\n"
            code += "            lab[:, :, 0] = clahe.apply(lab[:, :, 0])\n"
            code += "            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)\n"
            code += "        else:\n"
            code += "            result = clahe.apply(result)\n        \n"
            
        if not operations:
            code += "        # Add your processing logic here\n"
            code += "        # This is a placeholder - modify based on the original script\n        \n"
            
        code += "        return result\n        \n"
        code += "    except Exception as e:\n"
        code += f'        print(f"Error in processing: {{e}}")\n'
        code += "        return image\n"
        
        return code
        
    def clean_script(self, script_path):
        """Clean a single script and create a wrapper"""
        try:
            # Read the script safely
            content = self.read_script_safely(script_path)
            
            # Clean common issues
            content = self.clean_common_issues(content)
            
            # Check if it already has process_image function
            if 'def process_image' in content and 'hardcoded' not in content.lower():
                # Just clean it and save
                output_path = self.output_dir / script_path.name
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Already has process_image function"
                
            # Generate wrapper
            wrapper = self.generate_wrapper(script_path.name, content)
            
            # Save the wrapper
            output_path = self.output_dir / script_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(wrapper)
                
            return True, "Successfully created wrapper"
            
        except Exception as e:
            return False, str(e)
            
    def clean_common_issues(self, content):
        """Clean common issues in scripts"""
        # Remove hardcoded paths (Windows and Unix)
        content = re.sub(r'[A-Za-z]:[\\\/][^\'";\n]+', '', content)
        content = re.sub(r'\/home\/[^\'";\n]+', '', content)
        content = re.sub(r'\/Users\/[^\'";\n]+', '', content)
        
        # Remove common problematic lines
        lines_to_remove = [
            r'img_path\s*=.*',
            r'image_path\s*=.*',
            r'base_path\s*=.*',
            r'cv2\.imshow\(.*\)',
            r'cv2\.waitKey\(.*\)',
            r'cv2\.destroyAllWindows\(.*\)',
            r'plt\.show\(.*\)',
        ]
        
        for pattern in lines_to_remove:
            content = re.sub(pattern, '', content)
            
        return content
        
    def clean_all_scripts(self):
        """Clean all scripts in the source directory"""
        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }
        
        # Get all Python files
        script_files = list(self.source_dir.glob("*.py"))
        
        if not script_files:
            print(f"No Python files found in {self.source_dir}")
            return results
            
        print(f"Found {len(script_files)} scripts to process\n")
        
        for script_path in script_files:
            # Skip certain files
            if script_path.name.startswith('_') or script_path.name in [
                'setup_gui.py', 'image_processor_gui.py', 'enhanced_script_cleaner.py'
            ]:
                results['skipped'].append(script_path.name)
                if self.verbose:
                    print(f"⏭️  Skipped: {script_path.name}")
                continue
                
            if self.verbose:
                print(f"🔧 Processing: {script_path.name}...", end=' ')
                
            success, message = self.clean_script(script_path)
            
            if success:
                results['success'].append(script_path.name)
                if self.verbose:
                    print(f"✅ {message}")
            else:
                results['failed'].append((script_path.name, message))
                if self.verbose:
                    print(f"❌ Failed: {message}")
                    
        # Print summary
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        print(f"✅ Successfully processed: {len(results['success'])} scripts")
        print(f"❌ Failed: {len(results['failed'])} scripts")
        print(f"⏭️  Skipped: {len(results['skipped'])} scripts")
        
        if results['failed'] and self.verbose:
            print("\nFailed scripts:")
            for name, error in results['failed']:
                print(f"  - {name}: {error}")
                
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean scripts with Unicode errors and create GUI-compatible wrappers"
    )
    parser.add_argument(
        '--source', 
        default='.', 
        help='Source directory containing scripts (default: current directory)'
    )
    parser.add_argument(
        '--output', 
        default='scripts', 
        help='Output directory for cleaned scripts (default: scripts)'
    )
    parser.add_argument(
        '--quiet', 
        action='store_true', 
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    print("🧹 Enhanced Script Cleaner")
    print("=" * 60)
    print(f"Source directory: {args.source}")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    cleaner = EnhancedScriptCleaner(
        source_dir=args.source,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    results = cleaner.clean_all_scripts()
    
    if results['success']:
        print(f"\n✅ Cleaning complete! Cleaned scripts are in '{args.output}' directory")
        print("\nNext steps:")
        print("1. Review the cleaned scripts in the output directory")
        print("2. Run the Image Processing GUI:")
        print("   python image_processor_gui.py")
        print("\nThe GUI will automatically load all scripts from the 'scripts' directory")


if __name__ == "__main__":
    main()
