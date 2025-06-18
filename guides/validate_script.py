#!/usr/bin/env python3
"""
Script Validator for Image Processing GUI
Checks if your scripts are compatible with the GUI
"""

import sys
import importlib.util
import inspect
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any

class ScriptValidator:
    def __init__(self):
        self.test_images = self._create_test_images()
        
    def _create_test_images(self) -> Dict[str, np.ndarray]:
        """Create test images of different types"""
        return {
            'grayscale_small': np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            'grayscale_large': np.random.randint(0, 255, (800, 600), dtype=np.uint8),
            'color_small': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            'color_large': np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8),
            'color_rgba': np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8),
            'float_image': np.random.rand(100, 100).astype(np.float32) * 255,
        }
        
    def validate_script(self, script_path: Path) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single script.
        
        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check if file exists
        if not script_path.exists():
            errors.append(f"File not found: {script_path}")
            return False, errors, warnings
            
        # Try to load the module
        try:
            spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            errors.append(f"Failed to import script: {e}")
            return False, errors, warnings
            
        # Check for process_image function
        if not hasattr(module, 'process_image'):
            errors.append("Missing required function: process_image")
            return False, errors, warnings
            
        process_func = getattr(module, 'process_image')
        
        # Check if it's callable
        if not callable(process_func):
            errors.append("process_image is not a callable function")
            return False, errors, warnings
            
        # Check function signature
        sig = inspect.signature(process_func)
        params = list(sig.parameters.keys())
        
        if len(params) == 0:
            errors.append("process_image must accept at least one parameter (image)")
            return False, errors, warnings
            
        if params[0] != 'image':
            warnings.append(f"First parameter should be named 'image', not '{params[0]}'")
            
        # Check for type hints
        param_obj = sig.parameters[params[0]]
        if param_obj.annotation == inspect.Parameter.empty:
            warnings.append("Missing type hint for image parameter (should be np.ndarray)")
            
        # Check return type hint
        if sig.return_annotation == inspect.Signature.empty:
            warnings.append("Missing return type hint (should be -> np.ndarray)")
            
        # Check for docstring
        if not inspect.getdoc(process_func):
            warnings.append("Missing docstring for process_image function")
            
        # Test the function with different image types
        test_results = self._test_function(process_func)
        
        for test_name, (success, error_msg) in test_results.items():
            if not success:
                errors.append(f"Failed test '{test_name}': {error_msg}")
                
        # Check for common issues
        with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Check for hardcoded paths
        if any(pattern in content for pattern in ['C:\\', 'D:\\', '/home/', '/Users/']):
            warnings.append("Script contains hardcoded paths")
            
        # Check for display functions
        if any(func in content for func in ['cv2.imshow', 'plt.show', 'cv2.waitKey']):
            warnings.append("Script contains display functions (cv2.imshow, plt.show)")
            
        # Check for input operations
        if 'input(' in content:
            warnings.append("Script contains input() calls which will block the GUI")
            
        # Check parameter defaults
        for param_name in params[1:]:  # Skip 'image'
            param_obj = sig.parameters[param_name]
            if param_obj.default == inspect.Parameter.empty:
                warnings.append(f"Parameter '{param_name}' has no default value")
                
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
        
    def _test_function(self, func) -> Dict[str, Tuple[bool, str]]:
        """Test the function with various image types"""
        results = {}
        
        for test_name, test_image in self.test_images.items():
            try:
                # Get function signature to check for additional parameters
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())[1:]  # Skip 'image'
                
                # Build kwargs with default values
                kwargs = {}
                for param_name in params:
                    param = sig.parameters[param_name]
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                        
                # Call the function
                result = func(test_image.copy(), **kwargs)
                
                # Validate result
                if not isinstance(result, np.ndarray):
                    results[test_name] = (False, f"Returned {type(result).__name__}, expected np.ndarray")
                elif result.dtype not in [np.uint8, np.float32, np.float64]:
                    results[test_name] = (False, f"Unusual dtype: {result.dtype}")
                else:
                    results[test_name] = (True, "")
                    
            except Exception as e:
                results[test_name] = (False, str(e))
                
        return results
        
    def validate_directory(self, directory: Path) -> Dict[str, Tuple[bool, List[str], List[str]]]:
        """Validate all Python scripts in a directory"""
        results = {}
        
        for script_path in directory.glob("*.py"):
            if script_path.name.startswith('_'):
                continue
                
            is_valid, errors, warnings = self.validate_script(script_path)
            results[script_path.name] = (is_valid, errors, warnings)
            
        return results


def print_validation_results(results: Dict[str, Tuple[bool, List[str], List[str]]]):
    """Pretty print validation results"""
    total = len(results)
    valid = sum(1 for _, (is_valid, _, _) in results.items() if is_valid)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Total scripts: {total}")
    print(f"Valid scripts: {valid}")
    print(f"Invalid scripts: {total - valid}")
    print("="*60 + "\n")
    
    for script_name, (is_valid, errors, warnings) in sorted(results.items()):
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"\n{script_name}: {status}")
        
        if errors:
            print("  Errors:")
            for error in errors:
                print(f"    ❌ {error}")
                
        if warnings:
            print("  Warnings:")
            for warning in warnings:
                print(f"    ⚠️  {warning}")
                
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if valid == total:
        print("✅ All scripts are valid and ready to use!")
    else:
        print(f"❌ {total - valid} scripts need to be fixed")
        print("\nCommon fixes:")
        print("1. Add a process_image(image: np.ndarray) function")
        print("2. Remove hardcoded paths")
        print("3. Remove cv2.imshow() and plt.show() calls")
        print("4. Add default values for all parameters")
        print("5. Return a numpy array from process_image")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate scripts for GUI compatibility")
    parser.add_argument('path', nargs='?', default='scripts',
                       help='Script file or directory to validate (default: scripts)')
    parser.add_argument('--fix', action='store_true',
                       help='Attempt to fix common issues (creates .fixed.py files)')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    validator = ScriptValidator()
    
    if path.is_file():
        # Validate single file
        print(f"Validating {path.name}...")
        is_valid, errors, warnings = validator.validate_script(path)
        
        results = {path.name: (is_valid, errors, warnings)}
        print_validation_results(results)
        
    elif path.is_dir():
        # Validate directory
        print(f"Validating scripts in {path}...")
        results = validator.validate_directory(path)
        print_validation_results(results)
        
    else:
        print(f"Error: {path} not found")
        sys.exit(1)
        
    # If requested, attempt fixes
    if args.fix and any(not v[0] for v in results.values()):
        print("\n" + "="*60)
        print("ATTEMPTING FIXES")
        print("="*60)
        print("Fixed files will be saved with .fixed.py extension")
        print("(Feature coming soon...)")


if __name__ == "__main__":
    main()
