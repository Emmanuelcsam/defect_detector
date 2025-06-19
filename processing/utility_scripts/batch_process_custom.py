#!/usr/bin/env python
"""
Batch process images with custom settings for the pixel voting system
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

def create_custom_knowledge(output_dir: Path, settings: dict):
    """Create a custom knowledge file with specific settings"""
    knowledge = {
        'avg_core_radius_ratio': settings.get('core_ratio', 0.15),
        'avg_cladding_radius_ratio': settings.get('cladding_ratio', 0.5),
        'avg_center_offset': settings.get('center_offset', 0.02),
        'method_scores': settings.get('method_scores', {}),
        'method_accuracy': {}
    }
    
    knowledge_file = output_dir / "segmentation_knowledge.json"
    with open(knowledge_file, 'w') as f:
        json.dump(knowledge, f, indent=4)
    
    return knowledge_file

def batch_process_folder(input_folder: Path, settings: dict):
    """Process all images in a folder with custom settings"""
    print(f"\nBatch Processing: {input_folder}")
    print("=" * 60)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output = Path(f"batch_output_{timestamp}")
    batch_output.mkdir(exist_ok=True)
    
    # Save the custom settings
    settings_file = batch_output / "batch_settings.json"
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)
    
    # Create temporary output directory for the unified system
    temp_output = batch_output / "output"
    temp_output.mkdir(exist_ok=True)
    
    # Create custom knowledge file
    create_custom_knowledge(temp_output, settings)
    
    # Find all images
    image_extensions = settings.get('extensions', ['.jpg', '.jpeg', '.png', '.bmp'])
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_folder.glob(f'*{ext}'))
        image_files.extend(input_folder.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found with extensions: {image_extensions}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process statistics
    stats = {
        'total': len(image_files),
        'successful': 0,
        'failed': 0,
        'processing_times': [],
        'consensus_strengths': []
    }
    
    # Process each image
    for i, img_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        try:
            # Import and run the unified system
            import sys
            import subprocess
            
            # Create a script to run the separation
            runner_script = temp_output / "batch_runner.py"
            with open(runner_script, 'w') as f:
                f.write(f"""
import sys
sys.path.insert(0, r"{Path.cwd()}")
from separation import UnifiedSegmentationSystem

# Initialize system with custom output directory
system = UnifiedSegmentationSystem("zones_methods")
system.output_dir = r"{temp_output}"

# Process the image
result = system.process_image(r"{img_path}")

# Save result status
import json
status = {{'success': result is not None}}
if result:
    status['consensus_strength'] = result.get('consensus_strength', {{}})
    
with open(r"{temp_output / 'last_result.json'}", 'w') as f:
    json.dump(status, f)
""")
            
            # Run the processing
            import time
            start_time = time.time()
            
            process = subprocess.run(
                [sys.executable, str(runner_script)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            processing_time = time.time() - start_time
            stats['processing_times'].append(processing_time)
            
            # Check result
            result_file = temp_output / 'last_result.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result_status = json.load(f)
                    
                if result_status.get('success'):
                    stats['successful'] += 1
                    
                    # Move results to batch output
                    result_dirs = list(temp_output.glob(f"{img_path.stem}_*"))
                    if result_dirs:
                        latest = sorted(result_dirs, key=lambda x: x.stat().st_mtime)[-1]
                        dest = batch_output / latest.name
                        shutil.move(str(latest), str(dest))
                        
                        # Record consensus strength
                        if 'consensus_strength' in result_status:
                            stats['consensus_strengths'].append(result_status['consensus_strength'])
                else:
                    stats['failed'] += 1
                    print(f"  ✗ Processing failed")
            else:
                stats['failed'] += 1
                print(f"  ✗ No result generated")
                
        except Exception as e:
            stats['failed'] += 1
            print(f"  ✗ Error: {e}")
    
    # Save batch statistics
    stats_file = batch_output / "batch_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Batch Processing Complete")
    print("=" * 60)
    print(f"Total images: {stats['total']}")
    print(f"Successful: {stats['successful']} ({100*stats['successful']/stats['total']:.1f}%)")
    print(f"Failed: {stats['failed']}")
    
    if stats['processing_times']:
        avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
        print(f"Average processing time: {avg_time:.1f} seconds")
    
    print(f"\nResults saved to: {batch_output}")
    
    # Clean up temporary output
    shutil.rmtree(temp_output, ignore_errors=True)

def main():
    """Main function with examples"""
    print("Batch Processing with Custom Settings")
    print("=" * 60)
    
    # Example 1: Process with custom method weights
    print("\nExample 1: Custom method weights for noisy images")
    settings1 = {
        'method_scores': {
            'segmentation': 1.8,          # High trust in consensus method
            'gradient_approach': 1.5,      # Good for noisy images
            'guess_approach': 1.2,         # Multi-modal analysis
            'threshold_seperation': 0.3,   # Low trust for simple threshold
            'computational_separation': 0.5,  # Often gives wrong bounds
            'hough_seperation': 0.8,       # OK for clear circles
            'adaptive_intensity_approach': 0.7
        },
        'core_ratio': 0.15,
        'cladding_ratio': 0.45
    }
    
    # Example 2: Settings for clean, high-quality images
    print("\nExample 2: Settings for clean fiber images")
    settings2 = {
        'method_scores': {
            'hough_seperation': 1.5,       # Works well on clean images
            'computational_separation': 1.3,  # Geometric methods good
            'segmentation': 1.2,
            'gradient_approach': 1.0,
            'threshold_seperation': 0.8,
            'guess_approach': 1.0,
            'adaptive_intensity_approach': 0.6
        }
    }
    
    # Example 3: Settings for damaged or irregular fibers
    print("\nExample 3: Settings for damaged fibers")
    settings3 = {
        'method_scores': {
            'guess_approach': 1.6,         # Adaptive to irregularities
            'adaptive_intensity_approach': 1.4,  # Handles varying intensity
            'gradient_approach': 1.3,      # Multi-method approach
            'segmentation': 1.5,          # Consensus-based
            'hough_seperation': 0.4,      # Poor for non-circular
            'computational_separation': 0.3,  # Assumes perfect circles
            'threshold_seperation': 0.6
        },
        'extensions': ['.jpg', '.png', '.tiff']  # Include more formats
    }
    
    # Interactive mode
    print("\nSelect processing mode:")
    print("1. Process folder with custom weights")
    print("2. Use preset for noisy images")
    print("3. Use preset for clean images")
    print("4. Use preset for damaged fibers")
    print("5. Exit")
    
    choice = input("\nChoice (1-5): ").strip()
    
    if choice == '5':
        print("Exiting.")
        return
    
    # Get folder path
    folder_path = input("\nEnter folder path containing images: ").strip().strip('"\'')
    folder_path = Path(folder_path)
    
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    # Select settings
    if choice == '1':
        # Custom settings
        print("\nEnter custom method scores (0.1-2.0, or press Enter for default 1.0):")
        custom_scores = {}
        methods = [
            'guess_approach', 'hough_seperation', 'segmentation',
            'threshold_seperation', 'adaptive_intensity_approach',
            'computational_separation', 'gradient_approach'
        ]
        
        for method in methods:
            score_str = input(f"{method} [1.0]: ").strip()
            if score_str:
                try:
                    score = float(score_str)
                    if 0.1 <= score <= 2.0:
                        custom_scores[method] = score
                except ValueError:
                    print(f"Invalid score, using 1.0 for {method}")
        
        settings = {'method_scores': custom_scores}
        
    elif choice == '2':
        settings = settings1  # Noisy images
    elif choice == '3':
        settings = settings2  # Clean images
    elif choice == '4':
        settings = settings3  # Damaged fibers
    else:
        print("Invalid choice")
        return
    
    # Run batch processing
    batch_process_folder(folder_path, settings)

if __name__ == "__main__":
    main()
