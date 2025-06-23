import os
import sys
import json
import shutil
import time
from pathlib import Path
import logging
import shlex

# --- Setup Logging ---
# Configures a logger to print detailed, timestamped messages to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Add script directories to Python path ---
# This ensures that we can import our custom modules (process, separation, detection, data_acquisition)
# as long as they are in the same directory as this script.
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# --- Import from your custom scripts ---
# This block attempts to import the necessary components from your other scripts.
# If any import fails, it will log a fatal error and exit.
try:
    from process import reimagine_image
    from separation import UnifiedSegmentationSystem
    from detection import OmniFiberAnalyzer, OmniConfig
    from data_acquisition import integrate_with_pipeline as run_data_acquisition
    logging.info("Successfully imported all processing & analysis modules including data acquisition.")
except ImportError as e:
    logging.error(f"Fatal Error: Failed to import a required module: {e}")
    logging.error("Please ensure process.py, separation.py, detection.py, and data_acquisition.py are in the same directory as app.py.")
    sys.exit(1)


class PipelineOrchestrator:
    """
    This class manages the entire multi-stage defect analysis pipeline.
    It controls the flow from processing to separation to detection to final data acquisition.
    """
    def __init__(self, config_path):
        """Initializes the orchestrator with configuration."""
        logging.info("Initializing Pipeline Orchestrator...")
        self.config_path = Path(config_path).resolve()  # Store absolute config path
        self.config = self.load_config(config_path)
        self.config = self.resolve_config_paths(self.config)  # Resolve relative paths
        self.results_base_dir = Path(self.config['paths']['results_dir'])
        self.results_base_dir.mkdir(parents=True, exist_ok=True)  # Create with parents
        logging.info(f"Results will be saved in: {self.results_base_dir}")

    def load_config(self, config_path):
        """Loads the JSON configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Successfully loaded configuration from {config_path}")
            return config
        except Exception as e:
            logging.error(f"Fatal Error: Could not load or parse config.json: {e}")
            sys.exit(1)

    def resolve_config_paths(self, config):
        """Convert relative paths in config to absolute paths based on config file location"""
        config_dir = self.config_path.parent
        
        # Update paths to be absolute
        for key in ['results_dir', 'zones_methods_dir', 'detection_knowledge_base']:
            if key in config['paths']:
                path = Path(config['paths'][key])
                if not path.is_absolute():
                    # Make it absolute relative to the config directory
                    config['paths'][key] = str(config_dir / path)
        
        return config

    def run_full_pipeline(self, input_image_path: Path):
        """
        Runs the entire analysis pipeline for a single image, creating a
        dedicated folder for its results.
        """
        start_time = time.time()
        logging.info(f"--- Starting full pipeline for: {input_image_path.name} ---")

        # Create a unique directory for this image's results
        run_dir = self.results_base_dir / input_image_path.stem
        run_dir.mkdir(exist_ok=True)

        # === STAGE 1: PROCESSING (REIMAGINE) ===
        reimagined_dir, all_images_to_separate = self.run_processing_stage(input_image_path, run_dir)

        # === STAGE 2: SEPARATION (ZONING) ===
        separated_dir, all_images_to_detect = self.run_separation_stage(all_images_to_separate, run_dir, input_image_path)

        # === STAGE 3: DETECTION (ANALYSIS) ===
        self.run_detection_stage(all_images_to_detect, run_dir)

        # === STAGE 4: DATA ACQUISITION (FINAL ANALYSIS) ===
        final_report = self.run_data_acquisition_stage(input_image_path, run_dir)

        end_time = time.time()
        logging.info(f"--- Pipeline for {input_image_path.name} completed in {end_time - start_time:.2f} seconds ---")
        
        # Log final summary
        if final_report and 'analysis_summary' in final_report:
            summary = final_report['analysis_summary']
            logging.info(f"FINAL RESULTS: Status={summary['pass_fail_status']}, "
                        f"Quality Score={summary['quality_score']}/100, "
                        f"Total Defects={summary['total_merged_defects']}")
        
        return final_report

    def run_processing_stage(self, input_image_path, run_dir):
        """Runs the process.py script to generate multiple image versions."""
        logging.info(">>> STAGE 1: PROCESSING - Reimagining images...")
        process_cfg = self.config['process_settings']
        reimagined_dir = run_dir / process_cfg['output_folder_name']
        
        try:
            # We call the imported function directly.
            reimagine_image(str(input_image_path), str(reimagined_dir))
        except Exception as e:
            logging.error(f"Error during reimagine_image for {input_image_path.name}: {e}")
            # Even if it fails, we continue with the original image
        
        # Gather all images for the next stage. This includes the original image
        # and all its reimagined versions.
        all_images_to_separate = [input_image_path]
        reimagined_files = list(reimagined_dir.glob('*.jpg')) if reimagined_dir.exists() else []
        all_images_to_separate.extend(reimagined_files)
        
        logging.info(f"Processing stage complete. Found {len(reimagined_files)} reimagined images.")
        return reimagined_dir, all_images_to_separate

    def run_separation_stage(self, image_paths, run_dir, original_image_path):
        """Runs separation.py to create zoned regions for each image."""
        logging.info(">>> STAGE 2: SEPARATION - Generating zoned regions...")
        separation_cfg = self.config['separation_settings']
        zones_methods_dir = self.config['paths']['zones_methods_dir']
        separated_dir = run_dir / separation_cfg['output_folder_name']
        separated_dir.mkdir(exist_ok=True)
        
        all_separated_regions = []

        try:
            # Initialize the segmentation system with the correct methods directory
            separator = UnifiedSegmentationSystem(methods_dir=zones_methods_dir)
            
            for image_path in image_paths:
                logging.info(f"Separating image: {image_path.name}")
                # Define the output directory for this specific image's separation results
                image_separation_output_dir = separated_dir / image_path.stem
                
                # Run separation and get consensus masks
                consensus = separator.process_image(image_path, str(image_separation_output_dir))
                
                if consensus and consensus.get('saved_regions'):
                    all_separated_regions.extend([Path(p) for p in consensus['saved_regions']])
        
        except Exception as e:
            logging.error(f"A critical error occurred in the separation stage: {e}", exc_info=True)

        # The final list for detection must include all separated regions PLUS the un-separated original image.
        all_images_to_detect = all_separated_regions + [original_image_path]

        logging.info(f"Separation stage complete. Generated {len(all_separated_regions)} separated regions.")
        logging.info(f"Total inputs for detection stage: {len(all_images_to_detect)}")
        return separated_dir, all_images_to_detect
    
    def run_detection_stage(self, image_paths, run_dir):
        """Runs detection.py to perform defect analysis on all provided images."""
        logging.info(">>> STAGE 3: DETECTION - Analyzing for defects...")
        detection_cfg = self.config['detection_settings']
        detection_output_dir = run_dir / detection_cfg['output_folder_name']
        detection_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create the config object for the detector from our main config file
            detection_config = detection_cfg['config'].copy()
            
            # Handle the knowledge base path
            kb_path = self.config['paths'].get('detection_knowledge_base')
            if kb_path:
                detection_config['knowledge_base_path'] = kb_path
            
            # Map parameters to OmniConfig expected names
            # Handle parameters that might have different names
            omni_config_dict = {
                'knowledge_base_path': detection_config.get('knowledge_base_path'),
                'min_defect_size': detection_config.get('min_defect_size', 
                                                       detection_config.get('min_defect_area_px', 10)),
                'max_defect_size': detection_config.get('max_defect_size', 
                                                       detection_config.get('max_defect_area_px', 5000)),
                'severity_thresholds': detection_config.get('severity_thresholds'),
                'confidence_threshold': detection_config.get('confidence_threshold', 0.3),
                'anomaly_threshold_multiplier': detection_config.get('anomaly_threshold_multiplier', 2.5),
                'enable_visualization': detection_config.get('enable_visualization', 
                                                            detection_config.get('generate_json_report', True))
            }
            
            # Pass the mapped dictionary to the OmniConfig dataclass
            omni_config = OmniConfig(**omni_config_dict)

            # Initialize the analyzer once with the full configuration
            analyzer = OmniFiberAnalyzer(omni_config)

            for image_path in image_paths:
                logging.info(f"Detecting defects in: {image_path.name}")
                # Define the output dir for this specific image's detection report
                image_detection_output_dir = detection_output_dir / image_path.stem
                image_detection_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Run analysis. The modified detection.py accepts an output directory.
                analyzer.analyze_end_face(str(image_path), str(image_detection_output_dir))
        
        except Exception as e:
            logging.error(f"A critical error occurred in the detection stage: {e}", exc_info=True)
        
        logging.info("Detection stage complete.")

    def run_data_acquisition_stage(self, original_image_path, run_dir):
        """Runs data_acquisition.py to aggregate and analyze all detection results."""
        logging.info(">>> STAGE 4: DATA ACQUISITION - Aggregating and analyzing all results...")
        
        try:
            # Get clustering parameters from config if available
            data_acq_cfg = self.config.get('data_acquisition_settings', {})
            clustering_eps = data_acq_cfg.get('clustering_eps', 30.0)
            
            # Run data acquisition analysis
            final_report = run_data_acquisition(
                str(run_dir), 
                original_image_path.stem,
                clustering_eps=clustering_eps
            )
            
            if final_report:
                # Log summary of final results
                summary = final_report.get('analysis_summary', {})
                logging.info(f"Data acquisition complete. Final status: {summary.get('pass_fail_status', 'UNKNOWN')}")
                
                # Create a summary file in the root results directory for easy access
                summary_path = run_dir / "FINAL_SUMMARY.txt"
                with open(summary_path, 'w') as f:
                    f.write(f"FINAL ANALYSIS SUMMARY\n")
                    f.write(f"===================\n\n")
                    f.write(f"Image: {original_image_path.name}\n")
                    f.write(f"Status: {summary.get('pass_fail_status', 'UNKNOWN')}\n")
                    f.write(f"Quality Score: {summary.get('quality_score', 0)}/100\n")
                    f.write(f"Total Defects: {summary.get('total_merged_defects', 0)}\n")
                    
                    if summary.get('failure_reasons'):
                        f.write(f"\nFailure Reasons:\n")
                        for reason in summary['failure_reasons']:
                            f.write(f"  - {reason}\n")
                    
                    f.write(f"\nDetailed results available in: 4_final_analysis/\n")
                
                return final_report
            else:
                logging.error("Data acquisition stage failed to produce a report")
                return None
                
        except Exception as e:
            logging.error(f"Error during data acquisition stage: {e}", exc_info=True)
            return None

# --- New Interactive Functions ---

def ask_for_images() -> list[Path]:
    """
    Prompts the user to enter one or more image paths and validates them.
    Handles paths with spaces correctly if they are quoted.
    """
    print("\nEnter one or more full image paths. Separate paths with spaces.")
    print("Example: C:\\Users\\Test\\img1.png \"C:\\My Images\\test.png\"")
    paths_input = input("> ").strip()
    
    if not paths_input:
        return []
        
    # Use shlex to correctly parse command-line style input, handling quotes
    path_strings = shlex.split(paths_input)
    
    valid_paths = []
    invalid_paths = []
    for path_str in path_strings:
        path = Path(path_str)
        if path.is_file():
            valid_paths.append(path)
        else:
            invalid_paths.append(str(path))
            
    if invalid_paths:
        logging.warning(f"The following paths were not found and will be skipped: {', '.join(invalid_paths)}")
        
    return valid_paths

def ask_for_folder() -> Path | None:
    """Prompts the user for a single folder path and validates it."""
    folder_path_str = input("\nEnter the full path to the folder containing images: ").strip()
    
    # Handle quoted paths
    if folder_path_str:
        folder_path_str = shlex.split(folder_path_str)[0] if folder_path_str else ""
    
    if not folder_path_str:
        return None
        
    folder_path = Path(folder_path_str)
    
    if folder_path.is_dir():
        return folder_path
    else:
        logging.error(f"Directory not found: {folder_path}")
        return None

def main():
    """
    The main entry point of the application, featuring an interactive menu.
    """
    print("\n" + "="*80)
    print("UNIFIED FIBER OPTIC DEFECT DETECTION PIPELINE".center(80))
    print("Interactive Mode - Full Pipeline with Data Acquisition".center(80))
    print("="*80)
    
    # Get config path from user
    config_path_str = input("Enter path to config.json (or press Enter for default 'config.json'): ").strip()
    if not config_path_str:
        config_path_str = "config.json"

    # Remove leading/trailing quotes that might be pasted from file explorers
    if config_path_str.startswith('"') and config_path_str.endswith('"'):
        config_path_str = config_path_str[1:-1]
    
    config_path = Path(config_path_str)
    if not config_path.exists():
        logging.error(f"Fatal Error: Configuration file not found at: {config_path}")
        print("\nPlease run setup.py first to create the necessary files and directories.")
        sys.exit(1)
        
    # Initialize the orchestrator
    try:
        orchestrator = PipelineOrchestrator(str(config_path))
    except Exception as e:
        logging.error(f"Failed to initialize pipeline: {e}")
        print("\nPlease check your configuration file and ensure all required directories exist.")
        print("You may need to run setup.py first.")
        sys.exit(1)
    
    # Main menu loop
    while True:
        print("\n--- MAIN MENU ---")
        print("1. Process a list of specific images")
        print("2. Process all images in a folder")
        print("3. Exit")
        
        choice = input("Please select an option (1-3): ").strip()
        
        if choice == '1':
            image_paths = ask_for_images()
            if not image_paths:
                logging.warning("No valid image paths provided.")
                continue
            
            logging.info(f"Starting processing for {len(image_paths)} image(s).")
            for image_path in image_paths:
                try:
                    final_report = orchestrator.run_full_pipeline(image_path)
                    
                    # Display quick summary
                    if final_report and 'analysis_summary' in final_report:
                        summary = final_report['analysis_summary']
                        print(f"\n✓ {image_path.name}: {summary['pass_fail_status']} "
                              f"(Score: {summary['quality_score']}/100, "
                              f"Defects: {summary['total_merged_defects']})")
                    else:
                        print(f"\n✗ {image_path.name}: Processing failed")
                        
                except Exception as e:
                    logging.error(f"Failed to process {image_path}: {e}")
                    continue
            logging.info("Finished processing all specified images.")
            
        elif choice == '2':
            folder_path = ask_for_folder()
            if not folder_path:
                continue
                
            logging.info(f"Searching for images in directory: {folder_path}")
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
            # Remove duplicates
            image_files = list(set(image_files))
            
            if not image_files:
                logging.warning(f"No images with extensions ({', '.join(image_extensions)}) found in {folder_path}")
                continue
            
            logging.info(f"Found {len(image_files)} images to process. Starting batch.")
            
            # Summary statistics
            passed = 0
            failed = 0
            errors = 0
            
            for image_file in sorted(image_files):
                try:
                    final_report = orchestrator.run_full_pipeline(image_file)
                    
                    if final_report and 'analysis_summary' in final_report:
                        summary = final_report['analysis_summary']
                        if summary['pass_fail_status'] == 'PASS':
                            passed += 1
                        else:
                            failed += 1
                            
                        print(f"\n✓ {image_file.name}: {summary['pass_fail_status']} "
                              f"(Score: {summary['quality_score']}/100)")
                    else:
                        errors += 1
                        print(f"\n✗ {image_file.name}: Processing error")
                        
                except Exception as e:
                    logging.error(f"Failed to process {image_file}: {e}")
                    errors += 1
                    continue
                    
            # Print batch summary
            print(f"\n{'='*60}")
            print(f"BATCH PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Total Images: {len(image_files)}")
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")
            print(f"Errors: {errors}")
            print(f"{'='*60}")
            
        elif choice == '3':
            print("\nExiting the application. Goodbye!")
            break
            
        else:
            logging.warning("Invalid choice. Please enter a number from 1 to 3.")

if __name__ == "__main__":
    main()