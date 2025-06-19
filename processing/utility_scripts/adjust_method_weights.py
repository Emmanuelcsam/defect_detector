#!/usr/bin/env python
"""
Tool to manually adjust method scores/weights in the voting system
"""

import json
from pathlib import Path

def load_knowledge(knowledge_file: Path):
    """Load the current knowledge/scores"""
    if knowledge_file.exists():
        with open(knowledge_file, 'r') as f:
            return json.load(f)
    else:
        return {
            'avg_core_radius_ratio': 0.15,
            'avg_cladding_radius_ratio': 0.5,
            'avg_center_offset': 0.02,
            'method_scores': {},
            'method_accuracy': {}
        }

def save_knowledge(knowledge_file: Path, data: dict):
    """Save the updated knowledge"""
    with open(knowledge_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n✓ Saved updated knowledge to {knowledge_file}")

def adjust_scores_interactive():
    """Interactive tool to adjust method scores"""
    knowledge_file = Path("output/segmentation_knowledge.json")
    
    print("Method Score Adjustment Tool")
    print("=" * 50)
    
    # Load current knowledge
    knowledge = load_knowledge(knowledge_file)
    
    # List of all methods
    all_methods = [
        'guess_approach',
        'hough_seperation',
        'segmentation',
        'threshold_seperation',
        'adaptive_intensity_approach',
        'computational_separation',
        'gradient_approach'
    ]
    
    print("\nCurrent Method Scores:")
    print("-" * 30)
    for method in all_methods:
        score = knowledge['method_scores'].get(method, 1.0)
        print(f"{method:30s}: {score:.3f}")
    
    print("\nHigher scores = more influence in voting")
    print("Typical range: 0.1 (low trust) to 2.0 (high trust)")
    print("\nBased on your results:")
    print("- If a method often gives wrong results, lower its score")
    print("- If a method is consistently accurate, raise its score")
    
    while True:
        print("\nOptions:")
        print("1. Adjust a method score")
        print("2. Reset all scores to 1.0")
        print("3. Set scores based on common issues")
        print("4. Save and exit")
        print("5. Exit without saving")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            print("\nMethods:")
            for i, method in enumerate(all_methods, 1):
                print(f"{i}. {method}")
            
            method_idx = input("Select method number: ").strip()
            try:
                idx = int(method_idx) - 1
                if 0 <= idx < len(all_methods):
                    method = all_methods[idx]
                    current_score = knowledge['method_scores'].get(method, 1.0)
                    print(f"\nCurrent score for {method}: {current_score}")
                    
                    new_score = input("Enter new score (0.1-2.0): ").strip()
                    try:
                        score = float(new_score)
                        if 0.1 <= score <= 2.0:
                            knowledge['method_scores'][method] = score
                            print(f"✓ Updated {method} score to {score}")
                        else:
                            print("✗ Score must be between 0.1 and 2.0")
                    except ValueError:
                        print("✗ Invalid number")
            except (ValueError, IndexError):
                print("✗ Invalid selection")
                
        elif choice == '2':
            for method in all_methods:
                knowledge['method_scores'][method] = 1.0
            print("✓ Reset all scores to 1.0")
            
        elif choice == '3':
            print("\nCommon score adjustments:")
            print("1. Computational separation often wrong (coordinates too large)")
            print("2. Threshold separation too simplistic")
            print("3. Trust consensus methods more")
            print("4. Custom based on image type")
            
            preset = input("\nSelect preset (1-4): ").strip()
            
            if preset == '1':
                # Computational separation issues
                knowledge['method_scores']['computational_separation'] = 0.3
                knowledge['method_scores']['segmentation'] = 1.5  # Trust consensus method more
                print("✓ Lowered computational_separation score, raised segmentation score")
                
            elif preset == '2':
                # Simple methods less reliable
                knowledge['method_scores']['threshold_seperation'] = 0.5
                knowledge['method_scores']['adaptive_intensity_approach'] = 0.6
                knowledge['method_scores']['segmentation'] = 1.5
                knowledge['method_scores']['gradient_approach'] = 1.3
                print("✓ Adjusted scores for method complexity")
                
            elif preset == '3':
                # Trust multi-method approaches
                knowledge['method_scores']['segmentation'] = 1.8  # Uses multiple methods internally
                knowledge['method_scores']['gradient_approach'] = 1.5  # Also multi-method
                knowledge['method_scores']['guess_approach'] = 1.2  # Multi-modal
                print("✓ Increased scores for consensus-based methods")
                
            elif preset == '4':
                image_type = input("Image type (clean/noisy/damaged): ").strip().lower()
                
                if image_type == 'clean':
                    # For clean images, geometric methods work well
                    knowledge['method_scores']['hough_seperation'] = 1.5
                    knowledge['method_scores']['computational_separation'] = 1.3
                elif image_type == 'noisy':
                    # For noisy images, robust methods better
                    knowledge['method_scores']['segmentation'] = 1.8
                    knowledge['method_scores']['gradient_approach'] = 1.6
                    knowledge['method_scores']['threshold_seperation'] = 0.3
                elif image_type == 'damaged':
                    # For damaged fibers, adaptive methods
                    knowledge['method_scores']['guess_approach'] = 1.5
                    knowledge['method_scores']['adaptive_intensity_approach'] = 1.3
                    knowledge['method_scores']['hough_seperation'] = 0.5
                    
                print(f"✓ Adjusted scores for {image_type} images")
                
        elif choice == '4':
            save_knowledge(knowledge_file, knowledge)
            print("Changes saved. Exiting.")
            break
            
        elif choice == '5':
            print("Exiting without saving.")
            break
            
        # Show updated scores
        if choice in ['1', '2', '3']:
            print("\nUpdated scores:")
            for method in all_methods:
                score = knowledge['method_scores'].get(method, 1.0)
                print(f"{method:30s}: {score:.3f}")

def main():
    """Main function"""
    print("=" * 60)
    print("Method Weight Adjustment Tool".center(60))
    print("=" * 60)
    print("\nThis tool lets you manually adjust how much each segmentation")
    print("method influences the final pixel voting results.")
    print("\nUse this if you notice certain methods consistently give")
    print("wrong results for your specific type of fiber images.\n")
    
    adjust_scores_interactive()

if __name__ == "__main__":
    main()
