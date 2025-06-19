#!/usr/bin/env python
"""
Demonstrates how the pixel-by-pixel voting process works
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_example_masks(image_shape=(400, 400)):
    """Create example masks from different methods to show voting"""
    h, w = image_shape
    
    # Method 1: Slightly off-center, smaller radii
    method1_masks = {}
    center1 = (w//2 - 10, h//2 - 5)
    y_grid, x_grid = np.ogrid[:h, :w]
    dist1 = np.sqrt((x_grid - center1[0])**2 + (y_grid - center1[1])**2)
    
    method1_masks['core'] = (dist1 <= 40).astype(np.uint8)
    method1_masks['cladding'] = ((dist1 > 40) & (dist1 <= 90)).astype(np.uint8)
    method1_masks['ferrule'] = (dist1 > 90).astype(np.uint8)
    
    # Method 2: Centered, medium radii
    method2_masks = {}
    center2 = (w//2, h//2)
    dist2 = np.sqrt((x_grid - center2[0])**2 + (y_grid - center2[1])**2)
    
    method2_masks['core'] = (dist2 <= 45).astype(np.uint8)
    method2_masks['cladding'] = ((dist2 > 45) & (dist2 <= 100)).astype(np.uint8)
    method2_masks['ferrule'] = (dist2 > 100).astype(np.uint8)
    
    # Method 3: Slightly off-center other way, larger radii
    method3_masks = {}
    center3 = (w//2 + 8, h//2 + 6)
    dist3 = np.sqrt((x_grid - center3[0])**2 + (y_grid - center3[1])**2)
    
    method3_masks['core'] = (dist3 <= 50).astype(np.uint8)
    method3_masks['cladding'] = ((dist3 > 50) & (dist3 <= 110)).astype(np.uint8)
    method3_masks['ferrule'] = (dist3 > 110).astype(np.uint8)
    
    return [
        ('Method1', method1_masks, 0.7),  # confidence
        ('Method2', method2_masks, 0.9),  # higher confidence
        ('Method3', method3_masks, 0.8)
    ]

def perform_pixel_voting(methods_data):
    """Simulate the pixel voting process"""
    # Get image shape from first method
    h, w = methods_data[0][1]['core'].shape
    
    # Initialize vote accumulation
    core_votes = np.zeros((h, w), dtype=np.float32)
    cladding_votes = np.zeros((h, w), dtype=np.float32)
    ferrule_votes = np.zeros((h, w), dtype=np.float32)
    
    # Accumulate weighted votes
    for method_name, masks, confidence in methods_data:
        core_votes += masks['core'] * confidence
        cladding_votes += masks['cladding'] * confidence
        ferrule_votes += masks['ferrule'] * confidence
    
    # Create final masks - each pixel goes to region with most votes
    final_masks = {}
    max_votes = np.maximum(core_votes, np.maximum(cladding_votes, ferrule_votes))
    
    final_masks['core'] = (core_votes == max_votes).astype(np.uint8)
    final_masks['cladding'] = (cladding_votes == max_votes).astype(np.uint8)
    final_masks['ferrule'] = (ferrule_votes == max_votes).astype(np.uint8)
    
    # Handle ties by preferring core > cladding > ferrule
    tie_pixels = (final_masks['core'] + final_masks['cladding'] + final_masks['ferrule']) > 1
    if np.any(tie_pixels):
        # Reset tie pixels
        final_masks['core'][tie_pixels] = 0
        final_masks['cladding'][tie_pixels] = 0
        final_masks['ferrule'][tie_pixels] = 0
        
        # Assign based on priority
        tie_coords = np.where(tie_pixels)
        for i in range(len(tie_coords[0])):
            y, x = tie_coords[0][i], tie_coords[1][i]
            if core_votes[y, x] == max_votes[y, x]:
                final_masks['core'][y, x] = 1
            elif cladding_votes[y, x] == max_votes[y, x]:
                final_masks['cladding'][y, x] = 1
            else:
                final_masks['ferrule'][y, x] = 1
    
    return final_masks, (core_votes, cladding_votes, ferrule_votes)

def visualize_voting_process():
    """Create visualization of the voting process"""
    # Create example masks
    methods_data = create_example_masks()
    
    # Perform voting
    final_masks, vote_arrays = perform_pixel_voting(methods_data)
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # Show individual method masks
    for i, (method_name, masks, confidence) in enumerate(methods_data):
        # Create color visualization
        h, w = masks['core'].shape
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        viz[masks['core'] > 0] = [255, 0, 0]      # Red for core
        viz[masks['cladding'] > 0] = [0, 255, 0]  # Green for cladding
        viz[masks['ferrule'] > 0] = [0, 0, 255]   # Blue for ferrule
        
        ax = plt.subplot(3, 4, i+1)
        ax.imshow(viz)
        ax.set_title(f'{method_name} (conf={confidence})')
        ax.axis('off')
    
    # Show vote accumulation
    vote_names = ['Core Votes', 'Cladding Votes', 'Ferrule Votes']
    for i, (votes, name) in enumerate(zip(vote_arrays, vote_names)):
        ax = plt.subplot(3, 4, 5+i)
        im = ax.imshow(votes, cmap='hot')
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Show final result
    h, w = final_masks['core'].shape
    final_viz = np.zeros((h, w, 3), dtype=np.uint8)
    final_viz[final_masks['core'] > 0] = [255, 0, 0]
    final_viz[final_masks['cladding'] > 0] = [0, 255, 0]
    final_viz[final_masks['ferrule'] > 0] = [0, 0, 255]
    
    ax = plt.subplot(3, 4, 9)
    ax.imshow(final_viz)
    ax.set_title('Final Result (Pixel Voting)')
    ax.axis('off')
    
    # Show difference from each method
    for i, (method_name, masks, _) in enumerate(methods_data):
        diff_viz = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Show where final differs from this method
        diff_core = final_masks['core'] != masks['core']
        diff_clad = final_masks['cladding'] != masks['cladding']
        diff_ferrule = final_masks['ferrule'] != masks['ferrule']
        
        diff_any = diff_core | diff_clad | diff_ferrule
        diff_viz[diff_any] = [255, 255, 0]  # Yellow for differences
        
        ax = plt.subplot(3, 4, 10+i)
        ax.imshow(diff_viz)
        ax.set_title(f'Diff from {method_name}')
        ax.axis('off')
    
    plt.suptitle('Pixel-by-Pixel Voting Process Demonstration', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nPixel Voting Statistics:")
    print("=" * 50)
    total_pixels = h * w
    core_pixels = np.sum(final_masks['core'])
    cladding_pixels = np.sum(final_masks['cladding'])
    ferrule_pixels = np.sum(final_masks['ferrule'])
    
    print(f"Total pixels: {total_pixels}")
    print(f"Core: {core_pixels} pixels ({100*core_pixels/total_pixels:.1f}%)")
    print(f"Cladding: {cladding_pixels} pixels ({100*cladding_pixels/total_pixels:.1f}%)")
    print(f"Ferrule: {ferrule_pixels} pixels ({100*ferrule_pixels/total_pixels:.1f}%)")
    
    # Show how many pixels each method contributed to the final result
    print("\nMethod contributions to final result:")
    for method_name, masks, confidence in methods_data:
        core_match = np.sum(final_masks['core'] & masks['core'])
        clad_match = np.sum(final_masks['cladding'] & masks['cladding'])
        ferrule_match = np.sum(final_masks['ferrule'] & masks['ferrule'])
        total_match = core_match + clad_match + ferrule_match
        
        print(f"{method_name}: {total_match} pixels ({100*total_match/total_pixels:.1f}%) matched final result")

def main():
    """Main function"""
    print("Pixel-by-Pixel Voting Process Visualization")
    print("=" * 50)
    print("\nThis demonstrates how the voting system works:")
    print("1. Each method produces masks for core, cladding, and ferrule")
    print("2. Votes are weighted by method confidence")
    print("3. Each pixel is assigned to the region with the most votes")
    print("4. The final result is a consensus of all methods\n")
    
    visualize_voting_process()

if __name__ == "__main__":
    main()
