import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def analyze_defects_in_region(region_image, region_name="region"):
    """
    Analyze defects in a specific region by finding pixels that deviate from the mode intensity.
    
    Args:
        region_image: The segmented region image (from core.py, cladding.py, or ferrule.py)
        region_name: Name of the region for display purposes
    
    Returns:
        Dictionary containing analysis results and images
    """
    # Convert to grayscale if needed
    if len(region_image.shape) == 3:
        gray_region = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_region = region_image.copy()
    
    # Find non-black pixels (pixels that belong to the region)
    non_black_mask = gray_region > 0
    
    # Extract intensity values of non-black pixels
    non_black_pixels = gray_region[non_black_mask]
    
    if len(non_black_pixels) == 0:
        print(f"No non-black pixels found in {region_name}")
        return None
    
    # Calculate the mode (most common intensity)
    # Using scipy.stats.mode for discrete values
    mode_result = stats.mode(non_black_pixels, keepdims=False)
    mode_intensity = mode_result.mode
    mode_count = mode_result.count
    
    # Calculate statistics
    total_pixels = len(non_black_pixels)
    mode_percentage = (mode_count / total_pixels) * 100
    
    # Create histogram for visualization
    hist, bins = np.histogram(non_black_pixels, bins=256, range=(0, 256))
    
    # Find pixels that are different from the mode (defects)
    defect_mask = (gray_region != mode_intensity) & non_black_mask
    defect_count = np.sum(defect_mask)
    defect_percentage = (defect_count / total_pixels) * 100
    
    # Create visualization image
    if len(region_image.shape) == 3:
        visualization = region_image.copy()
    else:
        # Convert grayscale to BGR for colored visualization
        visualization = cv2.cvtColor(region_image, cv2.COLOR_GRAY2BGR)
    
    # Color the defects blue
    visualization[defect_mask] = [255, 0, 0]  # Blue color for defects
    
    # Create a defect-only mask for better visualization
    defect_only = np.zeros_like(visualization)
    defect_only[defect_mask] = [255, 0, 0]
    
    # Get unique intensities and their counts for detailed analysis
    unique_intensities, counts = np.unique(non_black_pixels, return_counts=True)
    intensity_distribution = list(zip(unique_intensities, counts))
    intensity_distribution.sort(key=lambda x: x[1], reverse=True)  # Sort by count
    
    # Print analysis results
    print(f"\n=== DEFECT ANALYSIS FOR {region_name.upper()} ===")
    print(f"Total pixels in region: {total_pixels}")
    print(f"Mode intensity: {mode_intensity}")
    print(f"Pixels with mode intensity: {mode_count} ({mode_percentage:.2f}%)")
    print(f"Defect pixels (non-mode): {defect_count} ({defect_percentage:.2f}%)")
    print(f"\nTop 5 intensity values:")
    for i, (intensity, count) in enumerate(intensity_distribution[:5]):
        percentage = (count / total_pixels) * 100
        print(f"  {i+1}. Intensity {intensity}: {count} pixels ({percentage:.2f}%)")
    
    # Calculate additional defect statistics
    if defect_count > 0:
        defect_intensities = gray_region[defect_mask]
        print(f"\nDefect intensity statistics:")
        print(f"  - Mean: {np.mean(defect_intensities):.2f}")
        print(f"  - Std: {np.std(defect_intensities):.2f}")
        print(f"  - Min: {np.min(defect_intensities)}")
        print(f"  - Max: {np.max(defect_intensities)}")
    
    return {
        'original': region_image,
        'visualization': visualization,
        'defect_only': defect_only,
        'defect_mask': defect_mask,
        'mode_intensity': mode_intensity,
        'defect_count': defect_count,
        'defect_percentage': defect_percentage,
        'histogram': (hist, bins),
        'intensity_distribution': intensity_distribution
    }

def create_analysis_report(results, region_name):
    """Create a comprehensive visual report of the defect analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Defect Analysis Report - {region_name}', fontsize=16)
    
    # Original region
    ax1 = axes[0, 0]
    if len(results['original'].shape) == 3:
        ax1.imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(results['original'], cmap='gray')
    ax1.set_title('Original Region')
    ax1.axis('off')
    
    # Defect visualization
    ax2 = axes[0, 1]
    ax2.imshow(cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Defects Highlighted (Blue)\n{results["defect_percentage"]:.2f}% defects')
    ax2.axis('off')
    
    # Defect mask only
    ax3 = axes[1, 0]
    ax3.imshow(cv2.cvtColor(results['defect_only'], cv2.COLOR_BGR2RGB))
    ax3.set_title('Defect Locations Only')
    ax3.axis('off')
    
    # Intensity histogram
    ax4 = axes[1, 1]
    hist, bins = results['histogram']
    ax4.bar(bins[:-1], hist, width=1, edgecolor='none')
    ax4.axvline(x=results['mode_intensity'], color='red', linestyle='--', 
                label=f'Mode: {results["mode_intensity"]}')
    ax4.set_xlabel('Intensity Value')
    ax4.set_ylabel('Pixel Count')
    ax4.set_title('Intensity Distribution')
    ax4.legend()
    ax4.set_xlim(0, 255)
    
    plt.tight_layout()
    return fig

def main():
    # This example assumes you've already run core.py to generate core.png
    # You can modify the path to analyze ferrule.png or cladding.png instead
    
    # Load the core region image
    core_image_path = "core.png"
    core_image = cv2.imread(core_image_path)
    
    if core_image is None:
        print(f"Error: Could not load image from {core_image_path}")
        print("Please run core.py first to generate the core region image.")
        return
    
    # Analyze defects in the core region
    results = analyze_defects_in_region(core_image, "core")
    
    if results is None:
        return
    
    # Save visualization results
    cv2.imwrite("core_defects_highlighted.png", results['visualization'])
    cv2.imwrite("core_defects_only.png", results['defect_only'])
    cv2.imwrite("core_defect_mask.png", results['defect_mask'].astype(np.uint8) * 255)
    
    print("\nDefect analysis images saved:")
    print("- core_defects_highlighted.png (original with defects in blue)")
    print("- core_defects_only.png (only defect locations)")
    print("- core_defect_mask.png (binary defect mask)")
    
    # Create and save analysis report
    fig = create_analysis_report(results, "Core")
    fig.savefig("core_defect_analysis_report.png", dpi=300, bbox_inches='tight')
    print("- core_defect_analysis_report.png (comprehensive report)")
    
    # Save detailed intensity distribution to text file
    with open("core_defect_analysis.txt", "w") as f:
        f.write(f"DEFECT ANALYSIS REPORT - CORE REGION\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Mode Intensity: {results['mode_intensity']}\n")
        f.write(f"Total Pixels: {np.sum(results['defect_mask'] | (results['original'][:,:,0] > 0) if len(results['original'].shape) == 3 else results['original'] > 0)}\n")
        f.write(f"Defect Pixels: {results['defect_count']}\n")
        f.write(f"Defect Percentage: {results['defect_percentage']:.2f}%\n\n")
        f.write(f"Intensity Distribution (sorted by frequency):\n")
        f.write(f"{'Intensity':<12}{'Count':<12}{'Percentage':<12}\n")
        f.write(f"{'-'*36}\n")
        for intensity, count in results['intensity_distribution'][:20]:  # Top 20
            percentage = (count / np.sum([c for _, c in results['intensity_distribution']])) * 100
            f.write(f"{intensity:<12}{count:<12}{percentage:<12.2f}\n")
    
    print("- core_defect_analysis.txt (detailed statistics)")
    
    return results

# Function to analyze any of the three regions
def analyze_region(region_type="core"):
    """
    Analyze defects in a specific region type.
    
    Args:
        region_type: "core", "cladding", or "ferrule"
    """
    # Map region types to their output files
    region_files = {
        "core": "core.png",
        "cladding": "cladding.png",
        "ferrule": "ferrule.png"
    }
    
    if region_type not in region_files:
        print(f"Invalid region type. Choose from: {list(region_files.keys())}")
        return
    
    # Load the region image
    image_path = region_files[region_type]
    region_image = cv2.imread(image_path)
    
    if region_image is None:
        print(f"Error: Could not load {image_path}")
        print(f"Please run {region_type}.py first to generate the region image.")
        return
    
    # Analyze defects
    results = analyze_defects_in_region(region_image, region_type)
    
    if results is None:
        return
    
    # Save results with region-specific names
    cv2.imwrite(f"{region_type}_defects_highlighted.png", results['visualization'])
    cv2.imwrite(f"{region_type}_defects_only.png", results['defect_only'])
    cv2.imwrite(f"{region_type}_defect_mask.png", results['defect_mask'].astype(np.uint8) * 255)
    
    # Create and save analysis report
    fig = create_analysis_report(results, region_type.capitalize())
    fig.savefig(f"{region_type}_defect_analysis_report.png", dpi=300, bbox_inches='tight')
    
    print(f"\nAll results saved with prefix '{region_type}_'")
    
    return results

if __name__ == "__main__":
    # Analyze the core region by default
    main()
    
    # To analyze other regions, uncomment one of these:
    # analyze_region("cladding")
    # analyze_region("ferrule")