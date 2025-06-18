import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_edges(img):
    """Extract edge points using Canny edge detection"""
    edges = cv2.Canny(img, 30, 100)  # Simple thresholds
    return np.column_stack(np.where(edges)[::-1])  # Return as (x, y)

def find_circles_ransac(pts, n_iter=1000):
    """RANSAC to find two concentric circles"""
    best_score, best_params = -1, None
    img_bounds = pts.max(axis=0)  # Get image dimensions
    
    for _ in range(n_iter):
        # Sample 3 points to find circle center
        idx = np.random.choice(len(pts), 3, replace=False)
        p1, p2, p3 = pts[idx]
        
        # Calculate circumcenter
        d = 2 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
        if abs(d) < 1e-6: continue
        
        cx = ((p1[0]**2+p1[1]**2)*(p2[1]-p3[1]) + (p2[0]**2+p2[1]**2)*(p3[1]-p1[1]) + 
              (p3[0]**2+p3[1]**2)*(p1[1]-p2[1])) / d
        cy = ((p1[0]**2+p1[1]**2)*(p3[0]-p2[0]) + (p2[0]**2+p2[1]**2)*(p1[0]-p3[0]) + 
              (p3[0]**2+p3[1]**2)*(p2[0]-p1[0])) / d
        
        # Skip if center is far outside image bounds
        if cx < -img_bounds[0] or cx > 2*img_bounds[0] or cy < -img_bounds[1] or cy > 2*img_bounds[1]:
            continue
        
        # Score by radial histogram
        dists = np.linalg.norm(pts - [cx, cy], axis=1)
        max_r = min(dists.max(), max(img_bounds))  # Limit max radius
        hist, bins = np.histogram(dists, bins=50, range=(0, max_r))
        peaks = np.argsort(hist)[-2:]  # Two highest bins
        score = hist[peaks].sum()
        
        if score > best_score:
            best_score = score
            r1, r2 = bins[peaks] + (bins[1]-bins[0])/2
            best_params = [cx, cy, min(r1, r2), max(r1, r2)]
    
    return best_params

def refine_circles(pts, params, n_iter=50):
    """Simple iterative refinement of circle parameters"""
    cx, cy, r1, r2 = params
    img_bounds = pts.max(axis=0)  # Get image dimensions
    
    for _ in range(n_iter):
        # Compute distances to center
        dists = np.linalg.norm(pts - [cx, cy], axis=1)
        
        # Assign points to nearest circle
        mask1 = np.abs(dists - r1) < np.abs(dists - r2)
        pts1, pts2 = pts[mask1], pts[~mask1]
        
        # Update radii as mean distances (with bounds)
        if len(pts1) > 0: 
            r1 = np.mean(np.linalg.norm(pts1 - [cx, cy], axis=1))
            r1 = min(r1, max(img_bounds))  # Limit radius to image size
        if len(pts2) > 0: 
            r2 = np.mean(np.linalg.norm(pts2 - [cx, cy], axis=1))
            r2 = min(r2, max(img_bounds))  # Limit radius to image size
        
        # Update center as weighted mean
        if len(pts1) + len(pts2) > 0:
            all_pts = np.vstack([pts1, pts2]) if len(pts1) > 0 and len(pts2) > 0 else pts1 if len(pts1) > 0 else pts2
            cx, cy = np.mean(all_pts, axis=0)
    
    return [cx, cy, min(r1, r2), max(r1, r2)]

def create_masks(shape, params):
    """Create core and cladding masks"""
    h, w = shape
    cx, cy, r_core, r_clad = params
    
    # Distance matrix
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx)**2 + (y - cy)**2
    
    # Create masks
    core_mask = (dist_sq <= r_core**2).astype(np.uint8) * 255
    clad_mask = ((dist_sq > r_core**2) & (dist_sq <= r_clad**2)).astype(np.uint8) * 255
    
    return core_mask, clad_mask

def process_fiber(img_path, out_dir='output'):
    """Main processing pipeline"""
    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Load and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract edges
    edges = get_edges(gray)
    print(f"Found {len(edges)} edge points")
    
    # Find circles with RANSAC
    params = find_circles_ransac(edges)
    print(f"Initial: center({params[0]:.1f},{params[1]:.1f}), radii({params[2]:.1f},{params[3]:.1f})")
    
    # Refine parameters
    params = refine_circles(edges, params)
    cx, cy, r1, r2 = params
    print(f"Refined: center({cx:.1f},{cy:.1f}), radii({r1:.1f},{r2:.1f})")
    
    # Create masks and isolate regions
    core_mask, clad_mask = create_masks(gray.shape, params)
    core = cv2.bitwise_and(gray, gray, mask=core_mask)
    clad = cv2.bitwise_and(gray, gray, mask=clad_mask)
    
    # Crop to content
    y, x = np.where(core_mask)
    if len(y) > 0: core = core[y.min():y.max()+1, x.min():x.max()+1]
    y, x = np.where(clad_mask)
    if len(y) > 0: clad = clad[y.min():y.max()+1, x.min():x.max()+1]
    
    # Visualize results
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.gca().add_patch(plt.Circle((cx, cy), r1, color='lime', fill=False, linewidth=2))
    plt.gca().add_patch(plt.Circle((cx, cy), r2, color='cyan', fill=False, linewidth=2))
    plt.scatter(edges[:, 0], edges[:, 1], s=1, c='red', alpha=0.3)
    plt.title('Fiber Circle Detection')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/result.png', dpi=150)
    plt.close()
    
    # Save outputs
    cv2.imwrite(f'{out_dir}/core.png', core)
    cv2.imwrite(f'{out_dir}/cladding.png', clad)
    print(f"Results saved to {out_dir}/")

# Example usage
if __name__ == '__main__':
    process_fiber(r'C:\Users\Saem1001\Documents\GitHub\IPPS\App\images\img63.jpg')