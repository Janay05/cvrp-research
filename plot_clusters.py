import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def main():
    csv_file = "clusters_output.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run test_visual.exe first.")
        return

    print("Loading data...")
    df = pd.read_csv(csv_file)
    
    # 8 distinct color maps for the 8 macro wedges.
    colormaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys', 'YlOrBr', 'PuBuGn']
    
    plt.figure(figsize=(16, 16), facecolor='whitesmoke')
    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    
    macros = sorted(df['macro_id'].unique())
    
    print("Plotting points and boundaries...")
    for m_idx, macro in enumerate(macros):
        macro_df = df[df['macro_id'] == macro]
        micros = sorted(macro_df['micro_id'].unique())
        
        # Choose a colormap for this macro wedge
        cmap_name = colormaps[m_idx % len(colormaps)]
        cmap = plt.get_cmap(cmap_name)
        
        for i, micro in enumerate(micros):
            micro_df = macro_df[macro_df['micro_id'] == micro]
            
            # Select a shade from the colormap. We avoid the extremely light ends.
            color_intensity = 0.5 + (0.5 * (i % 5) / 4.0) # Varies from 0.5 to 1.0
            color = cmap(color_intensity)
            
            # 1. Plot the points
            plt.scatter(
                micro_df['x'], 
                micro_df['y'], 
                s=8, # larger dot size for visibility
                color=color,
                alpha=0.85,
                edgecolors='none'
            )
            
            # 2. Draw convex hull (border) to make the partition distinctly visible
            if HAS_SCIPY and len(micro_df) >= 3:
                points = micro_df[['x', 'y']].values
                try:
                    hull = ConvexHull(points)
                    # Close the polygon by appending the first point at the end
                    hull_points = np.append(points[hull.vertices], [points[hull.vertices[0]]], axis=0)
                    plt.plot(hull_points[:, 0], hull_points[:, 1], color='black', linewidth=1.2, alpha=0.7)
                except Exception:
                    pass # Ignore collinear points or hull errors

    # Plot depot at 0,0
    plt.scatter([0], [0], color='black', marker='o', s=400, label='Depot (Node 0)', zorder=10, edgecolors='white', linewidths=1.5)

    plt.title(f"CVRP 3-Stage Partitioning: Micro-Clusters ({len(df)} nodes)", fontsize=24, fontweight='bold', pad=20)
    plt.xlabel("X Coordinate (Euclidean)", fontsize=16)
    plt.ylabel("Y Coordinate (Euclidean)", fontsize=16)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.4, color='gray')
    
    # Hide axis spines for a cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    print("Saving plot to clusters_plot_professional.png...")
    plt.savefig("clusters_plot_professional.png", dpi=400, bbox_inches='tight')
    print("Done! You can open clusters_plot_professional.png.")
    if not HAS_SCIPY:
        print("Note: Install 'scipy' (pip install scipy) to draw black boundaries around the petals.")

if __name__ == "__main__":
    main()
