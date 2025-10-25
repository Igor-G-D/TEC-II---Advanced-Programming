import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime
from PIL import Image
import matplotlib.patches as mpatches

def find_latest_simulation_folder():
    # Find the most recent simulation data folder
    folders = [f for f in os.listdir('.') if f.startswith('simulation_data_') and os.path.isdir(f)]
    if not folders:
        raise FileNotFoundError("No simulation data folders found")
    
    # Sort by creation time and get the latest
    latest_folder = max(folders, key=lambda f: os.path.getctime(f))
    return latest_folder

def create_graphs(folder_path=None):
    
    if folder_path is None:
        folder_path = find_latest_simulation_folder()
    
    print(f"Loading data from: {folder_path}")
    
    # Load CSV files
    try:
        performance_log_df = pd.read_csv(os.path.join(folder_path, 'performance_log.csv'))
        simulation_info_df = pd.read_csv(os.path.join(folder_path, 'simulation_info.csv'))
        distance_matrix_df = pd.read_csv(os.path.join(folder_path, 'distance_matrix.csv'))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return
    
    # Convert simulation info to dictionary
    sim_info = dict(zip(simulation_info_df['parameter'], simulation_info_df['value']))
    
    # Extract palette from simulation info
    palette_str = sim_info.get('pallete', '[]')
    
    palette = eval(palette_str) if isinstance(palette_str, str) else palette_str
    
    # robot vertices count
    robot_vertices = int(sim_info.get('robot_points_n', 0))
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Simulation Analysis - {folder_path}', fontsize=16, fontweight='bold', y=0.95)
    
    # Create grid for subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Final simulation image (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    img_path = os.path.join(folder_path, 'simulation_result.png')
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        ax1.imshow(img)
        ax1.set_title("Final Simulation Result", fontsize=16, fontweight='bold')
        ax1.axis('off')
    else:
        ax1.text(0.5, 0.5, 'simulation_result.png not found',
                ha='center', va='center', fontsize=12)
        ax1.set_title("Final Simulation Result", fontsize=14, fontweight='bold')
        ax1.axis('off')
    
    # 2. Robot image (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    robot_img_path = os.path.join(folder_path, 'robot_result.png')
    if os.path.exists(robot_img_path):
        robot_img = plt.imread(robot_img_path)
        ax2.imshow(robot_img)
        ax2.set_title(f"Robot Object\n(Vertices: {robot_vertices})", fontsize=14, fontweight='bold')
        ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, 'robot_result.png not found',
                ha='center', va='center', fontsize=12)
        ax2.set_title(f"Robot Object\n(Vertices: {robot_vertices})", fontsize=14, fontweight='bold')
        ax2.axis('off')
    
    # 3. Polygon vertices vs execution time (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(performance_log_df) > 0 and 'polygon_size' in performance_log_df.columns and 'minkowski_time_ms' in performance_log_df.columns:
        valid_data = performance_log_df.dropna(subset=['polygon_size', 'minkowski_time_ms'])
        
        if len(valid_data) > 0:
            for i, (idx, row) in enumerate(valid_data.iterrows()):
                color = tuple(c/255 for c in palette[i % len(palette)])  # Convert to 0-1 range for matplotlib
                
                ax3.scatter(row['polygon_size'], row['minkowski_time_ms'], 
                            color=color, s=100, alpha=0.7, edgecolors='black', linewidth=1)
                
                # Annotation for each point
                ax3.annotate(f"Poly {i}", (row['polygon_size'], row['minkowski_time_ms']),
                            textcoords="offset points", xytext=(5,5), ha='left', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            
            ax3.set_title(f'Polygon Vertices vs Minkowski Execution Time\n(Robot Vertices: {robot_vertices})', 
                            fontsize=12, fontweight='bold')
            ax3.set_xlabel('Number of Vertices in Polygon', fontsize=12)
            ax3.set_ylabel('Execution Time (ms)', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # statistics
            avg_time = valid_data['minkowski_time_ms'].mean()
            max_time = valid_data['minkowski_time_ms'].max()
            ax3.text(0.02, 0.98, f'Avg time: {avg_time:.2f} ms\nMax time: {max_time:.2f} ms', 
                    transform=ax3.transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'No valid polygon size vs time data', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Polygon Vertices vs Execution Time', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No polygon size vs time data available', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Polygon Vertices vs Execution Time', fontsize=14, fontweight='bold')
    
    # 4. Distance matrix table (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    if not distance_matrix_df.empty:
        # Create colored distance matrix table
        n_polygons = len(distance_matrix_df.columns)
        
        # Create the table data
        table_data = distance_matrix_df.values
        
        # Create row and column labels with colors
        row_labels = [f'Poly {i}' for i in range(n_polygons)]
        col_labels = [f'Poly {i}' for i in range(n_polygons)]
        
        # Create the table
        table = ax4.table(cellText=np.round(table_data, 2),
                        rowLabels=row_labels,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.8])
        
        # Style the table - color the labels
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color the row and column labels
        for i in range(n_polygons):
            if i < len(palette):
                color = tuple(c/255 for c in palette[i])  # Convert to 0-1 range
                # Color row labels
                table[(i+1, -1)].set_facecolor(color)
                table[(i+1, -1)].set_text_props(color='white' if sum(color) < 1.5 else 'black')
                # Color column labels  
                table[(0, i)].set_facecolor(color)
                table[(0, i)].set_text_props(color='white' if sum(color) < 1.5 else 'black')
        
        # Style the data cells
        for i in range(n_polygons):
            for j in range(n_polygons):
                if i == j:  # Diagonal - zero distance to self
                    table[(i+1, j)].set_facecolor('#f0f0f0')
                elif table_data[i, j] == 0:  # Intersecting polygons
                    table[(i+1, j)].set_facecolor('#ffcccc')
                else:  # Normal distance
                    table[(i+1, j)].set_facecolor('#e8f4f8')
        
        ax4.set_title('Distance Matrix Between Polygons (in pixels)\n(Colored by Polygon)', 
                    fontsize=14, fontweight='bold', pad=20)
    
    else:
        ax4.text(0.5, 0.5, 'No distance matrix data available', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Distance Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_filename = f"simulation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Analysis dashboard saved as: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    create_graphs()