import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime
from PIL import Image

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
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return
    
    # Convert simulation info to dictionary
    sim_info = dict(zip(simulation_info_df['parameter'], simulation_info_df['value']))
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Convex Hull Performance Analysis - {folder_path}', fontsize=16, fontweight='bold')
    
    # final simulation image
    img_path = os.path.join(folder_path, 'simulation_result.png')
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        axes[0].imshow(img)
        axes[0].set_title("Final Simulation Result", fontweight='bold')
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'simulation_result.png not found',
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0].set_title("Final Simulation Result", fontweight='bold')
        axes[0].axis('off')
    
    # Points composition over time
    if len(performance_log_df) > 0 and 'total_points' in performance_log_df.columns:
        # Remove rows with missing data
        valid_data = performance_log_df.dropna(subset=['total_points', 'convexHull_points'])
        
        if len(valid_data) > 0:
            total_points = valid_data['total_points']
            convex_hull_points = valid_data['convexHull_points']
            non_convex_hull_points = total_points - convex_hull_points
            
            x_pos = np.arange(len(total_points))
            
            axes[1].plot(x_pos, convex_hull_points, marker='s', linewidth=2, markersize=6, 
                        label='Convex Hull Points', color='red')
            axes[1].plot(x_pos, non_convex_hull_points, marker='^', linewidth=2, markersize=6, 
                        label='Non-Convex Hull Points', color='green')
            
            for i, (total, hull, non_hull) in enumerate(zip(total_points, convex_hull_points, non_convex_hull_points)):
                axes[1].annotate(f'{int(hull)}', (x_pos[i], hull), 
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                
                axes[1].annotate(f'{int(non_hull)}', (x_pos[i], non_hull), 
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
            axes[1].set_title('Points Composition Over Steps', fontweight='bold')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Number of Points')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            axes[1].set_xticks(x_pos)
            
            avg_hull_ratio = (convex_hull_points / total_points).mean()
            final_hull_ratio = convex_hull_points.iloc[-1] / total_points.iloc[-1]
            
            axes[1].text(0.02, 0.98, f'Avg hull ratio: {avg_hull_ratio:.3f}\nFinal ratio: {final_hull_ratio:.3f}', 
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[1].text(0.5, 0.5, 'No valid points composition data', 
                        ha='center', va='center', transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, 'No performance log data', 
                    ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Points Composition Over Steps', fontweight='bold')
    
    # execution time vs total points
    if len(performance_log_df) > 0 and 'convexhull_time_ms' in performance_log_df.columns:
        time_data = performance_log_df.dropna(subset=['convexhull_time_ms', 'total_points'])
        
        if len(time_data) > 0:
            axes[2].plot(time_data['total_points'], time_data['convexhull_time_ms'], 
                        marker='o', linewidth=2, markersize=6, alpha=0.7, label='Execution Time')
            
            axes[2].set_title('Convex Hull Execution Time', fontweight='bold')
            axes[2].set_xlabel('Total Points')
            axes[2].set_ylabel('Execution Time (ms)')
            axes[2].grid(True, alpha=0.3)
            
            max_time = time_data['convexhull_time_ms'].max()
            max_points = time_data['total_points'].max()
            min_time = time_data['convexhull_time_ms'].min()
            
            axes[2].text(0.02, 0.98, f'Max: {max_time:.2f} ms\nMin: {min_time:.2f} ms\nAt {max_points} points', 
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[2].text(0.5, 0.5, 'No valid execution time data', 
                        ha='center', va='center', transform=axes[2].transAxes)
    else:
        axes[2].text(0.5, 0.5, 'No execution time data', 
                    ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('Convex Hull Execution Time', fontweight='bold')
    
    plt.tight_layout()
    
    output_filename = f"convexhull_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graphs saved as: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    create_graphs()