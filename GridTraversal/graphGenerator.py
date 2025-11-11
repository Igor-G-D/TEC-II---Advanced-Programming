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
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return
    
    # Convert simulation info to dictionary
    sim_info = dict(zip(simulation_info_df['parameter'], simulation_info_df['value']))
    
    # Extract palette from simulation info
    palette_str = sim_info.get('pallete', '[]')
    palette = eval(palette_str) if isinstance(palette_str, str) else palette_str
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Simulation Analysis - {folder_path}', fontsize=18, fontweight='bold', y=0.97)
    
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # 1 - simulation image
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
    
    # Prepare indices for x-axis labels
    robot_indices = np.arange(len(performance_log_df))
    
    # 2 - Execution time per path
    ax2 = fig.add_subplot(gs[0, 1])
    colors = [tuple(np.array(c)/255) for c in palette[:len(performance_log_df)]]
    ax2.bar(robot_indices, performance_log_df['astar_exec_time'], color=colors)
    ax2.set_title("A* Execution Time per Robot", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Robot Index")
    ax2.set_ylabel("Execution Time (ms)")
    ax2.set_xticks(robot_indices)
    
    # color legend
    patches = [mpatches.Patch(color=colors[i], label=f'Robot {i}') for i in range(len(colors))]
    ax2.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # 3 - Execution Time vs Heuristic (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    for i, row in performance_log_df.iterrows():
        ax3.scatter(row['euclidian_distance'], row['astar_exec_time'],
                    color=colors[i], s=100, label=f'Robot {i}')
    ax3.set_title("Execution Time vs. Heuristic", fontsize=16, fontweight='bold')
    ax3.set_xlabel("Heuristic Distance")
    ax3.set_ylabel("Execution Time (ms)")
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # 4 - Path Size vs Execution Time (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    for i, row in performance_log_df.iterrows():
        ax4.scatter(row['path_size'], row['astar_exec_time'],
                    color=colors[i], s=100, label=f'Robot {i}')
    ax4.set_title("Path Size vs. Execution Time", fontsize=16, fontweight='bold')
    ax4.set_xlabel("Path Size (# of nodes)")
    ax4.set_ylabel("Execution Time (ms)")
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    handles, labels = ax4.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax4.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    output_filename = f"simulation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Analysis dashboard saved as: {output_filename}")
    
    plt.show()


if __name__ == "__main__":
    create_graphs()