import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

def find_latest_simulation_folder():
    #find the most recent simulation data folder
    folders = [f for f in os.listdir('.') if f.startswith('simulation_data_') and os.path.isdir(f)]
    if not folders:
        raise FileNotFoundError("No simulation data folders found")
    
    #sort by creation time and get the latest
    latest_folder = max(folders, key=lambda f: os.path.getctime(f))
    return latest_folder

def create_graphs(folder_path=None):
    
    if folder_path is None:
        folder_path = find_latest_simulation_folder()
    
    print(f"Loading data from: {folder_path}")
    
    #load CSV files
    try:
        perf_df = pd.read_csv(os.path.join(folder_path, 'performance_log.csv'))
        simulation_info_df = pd.read_csv(os.path.join(folder_path, 'simulation_info.csv'))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    #convert simulation info to dictionary
    sim_info = dict(zip(simulation_info_df['parameter'], simulation_info_df['value']))
    
    #creating figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Simulation Analysis - {folder_path}', fontsize=16, fontweight='bold')
    
    # final simulation image
    img_path = os.path.join(folder_path, 'simulation_result.png')
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Final Simulation Result", fontweight='bold')
        axes[0, 0].axis('off')
    else:
        axes[0, 0].text(0.5, 0.5, 'simulation_result.png not found',
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title("Final Simulation Result", fontweight='bold')
        axes[0, 0].axis('off')
    
    # point creation order
    axes[0, 1].scatter(perf_df['point_x'], perf_df['point_y'], c=perf_df['point_index'],
                        cmap='viridis', s=40, edgecolor='k')
    for _, row in perf_df.iterrows():
        axes[0, 1].annotate(
            str(int(row['point_index'])),
            xy=(row['point_x'], row['point_y']),
            xytext=(5, 0), 
            textcoords='offset points',
            fontsize=10,
            color='black',
            ha='left',
            va='center'
        )
    axes[0, 1].set_title('Point Creation Order', fontweight='bold')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[0, 1], label='Creation Order')
    
    axes[0, 1].set_xlim(0, sim_info['width']) 
    axes[0, 1].set_ylim(0, sim_info['height'])
    axes[0, 1].invert_yaxis()
    
    # exec times for voronoi and delaunay
    axes[1, 0].plot(perf_df['point_index'], perf_df['voronoi_time_ms'], 'o-', label='Voronoi Time (ms)')
    axes[1, 0].plot(perf_df['point_index'], perf_df['delaunay_time_ms'], 's-', label='Delaunay Time (ms)')
    axes[1, 0].set_title('Computation Time per Point', fontweight='bold')
    axes[1, 0].set_xlabel('Point Index (Creation Order)')
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    #  exec time vs number of bad triangles per insertion
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    ax1.plot(perf_df['point_index'], perf_df['delaunay_time_ms'], 'b-', marker='o', label='Delaunay Time (ms)')
    ax2.plot(perf_df['point_index'], perf_df['bad_triangle_count'], 'r--', marker='x', label='Bad Triangles')
    
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('Delaunay Time (ms)', color='b')
    ax2.set_ylabel('Bad Triangles', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Delaunay Time vs Bad Triangles', fontweight='bold')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    create_graphs()