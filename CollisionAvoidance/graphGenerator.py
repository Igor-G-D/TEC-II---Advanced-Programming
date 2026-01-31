import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import ast
from datetime import datetime
from matplotlib import ticker

def find_latest_simulation_folder():
    """Finds the most recent simulation data folder."""
    folders = [f for f in os.listdir('.') if f.startswith('simulation_data_') and os.path.isdir(f)]
    if not folders:
        return None
    latest_folder = max(folders, key=lambda f: os.path.getctime(f))
    return latest_folder

def create_graphs(folder_path=None):
    if folder_path is None:
        folder_path = find_latest_simulation_folder()
    
    if not folder_path:
        print("No simulation data folder found.")
        return
    
    print(f"Analyzing data from: {folder_path}")
    
    try:
        performance_df = pd.read_csv(os.path.join(folder_path, 'performance_log.csv'))
        sim_info_df = pd.read_csv(os.path.join(folder_path, 'simulation_info.csv'))
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    sim_info = dict(zip(sim_info_df['parameter'], sim_info_df['value']))
    palette_raw = ast.literal_eval(sim_info.get('palette', '[]'))
    colors = [tuple(np.array(c)/255) for c in palette_raw[:len(performance_df)]]

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f'Robot Performance & Movement Analysis\n{folder_path}', fontsize=22, fontweight='bold')

    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1])

    # fnal image exported
    ax1 = fig.add_subplot(gs[0, 0])
    img_path = os.path.join(folder_path, 'simulation_result.png')
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        ax1.imshow(img)
        ax1.set_title("Final Simulation State", fontsize=16, pad=10)
        ax1.axis('off')

    #individual overhead time
    ax2 = fig.add_subplot(gs[1, 0])
    for i, row in performance_df.iterrows():
        overhead_list = ast.literal_eval(row['individual_overhead'])
        overhead_ms = [t * 1000 for t in overhead_list]
        ax2.plot(range(len(overhead_ms)), overhead_ms, label=f'Robot {row["robot_id"]}', 
                    color=colors[i], marker='o', markersize=5, alpha=0.8, linewidth=2)

    ax2.set_title("Computational Overhead per Robot", fontsize=16)
    ax2.set_ylabel("Time (ms)")
    ax2.set_xlabel("Simulation Step")
    
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # movement progress timeline
    ax3 = fig.add_subplot(gs[2, 0])
    stopped_color = "#8D8D8D"
    
    max_steps = 0
    for i, row in performance_df.iterrows():
        history = ast.literal_eval(row['movement_history'])
        max_steps = max(max_steps, len(history))
        for step, moved in enumerate(history):
            color = colors[i] if moved else stopped_color
            alpha = 1.0 if moved else 0.8 
            
            ax3.hlines(y=i, xmin=step, xmax=step+1, 
                        color=color, linewidth=22, alpha=alpha)
            
    ax3.set_yticks(range(len(performance_df)))
    ax3.set_yticklabels([f'Robot {int(row["robot_id"])}' for _, row in performance_df.iterrows()])
    ax3.set_title("Movement Timeline: Advanced (Colored) vs. Stopped (Gray)", fontsize=16)
    ax3.set_xlabel("Simulation Step")
    
    # Force X-Axis to use Integers and fix range
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.set_xlim(0, max_steps)
    ax3.grid(axis='x', linestyle='--', alpha=0.7)

    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', lw=8),
                    Line2D([0], [0], color=stopped_color, lw=8, alpha=0.8)]
    ax3.legend(custom_lines, ['Advanced', 'Stopped/Waiting'], loc='upper left', bbox_to_anchor=(1, 1))

    # Increased hspace (0.6) to ensure titles/labels don't overlap
    plt.subplots_adjust(top=0.90, hspace=0.6, bottom=0.08, left=0.08, right=0.85)
    
    output_filename = f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Analysis saved as: {output_filename}")
    plt.show()

if __name__ == "__main__":
    create_graphs()