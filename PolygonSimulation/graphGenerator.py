import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime

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
        mouse_move_df = pd.read_csv(os.path.join(folder_path, 'mouse_movements.csv'))
        mouse_click_df = pd.read_csv(os.path.join(folder_path, 'mouse_clicks.csv'))
        objects_clicked_df = pd.read_csv(os.path.join(folder_path, 'objects_clicked.csv'))
        simulation_info_df = pd.read_csv(os.path.join(folder_path, 'simulation_info.csv'))
        points_info_df = pd.read_csv(os.path.join(folder_path, 'points_info.csv'))
        lines_info_df = pd.read_csv(os.path.join(folder_path, 'lines_info.csv'))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return
    
    #convert simulation info to dictionary
    sim_info = dict(zip(simulation_info_df['parameter'], simulation_info_df['value']))
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    #creating figure with subploots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Simulation Analysis - {folder_path}', fontsize=16, fontweight='bold')
    
    #heatmap of mouse movements
    if len(mouse_move_df) > 0:
        heatmap, xedges, yedges = np.histogram2d(
            mouse_move_df['x'], 
            mouse_move_df['y'], 
            bins=80,
            range=[[0, int(sim_info['width'])], [0, int(sim_info['height'])]]
        )
        
        im = axes[0, 0].imshow(heatmap.T, origin='upper', aspect='auto', 
                                extent=[0, int(sim_info['width']), 0, int(sim_info['height'])],
                                cmap='hot', alpha=0.8)
        axes[0, 0].set_title('Mouse Movement Heatmap', fontweight='bold')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        plt.colorbar(im, ax=axes[0, 0], label='Movement Density')
        
        total_movements = len(mouse_move_df)
        axes[0, 0].text(0.02, 0.98, f'Total movements: {total_movements:,}', 
                        transform=axes[0, 0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[0, 0].text(0.5, 0.5, 'No mouse movement data', 
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Mouse Movement Heatmap', fontweight='bold')
    
    #click heatmap
    if len(mouse_click_df) > 0:
        
        click_heatmap, xedges, yedges = np.histogram2d(
            mouse_click_df['x'], 
            mouse_click_df['y'], 
            bins=30,
            range=[[0, int(sim_info['width'])], [0, int(sim_info['height'])]]
        )
        
        im2 = axes[0, 1].imshow(click_heatmap.T, origin='upper', aspect='auto', 
                                extent=[0, int(sim_info['width']), 0, int(sim_info['height'])],
                                cmap='viridis', alpha=0.8)
        axes[0, 1].set_title('Mouse Click Heatmap', fontweight='bold')
        axes[0, 1].set_xlabel('X Position')
        axes[0, 1].set_ylabel('Y Position')
        plt.colorbar(im2, ax=axes[0, 1], label='Click Density')
        
        total_clicks = len(mouse_click_df)
        object_clicks = len(objects_clicked_df)
        axes[0, 1].text(0.02, 0.98, f'Total clicks: {total_clicks}\nObject clicks: {object_clicks}', 
                        transform=axes[0, 1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[0, 1].text(0.5, 0.5, 'No click data', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Mouse Click Heatmap', fontweight='bold')
    
    #time and objects clicked
    if len(objects_clicked_df) > 0:
        #converting timestamp to instead be relative to start time
        start_time = sim_info['timestamp_start']
        objects_clicked_df['relative_time'] = objects_clicked_df['timestamp'] - start_time
        
        scatter = axes[1, 0].scatter(
            objects_clicked_df['object_index'], 
            objects_clicked_df['relative_time'],
            c=objects_clicked_df['object_type'],
            cmap='Set1',
            alpha=0.7,
            s=60
        )
        
        axes[1, 0].set_title('Object Clicks Over Time', fontweight='bold')
        axes[1, 0].set_xlabel('Object Index')
        axes[1, 0].set_ylabel('Time (seconds from start)')
        
        #create a legend for object types
        unique_types = objects_clicked_df['object_type_name'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_types)))
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=colors[i], markersize=8, 
                                    label=unique_types[i]) 
                                    for i in range(len(unique_types))]
        axes[1, 0].legend(handles=legend_elements, title='Object Type')
        
        total_object_clicks = len(objects_clicked_df)
        point_clicks = len(objects_clicked_df[objects_clicked_df['object_type'] == 0])
        line_clicks = len(objects_clicked_df[objects_clicked_df['object_type'] == 1])
        
        axes[1, 0].text(0.02, 0.98, f'Total object clicks: {total_object_clicks}\n'
                                    f'Point clicks: {point_clicks}\n'
                                    f'Line clicks: {line_clicks}', 
                                    transform=axes[1, 0].transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1, 0].text(0.5, 0.5, 'No object click data', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Object Clicks Over Time', fontweight='bold')
    
    #click distribution between objects
    if len(objects_clicked_df) > 0:
        # counting the number of clicks per object
        click_summary = objects_clicked_df.groupby(['object_type_name', 'object_index']).size().reset_index(name='click_count')
        
        unique_objects = click_summary['object_index'].unique()
        x_pos = np.arange(len(unique_objects))
        
        point_counts = []
        line_counts = []
        
        for obj_idx in unique_objects:
            point_click = click_summary[(click_summary['object_index'] == obj_idx) & 
                                        (click_summary['object_type_name'] == 'Point')]
            line_click = click_summary[(click_summary['object_index'] == obj_idx) & 
                                        (click_summary['object_type_name'] == 'Line')]
            
            point_counts.append(point_click['click_count'].sum() if not point_click.empty else 0)
            line_counts.append(line_click['click_count'].sum() if not line_click.empty else 0)
        
        bar_width = 0.35
        axes[1, 1].bar(x_pos - bar_width/2, point_counts, bar_width, label='Points', alpha=0.7)
        axes[1, 1].bar(x_pos + bar_width/2, line_counts, bar_width, label='Lines', alpha=0.7)
        
        axes[1, 1].set_title('Click Distribution by Object', fontweight='bold')
        axes[1, 1].set_xlabel('Object Index')
        axes[1, 1].set_ylabel('Number of Clicks')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(unique_objects)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No object click data for distribution', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Click Distribution by Object', fontweight='bold')
    
    plt.tight_layout()
    
    output_filename = f"simulation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graphs saved as: {output_filename}")
    
    plt.show()
if __name__ == "__main__":
    
    create_graphs()