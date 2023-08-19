import os
from omegaconf import OmegaConf
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np

if __name__ == '__main__':
    
    cfg = OmegaConf.load('../cfg/plot_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    
    # Load plot data from pickle and plot max first
    fig, ax = plt.subplots()
    for i, data_file in enumerate(cfg.data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        num_nodes = data['num_nodes']
        if i==0:
            ax.violinplot(data['max_node_displacements_per_tree_size'], num_nodes, showmeans=True, showextrema=False, showmedians=False)
        ax.violinplot(data['max_node_dist_errors_per_tree_size'], num_nodes, showmeans=True, showextrema=False, showmedians=False)
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Distance (m)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_elements = [Line2D([0], [0], color='C0', label='Average distance of node with maximum distance between initial and final state')]
    for i, data_file in enumerate(cfg.data_files):
        legend_elements.append(Line2D([0], [0], color=f'C{i+1}', label=f'\" predicted and final state: {data_file}'))
    ax.legend(handles=legend_elements, loc='upper left')
    # Set y axis limit
    ax.set_ylim(cfg.y_axis_range[0], cfg.y_axis_range[1])
    plt.xticks(num_nodes)
    plt.show()
        

    # Load plot data from pickle and plot mean second
    fig, ax = plt.subplots()
    for i, data_file in enumerate(cfg.data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        num_nodes = data['num_nodes']
        if i==0:
            ax.violinplot(data['mean_node_displacements_per_tree_size'], num_nodes, showmeans=True, showextrema=False, showmedians=False)
        ax.violinplot(data['mean_node_dist_errors_per_tree_size'], num_nodes, showmeans=True, showextrema=False, showmedians=False)
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Distance (m)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_elements = [Line2D([0], [0], color='C0', label='Mean distance of nodes between initial and final state')]
    for i, data_file in enumerate(cfg.data_files):
        legend_elements.append(Line2D([0], [0], color=f'C{i+1}', label=f'\" predicted and final state: {data_file}'))
    ax.legend(handles=legend_elements, loc='upper left')
    # Set y axis limit
    ax.set_ylim(cfg.y_axis_range[0], cfg.y_axis_range[1])
    plt.xticks(num_nodes)
    plt.show()
