import os
from omegaconf import OmegaConf
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np

# Set default color palette for bar plots



tab20c = [
(0.4549019607843137, 0.7686274509803922, 0.4627450980392157),       
(0.6196078431372549, 0.6039215686274509, 0.7843137254901961), 
(0.9921568627450981, 0.5529411764705883, 0.23529411764705882), 
(0.5882352941176471, 0.5882352941176471, 0.5882352941176471), 
(0.4196078431372549, 0.6823529411764706, 0.8392156862745098),
] 

tab20c_light = [ 
(0.6313725490196078, 0.8509803921568627, 0.6078431372549019), 
(0.7372549019607844, 0.7411764705882353, 0.8627450980392157), 
(0.9921568627450981, 0.6823529411764706, 0.4196078431372549),
(0.7411764705882353, 0.7411764705882353, 0.7411764705882353),
(0.6196078431372549, 0.792156862745098, 0.8823529411764706)
]

tab20c_dark = [
(0.19215686274509805, 0.6392156862745098, 0.32941176470588235), 
(0.4588235294117647, 0.4196078431372549, 0.6941176470588235), 
(0.9019607843137255, 0.3333333333333333, 0.050980392156862744),
(0.38823529411764707, 0.38823529411764707, 0.38823529411764707),
(0.19215686274509805, 0.5098039215686274, 0.7411764705882353)
] 

trained_systems = np.array([11, 13, 15, 17, 19]) - 10

if __name__ == '__main__':
    
    cfg = OmegaConf.load('../cfg/plot_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    if False:
        # Load plot data from pickle and plot max first
        fig, ax = plt.subplots()
        for i, data_file in enumerate(cfg.evaluation_data_files):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            num_nodes = data['num_nodes']
            if i==0:
                ax.violinplot(data['max_node_displacements_per_tree_size'], num_nodes, showmeans=True, showextrema=False, showmedians=False)
            ax.violinplot(data['max_node_dist_errors_per_tree_size'], num_nodes, showmeans=True, showextrema=False, showmedians=False)
        ax.set_xlabel('Number of nodes per tree')
        ax.set_ylabel('Distance (m)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend_elements = [Line2D([0], [0], color='C0', label='Maximum distance of nodes between initial and target state')]
        for i, data_file in enumerate(cfg.evaluation_data_files):
            policy = data_file.split('-')[1].split('_')[1]
            legend_elements.append(Line2D([0], [0], color=f'C{i+1}', label=f'Maximum distance between predicted and target state: {policy}'))
        ax.legend(handles=legend_elements, loc='upper left')
        # Set y axis limit
        ax.set_ylim(cfg.y_axis_range[0], cfg.y_axis_range[1])
        plt.xticks(num_nodes)
        plt.show()
        
    if False:
        # Load plot data from pickle and plot mean second
        fig, ax = plt.subplots()
        for i, data_file in enumerate(cfg.evaluation_data_files):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            num_nodes = data['num_nodes']
            if i==0:
                ax.violinplot(data['mean_node_displacements_per_tree_size'], num_nodes, showmeans=True, showextrema=False, showmedians=False)
            ax.violinplot(data['mean_node_dist_errors_per_tree_size'], num_nodes, showmeans=True, showextrema=False, showmedians=False)
        ax.set_xlabel('Number of nodes per tree')
        ax.set_ylabel('Distance (m)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend_elements = [Line2D([0], [0], color='C0', label='Mean distance of nodes between initial and target state')]
        for i, data_file in enumerate(cfg.evaluation_data_files):
            legend_elements.append(Line2D([0], [0], color=f'C{i+1}', label=f'\" predicted and target state: {data_file}'))
        ax.legend(handles=legend_elements, loc='upper left')
        # Set y axis limit
        ax.set_ylim(cfg.y_axis_range[0], cfg.y_axis_range[1])
        plt.xticks(num_nodes)
        plt.show()

    # Plot mean and max bar plot
    fig_size = (14, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    width = 0.15
    for i, data_file in enumerate(cfg.evaluation_data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        ind = data['num_nodes']
        if i==0:
            bar = ax.bar(ind, np.mean(data['max_node_displacements_per_tree_size'], axis=1), width=0.2, color=tab20c[i])      
            for idx in trained_systems:  
                bar[idx].set_color(tab20c_dark[i])
        bar = ax.bar(ind+width*(i+1), np.mean(data['max_node_dist_errors_per_tree_size'], axis=1), width=0.2, color=tab20c[i+1])
        for idx in trained_systems:  
            bar[idx].set_color(tab20c_dark[i+1])

    ax.set_xlabel('Number of nodes per tree')
    ax.set_ylabel('Distance (m)')
    # Make axis label bold
    ax.xaxis.label.set_fontweight('bold')  
    ax.yaxis.label.set_fontweight('bold')  
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_elements = [Line2D([0], [0], linewidth=5, color=tab20c_dark[0], label='Maximum node distance between initial & target state')]
    for i, data_file in enumerate(cfg.evaluation_data_files):
        policy = data_file.split('-')[1].split('_')[1]
        legend_elements.append(Line2D([0], [0], linewidth=5, color=tab20c_dark[i+1], label='Maximum node error between predicted & target state: ' + r'$\bf{ ' + policy + '}$'))
    ax.legend(handles=legend_elements, loc='upper left')
    # Set y axis limit
    ax.set_ylim(cfg.y_axis_range[0], 0.2)
    # Create a new set of tick labels with certain indices bolded
    new_tick_labels = [r"$\bf{"+str(label)+"}$" if i in trained_systems else label for i, label in enumerate(ind)]
    plt.xticks(ind+width, new_tick_labels)
    plt.show()




    # Plot mean and max bar plot
    fig, ax = plt.subplots(figsize=fig_size)
    width = 0.15
    for i, data_file in enumerate(cfg.evaluation_data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        ind = data['num_nodes']
        if i==0:
            bar = ax.bar(ind, np.mean(data['mean_node_displacements_per_tree_size'], axis=1), width=0.2, color=tab20c[i])      
            for idx in trained_systems:  
                bar[idx].set_color(tab20c_dark[i])
        bar = ax.bar(ind+width*(i+1), np.mean(data['mean_node_dist_errors_per_tree_size'], axis=1), width=0.2, color=tab20c[i+1])
        for idx in trained_systems:  
            bar[idx].set_color(tab20c_dark[i+1])

    ax.set_xlabel('Number of nodes per tree')
    ax.set_ylabel('Distance (m)')
    # Make axis label bold
    ax.xaxis.label.set_fontweight('bold')  
    ax.yaxis.label.set_fontweight('bold')  
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_elements = [Line2D([0], [0], linewidth=5, color=tab20c_dark[0], label='Mean node distance between initial & target state')]
    for i, data_file in enumerate(cfg.evaluation_data_files):
        policy = data_file.split('-')[1].split('_')[1]
        legend_elements.append(Line2D([0], [0], linewidth=5, color=tab20c_dark[i+1], label='Mean node error between predicted & target state: ' + r'$\bf{ ' + policy + '}$'))
    ax.legend(handles=legend_elements, loc='upper left')
    # Set y axis limit
    ax.set_ylim(cfg.y_axis_range[0], 0.1)
    plt.xticks(ind+width, new_tick_labels)
    plt.show()