import os
from omegaconf import OmegaConf
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np

# Set default color palette for bar plots



tab20c = [(0.9921568627450981, 0.5529411764705883, 0.23529411764705882), 
          (0.6196078431372549, 0.6039215686274509, 0.7843137254901961), 
          (0.4549019607843137, 0.7686274509803922, 0.4627450980392157), 
          (0.5882352941176471, 0.5882352941176471, 0.5882352941176471), 
          (0.4196078431372549, 0.6823529411764706, 0.8392156862745098),] 

tab20c_light = [(0.9921568627450981, 0.6823529411764706, 0.4196078431372549), 
(0.7372549019607844, 0.7411764705882353, 0.8627450980392157), 
(0.6313725490196078, 0.8509803921568627, 0.6078431372549019), 
(0.7411764705882353, 0.7411764705882353, 0.7411764705882353),
(0.6196078431372549, 0.792156862745098, 0.8823529411764706)]

tab20c_dark = [(0.9019607843137255, 0.3333333333333333, 0.050980392156862744),
(0.4588235294117647, 0.4196078431372549, 0.6941176470588235), 
(0.19215686274509805, 0.6392156862745098, 0.32941176470588235), 
(0.38823529411764707, 0.38823529411764707, 0.38823529411764707),
(0.19215686274509805, 0.5098039215686274, 0.7411764705882353)] 

trained_systems = np.array([11, 13, 15, 17, 19]) - 10

if __name__ == '__main__':
    
    cfg = OmegaConf.load('../cfg/plot_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    
    for i, data_file in enumerate(cfg.analysis_data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        node_order = data['node_order']
        node_probs = data['node_probs']
        mean_dist_errors = data['mean_dist_errors']
        max_dist_errors = data['max_dist_errors']

        # Make violin plot of node selection probabilities
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.violinplot(node_probs, node_order, showmeans=True, showextrema=False, showmedians=False)
        #ax1.set_xlabel('Node confidence rank')
        ax1.set_ylabel('Node confidence score')
        ax1.set_xlabel('Node confidence rank')
        #ax1.xaxis.label.set_fontweight('bold')
        #ax1.yaxis.label.set_fontweight('bold')
        ax1.set_title('Node Confidence Score Distribution')
        # Make title bold
        ax1.title.set_fontweight('bold')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend_elements = [Line2D([0], [0], color=tab20c_light[-1], linewidth=5, label='Node confidence score distribution'),
                        Line2D([0], [0], color=tab20c_dark[-1], linewidth=2, label='Mean node confidence score')]
        #ax1.legend(handles=legend_elements, loc='upper right')
        ax1.set_xticks(node_order)
        ax1.set_xticklabels(node_order)
        # Add some space between ax1 and ax2
        fig.subplots_adjust(hspace=0.3)

        #fig, ax = plt.subplots()
        ax2.bar(node_order, np.mean(max_dist_errors, axis=1), width=0.5, color=tab20c_light[-1])      
        ax2.plot(node_order, np.mean(mean_dist_errors, axis=1), color=tab20c_dark[-1], linestyle='None', marker='_', markersize=15, markeredgewidth=2)    
        #ax.violinplot(max_dist_errors, node_order, showmeans=True, showextrema=False, showmedians=False)
        #ax.violinplot(mean_dist_errors, node_order, showmeans=True, showextrema=False, showmedians=False)
        ax2.set_xlabel('Node confidence rank')
        ax2.set_ylabel('Distance (m)')
        #ax2.xaxis.label.set_fontweight('bold')
        #ax2.yaxis.label.set_fontweight('bold')
        ax2.set_title('Node Position Error by Confidence Rank')
        ax2.title.set_fontweight('bold')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend_elements = [Line2D([0], [0], color=tab20c_light[-1], linewidth=5, label='Maximum node error between predicted and target state'),
                        Line2D([0], [0], color=tab20c_dark[-1], linewidth=2, label='Mean node error between predicted and target state')]
        #ax2.legend(handles=legend_elements, loc='upper left')
        ax2.set_xticks(node_order)
        ax2.set_xticklabels(node_order)
        ax2.set_ylim(0, 0.2)
        # Set y axis ticks to increment by 0.05
        ax2.yaxis.set_ticks(np.arange(0, 0.2, 0.05))
        plt.show()

    # Make the two plots above side by side
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))



    '''
    # Plot mean and max bar plot
    fig, ax = plt.subplots()
    width = 0.15
    for i, data_file in enumerate(cfg.data_files):
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
    legend_elements = [Line2D([0], [0], linewidth=5, color=tab20c_dark[0], label='Maximum distance of nodes between initial & final state')]
    for i, data_file in enumerate(cfg.data_files):
        policy = data_file.split('-')[1].split('_')[1]
        legend_elements.append(Line2D([0], [0], linewidth=5, color=tab20c_dark[i+1], label=' '*60+'predicted & final state: ' + r'$\bf{ ' + policy + '}$'))
    ax.legend(handles=legend_elements, loc='upper left')
    # Set y axis limit
    ax.set_ylim(cfg.y_axis_range[0], 0.2)
    plt.xticks(ind+width, ind)
    plt.show()

    # Plot mean and max bar plot
    fig, ax = plt.subplots()
    width = 0.15
    for i, data_file in enumerate(cfg.data_files):
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
    legend_elements = [Line2D([0], [0], linewidth=5, color=tab20c_dark[0], label='Mean distance of nodes between initial & final state')]
    for i, data_file in enumerate(cfg.data_files):
        policy = data_file.split('-')[1].split('_')[1]
        legend_elements.append(Line2D([0], [0], linewidth=5, color=tab20c_dark[i+1], label=' '*53+'predicted & final state: ' + r'$\bf{ ' + policy + '}$'))
    ax.legend(handles=legend_elements, loc='upper left')
    # Set y axis limit
    ax.set_ylim(cfg.y_axis_range[0], 0.1)
    plt.xticks(ind+width, ind)
    plt.show()
    '''