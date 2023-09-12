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
(0.5882352941176471, 0.5882352941176471, 0.5882352941176471), 
(0.9921568627450981, 0.5529411764705883, 0.23529411764705882), 
(0.4196078431372549, 0.6823529411764706, 0.8392156862745098),
] 

tab20c_light = [ 
(0.6313725490196078, 0.8509803921568627, 0.6078431372549019), 
(0.7372549019607844, 0.7411764705882353, 0.8627450980392157), 
(0.7411764705882353, 0.7411764705882353, 0.7411764705882353),
(0.9921568627450981, 0.6823529411764706, 0.4196078431372549),
(0.6196078431372549, 0.792156862745098, 0.8823529411764706)
]

tab20c_dark = [
(0.19215686274509805, 0.6392156862745098, 0.32941176470588235), 
(0.4588235294117647, 0.4196078431372549, 0.6941176470588235), 
(0.38823529411764707, 0.38823529411764707, 0.38823529411764707),
(0.9019607843137255, 0.3333333333333333, 0.050980392156862744),
(0.19215686274509805, 0.5098039215686274, 0.7411764705882353)
] 

#print(255*np.array(tab20c_dark))
#import sys; sys.exit()


if __name__ == '__main__':
    
    cfg = OmegaConf.load('../cfg/plot_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    # Plot mean and max bar plot
    fig_size = (7, 4)
    '''
    fig, ax = plt.subplots(figsize=fig_size)
    width = 0.15
    for i, data_file in enumerate(cfg.robot_evaluation_data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        ind = data['num_nodes']
        if i==0:
            bar = ax.bar(ind, np.mean(data['max_node_displacements_per_tree_size'], axis=1), width=0.2, color=tab20c[i])      
        bar = ax.bar(ind+width*(i+1), np.mean(data['max_node_dist_errors_per_tree_size'], axis=1), width=0.2, color=tab20c[i+1])

    #ax.set_xlabel('Number of nodes per tree')
    #ax.set_ylabel('Distance (m)')
    # Make axis label bold
    ax.xaxis.label.set_fontweight('bold')  
    ax.yaxis.label.set_fontweight('bold')  
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_elements = [Line2D([0], [0], linewidth=5, color=tab20c_dark[0], label='Maximum node distance between initial & target state')]
    for i, data_file in enumerate(cfg.robot_evaluation_data_files):
        policy = data_file.split('-')[1].split('_')[1]
        legend_elements.append(Line2D([0], [0], linewidth=5, color=tab20c_dark[i+1], label='Maximum node error between predicted & target state: ' + r'$\bf{ ' + policy + '}$'))
    #ax.legend(handles=legend_elements, loc='upper left')
    # Set y axis limit
    ax.set_ylim(cfg.y_axis_range[0], 0.2)
    # Create a new set of tick labels with certain indices bolded
    new_tick_labels = ['' for i, label in enumerate(ind)]
    plt.xticks(ind+width*2, new_tick_labels)
    # remove numbers from y axis
    plt.yticks(np.arange(0,0.21, 0.05))
    #plt.show()




    # Plot mean and max bar plot
    fig, ax = plt.subplots(figsize=fig_size)
    width = 0.15
    for i, data_file in enumerate(cfg.robot_evaluation_data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        ind = data['num_nodes']
        if i==0:
            bar = ax.bar(ind, np.mean(data['mean_node_displacements_per_tree_size'], axis=1), width=0.2, color=tab20c[i])      
        bar = ax.bar(ind+width*(i+1), np.mean(data['mean_node_dist_errors_per_tree_size'], axis=1), width=0.2, color=tab20c[i+1])

    ax.set_xlabel('Number of nodes per tree')
    ax.set_ylabel('Distance (m)')
    # Make axis label bold
    ax.xaxis.label.set_fontweight('bold')  
    ax.yaxis.label.set_fontweight('bold')  
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_elements = [Line2D([0], [0], linewidth=5, color=tab20c_dark[0], label='Mean node distance between initial & target state')]
    for i, data_file in enumerate(cfg.robot_evaluation_data_files):
        policy = data_file.split('-')[1].split('_')[1]
        legend_elements.append(Line2D([0], [0], linewidth=5, color=tab20c_dark[i+1], label='Mean node error between predicted & target state: ' + r'$\bf{ ' + policy + '}$'))
    #ax.legend(handles=legend_elements, loc='upper left')
    # Set y axis limit
    ax.set_ylim(cfg.y_axis_range[0], 0.1)
    plt.xticks(ind+width, new_tick_labels)
    #plt.show()
    '''
    
    # Plot overall histogram 
    overall_histogram = []
    planning_failed = 0
    planning_success = 0
    error_dict = {}
    displacement_dict = {}
    for i in range(1, 31):
        error_dict[i] = []
        displacement_dict[i] = []
    for i, data_file in enumerate(cfg.robot_evaluation_data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        ind = data['num_nodes']
        for num_nodes, ranks, errors, displacements in zip(ind, data['rank_per_tree_size'], data['max_node_dist_errors_per_tree_size'], data['max_node_displacements_per_tree_size']):
            for rank, error, displacement in zip(ranks, errors, displacements):
                if rank+1!=num_nodes:
                    overall_histogram.append(rank+1)
                    planning_success+=1
                    error_dict[rank+1].append(error)
                    displacement_dict[rank+1].append(displacement)
                else:
                    planning_failed+=1
    # Compute mean error per rank
    mean_error_per_rank = []
    mean_displacement_per_rank = []
    for i in range(1, 31):
        mean_error_per_rank.append(np.mean(error_dict[i]))
        mean_displacement_per_rank.append(np.mean(displacement_dict[i]))
    
    print('Planning success: ', planning_success)
    print('Planning failed: ', planning_failed)
    print('Planning success rate: ', planning_success/(planning_success+planning_failed))
    fig, ax1 = plt.subplots(figsize=fig_size)
    h = ax1.hist(overall_histogram, bins=np.arange(1,32)-0.5, rwidth=0.8, color='orange')
    print(h[0])
    # Histogram height is the number of trees
    # Write the mean error and mean displacement in text on top of the bars
    #for i in range(30):
    #    if mean_error_per_rank[i] > 0:
    #        ax.text(i+1, h[0][i]+0.01, str(round(mean_error_per_rank[i], 2)), color='black', fontweight='bold')

    #for i, v in enumerate(mean_error_per_rank):
    #    ax.text(i+1-0.25, v+0.01, str(round(v, 2)), color='black', fontweight='bold')
    #for i, v in enumerate(mean_displacement_per_rank):
    #    ax.text(i+1+0.25, v+0.01, str(round(v, 2)), color='black', fontweight='bold')
    # Shift bars to the left
    #ax1.set_xticks(np.arange(1,28)-0.5)
    
    ax1.set_xlabel('Node Affordance Rank')
    #ax1.set_ylabel('Number of feasible plans found')
    ax1.set_xlim(0, 27)

    # Make axis label bold
    #ax.xaxis.label.set_fontweight('bold')
    #ax.yaxis.label.set_fontweight('bold')
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1,31), mean_error_per_rank, color=tab20c_dark[-1], linewidth=3, linestyle='dashed', label='Mean node error')
    ax2.plot(np.arange(1,31), mean_displacement_per_rank, color=tab20c_dark[0], linewidth=3, linestyle='dashed', label='Mean node displacement')
    
    # Add custom x axis labels
    ax2.set_xticks(np.arange(1,27))
    #ax2.set_ylabel('Distance (m)')
    ax2.yaxis.tick_left()
    ax1.yaxis.tick_right()
    
    ax2.set_ylim(0, 0.25)
    plt.show()

    import sys; sys.exit()
    
    # Plot histogram of rank
    for i, data_file in enumerate(cfg.robot_evaluation_data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        ind = data['num_nodes']
        for num_nodes, ranks in zip(ind, data['rank_per_tree_size']):
            fig, ax = plt.subplots(figsize=fig_size)
            ax.hist(ranks, bins=num_nodes, range=(0,num_nodes))
            ax.set_xlabel('Rank')
            ax.set_ylabel('Number of trees')
            # Make x-axis numbers show all ints
            # Make axis label bold
            ax.xaxis.label.set_fontweight('bold')
            ax.yaxis.label.set_fontweight('bold')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.show()
            
