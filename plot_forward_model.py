import os
from omegaconf import OmegaConf
import pickle
import utils
import torch
from model import GNNSimulator, PointNet
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np

def test(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0
    max_distance_errors = []
    with torch.no_grad():
        for num_batch, batch in enumerate(data_loader):
            batch.to(device)
            out = model(batch)
            prediction = batch.initial_position+out
            ground_truth = batch.final_position
            loss = criterion(prediction, ground_truth)
            running_loss += loss.item()

            # Metric: Maximum node distance error per tree
            distance_error = torch.norm(prediction-ground_truth, dim=-1)
            max_distance_error = torch.stack([torch.max(distance_error[batch.batch==i]) for i in range(batch.num_graphs)])
            max_distance_errors.append(max_distance_error)
        test_loss = running_loss/len(data_loader)
    max_distance_errors = torch.cat(max_distance_errors)
    return test_loss, max_distance_errors

def dataset_stats(data_loader, device):
    # For each tree, compute the max_distance_displacement between initial and final positions
    max_distance_displacements = []
    for num_batch, batch in enumerate(data_loader):
        batch.to(device)
        displacement_distance = torch.norm(batch.final_position-batch.initial_position, dim=-1)
        max_distance_displacement = torch.stack([torch.max(displacement_distance[batch.batch==i]) for i in range(batch.num_graphs)])
        max_distance_displacements.append(max_distance_displacement)
    max_distance_displacements = torch.cat(max_distance_displacements)
    return max_distance_displacements

if __name__ == '__main__':

    cfg = OmegaConf.load('cfg/test_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.policy=='gnn':
        model = GNNSimulator(hidden_size=cfg.model.hidden_size, 
                             num_IN_layers=cfg.model.num_IN_layers,
                             forward_model=True).to(device)
    elif cfg.policy=='pointnet':
        model = PointNet(forward_model=True).to(device)
    # load checkpoint
    model.load_state_dict(torch.load(cfg.forward_model_ckpt_path))
    criterion = torch.nn.MSELoss()
    
    first_node_size = 8 # TODO: Make this a parameter in the config file
    num_nodes = np.arange(first_node_size,first_node_size+len(cfg.test_data_name))
    assert(len(num_nodes)==len(cfg.test_data_name))
    test_loss_list = []
    max_distance_errors_list = []
    max_distance_displacemnts_list = []
    for i, test_data in enumerate(cfg.test_data_name):    
        
        test_data_path = os.path.join(cfg.data_root, cfg.mode, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graphs = test_graphs[:len(test_graphs)]
        if cfg.fully_connected:
            test_graph_list = utils.preprocess_graphs_to_fully_connected(test_graphs)
        else:
            test_graph_list = utils.preprocess_graphs(test_graphs)
        test_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=cfg.test.batch_size, shuffle=True)
        
        # Compute dataset stats
        max_distance_displacemnts = dataset_stats(test_loader, device)
        max_distance_displacemnts_list.append(max_distance_displacemnts.detach().cpu().numpy())

        # Evaluate model
        test_loss, max_distance_errors = test(model, criterion, test_loader, device)
        test_loss_list.append(test_loss)
        max_distance_errors_list.append(max_distance_errors.detach().cpu().numpy())
        


        print(f'Finished evaluating {i+1}/{len(cfg.test_data_name)} graph lists.')

    # For each number of nodes, plot the violin plot of max distance displacement
    # Plot the two violin plots in the same plot
    fig, ax = plt.subplots()
    ax.violinplot(max_distance_displacemnts_list, num_nodes, showmeans=True, showextrema=False, showmedians=False)
    ax.violinplot(max_distance_errors_list, num_nodes, showmeans=True, showextrema=False, showmedians=False)
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Distance (m)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_elements = [Line2D([0], [0], color='C0', label='Average distance of node with maximum ditsance between initial and final state'),
                       Line2D([0], [0], color='C1', label='Average distance of node with maximum distance between predicted and final state')]
    ax.legend(handles=legend_elements, loc='upper left')
    # Set y axis limit
    ax.set_ylim(0, 1.0)
    plt.xticks(num_nodes)
    plt.show()


    
    

