import os
from omegaconf import OmegaConf
import pickle
import utils
import torch
from model import LearnedPolicy
import json
import subprocess
import tempfile
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm
import torch_scatter

def test(model, data_loader, device, cfg):
    max_node_dist_errors = []
    max_node_displacements = []
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader) as tepoch:
            for num_data, data in enumerate(tepoch):
                data.to(device)
                # Predictions
                node_selection_logits, contact_force_pred = model(data)

                # Get idx of node with max node_selection_logits for each graph
                _, contact_node_pred = torch_scatter.scatter_max(node_selection_logits, data.batch, dim=0)
                # Index into contact_force_red using contact_node_idx
                contact_force_pred = contact_force_pred[contact_node_pred].reshape(-1, 3)
                
                # Ground truth
                _, contact_node_gt = torch_scatter.scatter_max(data.contact_node, data.batch, dim=0)
                contact_force_gt = data.contact_force.reshape(-1, 3)

                #initial_pos = data.initial_position.cpu().numpy()
                #final_pos = data.final_position.cpu().numpy()
                #edge_index = #.cpu().numpy()
                edge_index = data.edge_index.T[(data.branch==1) & (data.parent2child==1)]
                # Get the information on edge stiffness
                #edge_stiffness = data.stiffness.cpu().numpy()
                edge_stiffness = data.stiffness[(data.branch==1) & (data.parent2child==1)]

                # Convert the nodes and edges to nx graph
                g_initials = []
                g_finals = []
                g_predictions = []
                for i in range(torch.max(data.batch)+1):
                    initial_pos = data.initial_position[data.batch==i].cpu().numpy()
                    final_pos = data.final_position[data.batch==i].cpu().numpy()
                    edge_index_ = edge_index[data.batch[edge_index[:,0]]==i]
                    edge_index_ = edge_index_ - torch.min(edge_index_) # Reindex the edge index so that it starts from 0
                    edge_index_ = edge_index_.cpu().numpy()
                    edge_stiffness_ = edge_stiffness[data.batch[edge_index[:,0]]==i].cpu().numpy()
                    g_initials.append(utils.tensor_to_nx(initial_pos, edge_index_, edge_stiffness_))
                    g_finals.append(utils.tensor_to_nx(final_pos, edge_index_, edge_stiffness_))            
                    g_predictions.append(utils.tensor_to_nx(initial_pos, edge_index_, edge_stiffness_))
                    # Reindex contact_node_pred and contact_node_gt so that each graph starts from 0 based on the batch
                    contact_node_pred[i] = contact_node_pred[i] - torch.sum(data.batch<i)
                    contact_node_gt[i] = contact_node_gt[i] - torch.sum(data.batch<i)
                
                # Serialize data to pass to URDF_visualizer.py
                auto_close = 100
                data = [g_initials, g_finals, g_predictions, contact_node_gt, contact_force_gt, contact_node_pred, contact_force_pred, auto_close]
                
                serialized_data = pickle.dumps(data)
                # Create temporary file to store serialized data
                with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                    temp_file.write(serialized_data)
                    # Launch URDF_visualizer.py and pass the temporary file name as argument
                    process = subprocess.run(['python', 'urdf_visualizer.py', '--temp_file', temp_file.name, '--mode', cfg.mode], capture_output=True, text=True) # TODO: Temp commented
                temp_file.close()
                
                # Get the standard output from the completed process
                serialized_data = process.stdout.strip()
                    
                # Find the positions of the delimiters
                start_idx = serialized_data.find("DATA_START") + len("DATA_START")
                end_idx = serialized_data.find("DATA_END")
                if start_idx != -1 and end_idx != -1:
                    # Extract the pickled data between delimiters
                    result = json.loads(serialized_data[start_idx:end_idx])
                max_node_dist_errors+=result['max_node_dist_error']
                max_node_displacements+=result['max_node_displacement']
    return max_node_dist_errors, max_node_displacements

if __name__ == '__main__':
    
    cfg = OmegaConf.load('cfg/test_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LearnedPolicy(hidden_size=cfg.model.hidden_size, num_IN_layers=cfg.model.num_IN_layers).to(device)
    model.load_state_dict(torch.load(cfg.control_policy_ckpt_path))

    max_node_dist_errors_per_tree_size = []
    max_node_displacements_per_tree_size = []
    for test_data in cfg.test_data_name:
        print(test_data)
        test_data_path = os.path.join(cfg.data_root, cfg.mode, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list = test_graphs[:len(test_graphs)]
        test_loader = utils.nx_to_pyg_dataloader(utils.preprocess_graphs_to_fully_connected(test_graph_list), 
                                                        batch_size=cfg.test.batch_size, shuffle=False)
        max_node_dist_errors, max_node_displacements = test(model, test_loader, device, cfg)
        max_node_dist_errors_per_tree_size.append(max_node_dist_errors)
        max_node_displacements_per_tree_size.append(max_node_displacements)
    
    # Plot the two violin plots in the same plot
    first_node_size = 8 # TODO: Make this a parameter in the config file
    num_nodes = np.arange(first_node_size,first_node_size+len(cfg.test_data_name))
    print(len(max_node_displacements_per_tree_size))
    print(len(num_nodes))
    fig, ax = plt.subplots()
    ax.violinplot(max_node_displacements_per_tree_size, num_nodes, showmeans=True, showextrema=False, showmedians=False)
    ax.violinplot(max_node_dist_errors_per_tree_size, num_nodes, showmeans=True, showextrema=False, showmedians=False)
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

