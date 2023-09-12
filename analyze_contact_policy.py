import os
from omegaconf import OmegaConf
import pickle
from utils import utils
import torch
from model import GNNSimulator, PointNet
import subprocess
import tempfile
import numpy as np 
import json 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from tqdm import tqdm

def visualize(model, data_loader, device, cfg):
    node_probs = []
    mean_dist_errors = []
    max_dist_errors = []
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader) as tepoch:
            for _, data in enumerate(tepoch):
                data.to(device)
                # Predictions
                node_selection_logits, contact_force = model(data)
                contact_node = torch.argmax(node_selection_logits)
                #contact_force = contact_force[contact_node]
                node_selection_probs = torch.softmax(node_selection_logits, dim=-1)

                # Ground Truth
                contact_node_gt = torch.argmax(data.contact_node.float())
                contact_force_gt = data.contact_force            
                
                #prediction = (data.initial_position + out).cpu().numpy()
                initial_pos = data.initial_position.cpu().numpy()
                final_pos = data.final_position.cpu().numpy()
                edge_index = data.edge_index.T.cpu().numpy()
                edge_index = edge_index[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]
                # Get the information on edge stiffness
                edge_stiffness = data.stiffness.cpu().numpy()
                edge_stiffness = edge_stiffness[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]
                edge_radius = data.radius.cpu().numpy()
                edge_radius = edge_radius[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]

                # Convert the nodes and edges to nx graph
                g_prediction = utils.tensor_to_nx(initial_pos, edge_index, edge_stiffness, edge_radius)
                g_initial = utils.tensor_to_nx(initial_pos, edge_index, edge_stiffness, edge_radius)
                g_final = utils.tensor_to_nx(final_pos, edge_index, edge_stiffness, edge_radius)

                # Serialize data to pass to URDF_visualizer.py
                auto_close = 100
                data = [[g_initial], [g_final], [g_prediction], contact_node_gt, contact_force_gt, node_selection_probs, contact_force, auto_close]
                serialized_data = pickle.dumps(data)
                # Create temporary file to store serialized data
                with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                    temp_file.write(serialized_data)
                    # Launch URDF_visualizer.py and pass the temporary file name as argument
                    process = subprocess.run(['python', 'urdf_visualizer.py', '--temp_file', temp_file.name, '--mode', cfg.mode], capture_output=True, text=True) 
                # Clean up the temporary file
                temp_file.close()

                # Get the standard output from the completed process
                serialized_data = process.stdout.strip()
                    
                # Find the positions of the delimiters
                start_idx = serialized_data.find("DATA_START") + len("DATA_START")
                end_idx = serialized_data.find("DATA_END")
                if start_idx != -1 and end_idx != -1:
                    # Extract the pickled data between delimiters
                    result = json.loads(serialized_data[start_idx:end_idx])
                mean_dist_errors.append(result['mean_dist_error'])
                max_dist_errors.append(result['max_dist_error'])
                node_probs.append(result['node_probs'])
    return node_probs, mean_dist_errors, max_dist_errors

if __name__ == '__main__':
    
    cfg = OmegaConf.load('cfg/test_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    
    test_graph_list = []
    for test_data in cfg.test_data_name:
        test_data_path = os.path.join(cfg.data_root, cfg.mode, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list += test_graphs[:len(test_graphs)//10]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.policy=='gnn':
        model = GNNSimulator(hidden_size=cfg.model.hidden_size, 
                             num_IN_layers=cfg.model.num_IN_layers,
                             forward_model=False).to(device)
    elif cfg.policy=='pointnet':
        model = PointNet(forward_model=False).to(device)
    print(model)
    model.load_state_dict(torch.load(cfg.contact_policy_ckpt_path))
    if cfg.randomize_target:
        test_graph_list = utils.set_random_target_configuration(test_graph_list)
    if cfg.fully_connected:
        test_graph_list = utils.preprocess_graphs_to_fully_connected(test_graph_list)
    else:
        test_graph_list = utils.preprocess_graphs(test_graph_list)
    test_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=1, shuffle=True)
    node_probs, mean_dist_errors, max_dist_errors = visualize(model, test_loader, device, cfg)
    node_probs = np.array(node_probs).T.tolist()
    mean_dist_errors = np.array(mean_dist_errors).T.tolist()
    max_dist_errors = np.array(max_dist_errors).T.tolist()
    # Transpose the arrays

    num_nodes = test_graphs[0].number_of_nodes()
    node_order = np.arange(1, num_nodes+1)

    plot_data = {'node_order': node_order,
                 'node_probs': node_probs,
                 'max_dist_errors': max_dist_errors,  
                 'mean_dist_errors': mean_dist_errors,}
    
    out_name = os.path.join('evaluation', f'analysis_{str(cfg.mode)}-policy_{str(cfg.policy)}-randomized_target_{str(cfg.randomize_target)}.pkl')
    with open(out_name, 'wb') as f:
        pickle.dump(plot_data, f)


    # Make violin plot of node selection probabilities
    fig, ax = plt.subplots()
    ax.violinplot(node_probs, node_order, showmeans=True, showextrema=False, showmedians=False)
    ax.set_xlabel('Node confidence order')
    ax.set_ylabel('Node confidence score')
    ax.set_title('Node selection confidence distribution')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(node_order)
    ax.set_xticklabels(node_order)
    plt.show()

    fig, ax = plt.subplots()
    ax.violinplot(max_dist_errors, node_order, showmeans=True, showextrema=False, showmedians=False)
    ax.violinplot(mean_dist_errors, node_order, showmeans=True, showextrema=False, showmedians=False)
    ax.set_xlabel('Node confidence order')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Node distance error distribution in order of node confidence')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_elements = [Line2D([0], [0], color='C0', label='Average distance of node with maximum distance between predicted and final state (WORST CASE)'),
                       Line2D([0], [0], color='C1', label='Average distance of nodes between predicted and final state (AVERAGE CASE)')]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_xticks(node_order)
    ax.set_xticklabels(node_order)
    ax.set_ylim(0, 0.6)
    plt.show()

