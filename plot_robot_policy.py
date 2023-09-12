import os
from omegaconf import OmegaConf
import pickle
from utils import utils
import torch
from model import GNNSimulator, HeuristicBaseline, PointNet
import subprocess
import tempfile
import json 
import numpy as np
from tqdm import tqdm

def test(model, data_loader, device, cfg):
    max_node_dist_errors = []
    max_node_displacements = []
    mean_node_dist_errors = []
    mean_node_displacements = []
    ranks = []
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader) as tepoch:
            for num_data, data in enumerate(tepoch):
                data.to(device)
                # Predictions
                per_node_affordance, per_node_vector = model(data)
                #contact_node = torch.argmax(node_selection_logits)
                #contact_force = contact_force[contact_node]
                # Ground Truth
                contact_node_gt = torch.argmax(data.contact_node.float())
                contact_force_gt = data.contact_force
                
                #print(f'Contact node - Predicted: {contact_node.detach().cpu().numpy()}')
                #print(f'            Ground Truth: {contact_node_gt.detach().cpu().numpy()}')
                #print(f'Contact force - Predicted: {contact_force.flatten().detach().cpu().numpy()}')
                #print(f'             Ground Truth: {contact_force_gt.flatten().detach().cpu().numpy()}')
                initial_pos = data.initial_position.cpu().numpy()
                final_pos = data.final_position.cpu().numpy()
                edge_index = data.edge_index.T.cpu().numpy()
                edge_index = edge_index[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]

                # Get the information on edge stiffness and radius
                edge_stiffness = data.stiffness.cpu().numpy()
                edge_stiffness = edge_stiffness[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]
                edge_radius = data.radius.cpu().numpy()
                edge_radius = edge_radius[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]

                # print edge property stiffness from nx graph
                # Convert the nodes and edges to nx graph
                g_prediction = utils.tensor_to_nx(initial_pos, edge_index, edge_stiffness, edge_radius)
                g_initial = utils.tensor_to_nx(initial_pos, edge_index, edge_stiffness, edge_radius)
                g_final = utils.tensor_to_nx(final_pos, edge_index, edge_stiffness, edge_radius)
                
                # Serialize data to pass to URDF_visualizer.py
                data = [[g_initial], [g_final], [g_prediction], 
                        contact_node_gt.unsqueeze(0), contact_force_gt.unsqueeze(0), per_node_affordance, per_node_vector, 50]
                        #contact_node_gt.unsqueeze(0), contact_force_gt.unsqueeze(0), contact_node.unsqueeze(0), contact_force.unsqueeze(0)]
                serialized_data = pickle.dumps(data)
                # Create temporary file to store serialized data
                with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                    temp_file.write(serialized_data)
                    # Launch URDF_visualizer.py and pass the temporary file name as argument
                    process = subprocess.run(['python', 'robot_simulator.py', '--temp_file', temp_file.name], capture_output=True, text=True)
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
                max_node_dist_errors+=result['max_node_dist_error']
                max_node_displacements+=result['max_node_displacement']
                mean_node_dist_errors+=result['mean_node_dist_error']
                mean_node_displacements+=result['mean_node_displacement']      
                ranks+=result['rank']
    return max_node_dist_errors, max_node_displacements, mean_node_dist_errors, mean_node_displacements, ranks

if __name__ == '__main__':
    
    cfg = OmegaConf.load('cfg/test_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.policy=='gnn':
        model = GNNSimulator(hidden_size=cfg.model.hidden_size, 
                             num_IN_layers=cfg.model.num_IN_layers,
                             forward_model=False).to(device)
        model.load_state_dict(torch.load(cfg.contact_policy_ckpt_path))
    elif cfg.policy=='pointnet':
        model = PointNet(forward_model=False).to(device)
        model.load_state_dict(torch.load(cfg.contact_policy_ckpt_path))        
    else:
        model = HeuristicBaseline(cfg.policy)
    
    print(model)
    max_node_dist_errors_per_tree_size = []
    max_node_displacements_per_tree_size = []
    mean_node_displacements_per_tree_size = []
    mean_node_dist_errors_per_tree_size = []
    rank_per_tree_size = []
    for test_data in cfg.test_data_name:
        print(test_data)
        test_data_path = os.path.join(cfg.data_root, cfg.mode, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list = test_graphs[:len(test_graphs)//10]
        if cfg.randomize_target:
            test_graph_list = utils.set_random_target_configuration(test_graph_list)
        if cfg.fully_connected:
            test_graph_list = utils.preprocess_graphs_to_fully_connected(test_graph_list)
        else:
            test_graph_list = utils.preprocess_graphs(test_graph_list)
        test_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=1, shuffle=False)
        max_node_dist_errors, max_node_displacements, mean_node_dist_errors, mean_node_displacements, rank = test(model, test_loader, device, cfg)

        max_node_dist_errors_per_tree_size.append([x for x in max_node_dist_errors if str(x) != 'nan'])
        max_node_displacements_per_tree_size.append([x for x in max_node_displacements if str(x) != 'nan'])
        mean_node_dist_errors_per_tree_size.append([x for x in mean_node_dist_errors if str(x) != 'nan'])
        mean_node_displacements_per_tree_size.append([x for x in mean_node_displacements if str(x) != 'nan'])
        rank_per_tree_size.append([x for x in rank if str(x) != 'nan'])

    # Plot the two violin plots in the same plot
    first_node_size = 10 # TODO: Make this a parameter in the config file
    num_nodes = np.arange(first_node_size,first_node_size+len(cfg.test_data_name))

    print(np.array(max_node_dist_errors_per_tree_size).shape)
    print(np.array(max_node_displacements_per_tree_size).shape)
    print(np.array(mean_node_dist_errors_per_tree_size).shape)
    print(np.array(mean_node_displacements_per_tree_size).shape)
    print(np.array(rank_per_tree_size).shape)
 
    # Save plot data to pickle
    plot_data = {'max_node_displacements_per_tree_size': max_node_displacements_per_tree_size,
                 'max_node_dist_errors_per_tree_size': max_node_dist_errors_per_tree_size,  
                 'mean_node_displacements_per_tree_size': mean_node_displacements_per_tree_size,
                 'mean_node_dist_errors_per_tree_size': mean_node_dist_errors_per_tree_size,
                 'rank_per_tree_size': rank_per_tree_size,
                 'num_nodes': num_nodes}

    out_name = os.path.join('evaluation', f'ur5-policy_{str(cfg.policy)}-randomized_target_{str(cfg.randomize_target)}.pkl')
    with open(out_name, 'wb') as f:
        pickle.dump(plot_data, f)

