import os
from omegaconf import OmegaConf
import pickle
import utils
import torch
from model import LearnedPolicy
import subprocess
import tempfile
import numpy as np 
def visualize(model, data_loader, device, cfg):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data.to(device)
            # Predictions
            node_selection_logits, contact_force = model(data)
            contact_node = torch.argmax(node_selection_logits)
            contact_force = contact_force[contact_node]
            # Ground Truth
            contact_node_gt = torch.argmax(data.contact_node.float())
            contact_force_gt = data.contact_force
            
            print(f'Contact node - Predicted: {contact_node.detach().cpu().numpy()}')
            print(f'            Ground Truth: {contact_node_gt.detach().cpu().numpy()}')
            print(f'Contact force - Predicted: {contact_force.flatten().detach().cpu().numpy()}')
            print(f'             Ground Truth: {contact_force_gt.flatten().detach().cpu().numpy()}')
            
            #prediction = (data.initial_position + out).cpu().numpy()
            initial_pos = data.initial_position.cpu().numpy()
            final_pos = data.final_position.cpu().numpy()
            edge_index = data.edge_index.T.cpu().numpy()
            edge_index = edge_index[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]
    
            # Get the information on edge stiffness
            edge_stiffness = data.stiffness.cpu().numpy()
            edge_stiffness = edge_stiffness[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]

            # print edge property stiffness from nx graph
            # Convert the nodes and edges to nx graph
            g_prediction = utils.tensor_to_nx(initial_pos, edge_index, edge_stiffness)
            g_initial = utils.tensor_to_nx(initial_pos, edge_index, edge_stiffness)
            g_final = utils.tensor_to_nx(final_pos, edge_index, edge_stiffness)
            
            # Serialize data to pass to URDF_visualizer.py
            data = [[g_initial], [g_final], [g_prediction], 
                    contact_node_gt.unsqueeze(0), contact_force_gt.unsqueeze(0), contact_node.unsqueeze(0), contact_force.unsqueeze(0)]
            serialized_data = pickle.dumps(data)
            # Create temporary file to store serialized data
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_file.write(serialized_data)
                # Launch URDF_visualizer.py and pass the temporary file name as argument
                subprocess.run(['python', 'urdf_visualizer.py', '--temp_file', temp_file.name, '--mode', cfg.mode])
            # Clean up the temporary file
            temp_file.close()

if __name__ == '__main__':
    
    cfg = OmegaConf.load('cfg/test_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LearnedPolicy(hidden_size=cfg.model.hidden_size, num_IN_layers=cfg.model.num_IN_layers).to(device)
    model.load_state_dict(torch.load(cfg.control_policy_ckpt_path))
    
    test_graph_list = []
    for test_data in cfg.test_data_name:
        test_data_path = os.path.join(cfg.data_root, cfg.mode, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list += test_graphs[:len(test_graphs)//10]
    test_graph_list = utils.preprocess_graphs_to_fully_connected(test_graph_list)
    test_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=1, shuffle=True)
    visualize(model, test_loader, device, cfg)
        

