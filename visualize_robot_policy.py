import os
from omegaconf import OmegaConf
import pickle
from utils import utils
import torch
from model import GNNSimulator, HeuristicBaseline, PointNet
import subprocess
import tempfile

def visualize(model, data_loader, device, cfg):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
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
                    contact_node_gt.unsqueeze(0), contact_force_gt.unsqueeze(0), per_node_affordance, per_node_vector]
                    #contact_node_gt.unsqueeze(0), contact_force_gt.unsqueeze(0), contact_node.unsqueeze(0), contact_force.unsqueeze(0)]
            serialized_data = pickle.dumps(data)
            # Create temporary file to store serialized data
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_file.write(serialized_data)
                # Launch URDF_visualizer.py and pass the temporary file name as argument
                if cfg.test.record_video:
                    subprocess.run(['python', 'robot_simulator.py', '--temp_file', temp_file.name, '--record_video', 'True'])
                else:
                    subprocess.run(['python', 'robot_simulator.py', '--temp_file', temp_file.name])
            # Clean up the temporary file
            temp_file.close()

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
    
    test_graph_list = []
    for test_data in cfg.test_data_name:
        test_data_path = os.path.join(cfg.data_root, cfg.mode, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list += test_graphs[:len(test_graphs)//100]

    if cfg.randomize_target:
        test_graph_list = utils.set_random_target_configuration(test_graph_list)
    if cfg.fully_connected:
        test_graph_list = utils.preprocess_graphs_to_fully_connected(test_graph_list)
    else:
        test_graph_list = utils.preprocess_graphs(test_graph_list)
    test_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=1, shuffle=False)
    visualize(model, test_loader, device, cfg)
        

