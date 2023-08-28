import os
from omegaconf import OmegaConf
import pickle
import utils
import torch
from model import GNNSimulator, PointNet
import subprocess
import tempfile 

def visualize(model, data_loader, device, cfg):
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data.to(device)
            out = model(data)
            prediction = (data.initial_position + out).cpu().numpy()
            initial_pos = data.initial_position.cpu().numpy()
            final_pos = data.final_position.cpu().numpy()
            edge_index = data.edge_index.T.cpu().numpy()
            edge_index = edge_index[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]
            contact_node = torch.argmax(data.contact_node.float())
            contact_force = data.contact_force

            # Get the information on edge stiffness
            edge_stiffness = data.stiffness.cpu().numpy()
            edge_stiffness = edge_stiffness[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]
            edge_radius = data.radius.cpu().numpy()
            edge_radius = edge_radius[(data.branch.cpu().numpy()==1) & (data.parent2child.cpu().numpy()==1)]

            # Convert the nodes and edges to nx graph
            g_prediction = utils.tensor_to_nx(prediction, edge_index, edge_stiffness, edge_radius)
            g_initial = utils.tensor_to_nx(initial_pos, edge_index, edge_stiffness, edge_radius)
            g_final = utils.tensor_to_nx(final_pos, edge_index, edge_stiffness, edge_radius)
            # Seriealize data to pass to URDF_visualizer.py
            #data = [g_initial, g_final, g_prediction, contact_node, contact_force]
            data = [[g_initial], [g_final], [g_prediction], contact_node.unsqueeze(0), contact_force.unsqueeze(0)]
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

    test_graph_list = []
    for test_data in cfg.test_data_name:
        test_data_path = os.path.join(cfg.data_root, cfg.mode, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list += test_graphs[:len(test_graphs)//10]
    if cfg.fully_connected:
        test_graph_list = utils.preprocess_graphs_to_fully_connected(test_graph_list)
    else:
        test_graph_list = utils.preprocess_graphs(test_graph_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.policy=='gnn':
        model = GNNSimulator(hidden_size=cfg.model.hidden_size, 
                             num_IN_layers=cfg.model.num_IN_layers,
                             forward_model=True).to(device)
    elif cfg.policy=='pointnet':
        model = PointNet(forward_model=True).to(device)
    model.load_state_dict(torch.load(cfg.forward_model_ckpt_path))
    
    test_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=1, shuffle=True)
    visualize(model, test_loader, device, cfg)        

    

