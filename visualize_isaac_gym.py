import os
from omegaconf import OmegaConf
import pickle
from URDF_visualizer import URDFVisualizer # TODO: There is a memory leak problem with this class
import utils
import torch
from model import LearnedSimulator

def visualize(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data.to(device)
            out = model(data)
            prediction = (data.initial_position + out).cpu().numpy()
            initial_pos = data.initial_position.cpu().numpy()
            final_pos = data.final_position.cpu().numpy()
            edge_index = data.edge_index.T.cpu().numpy()
            edge_index = edge_index[data.branch.cpu().numpy()==1]

            # Convert the nodes and edges to nx graph
            g_prediction = utils.tensor_to_nx(prediction, edge_index)
            g_initial = utils.tensor_to_nx(initial_pos, edge_index)
            g_final = utils.tensor_to_nx(final_pos, edge_index)
            visualizer = URDFVisualizer(g_initial, g_final, g_prediction)
            del visualizer
            
if __name__ == '__main__':

    cfg = OmegaConf.load('cfg/test_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))

    test_graph_list = []
    for test_data in cfg.test_data_name:
        test_data_path = os.path.join(cfg.data_root, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list += test_graphs[:len(test_graphs)//2]
    test_graph_list = utils.preprocess_graphs_to_fully_connected(test_graph_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LearnedSimulator(hidden_size=cfg.model.hidden_size, num_IN_layers=cfg.model.num_IN_layers).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint_path))
    
    test_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=1, shuffle=True)
    visualize(model, test_loader, device)        

    

