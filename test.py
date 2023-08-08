import os
from omegaconf import OmegaConf
import pickle
import utils
import torch
from model import LearnedSimulator
import matplotlib.pyplot as plt

def test(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for num_batch, batch in enumerate(data_loader):
            batch.to(device)
            out = model(batch)
            loss = criterion(batch.initial_position+out, batch.final_position)
            running_loss += loss.item()
        test_loss = running_loss/len(data_loader)
    return test_loss

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
            num_nodes = len(initial_pos)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Plot inital tree edges as line segments0
            for edge in edge_index:
                ax.plot([initial_pos[edge[0],0], initial_pos[edge[1],0]], [initial_pos[edge[0],1], initial_pos[edge[1],1]], [initial_pos[edge[0],2], initial_pos[edge[1],2]], color='b')
                ax.plot([final_pos[edge[0],0], final_pos[edge[1],0]], [final_pos[edge[0],1], final_pos[edge[1],1]], [final_pos[edge[0],2], final_pos[edge[1],2]], color='g')
                ax.plot([prediction[edge[0],0], prediction[edge[1],0]], [prediction[edge[0],1], prediction[edge[1],1]], [prediction[edge[0],2], prediction[edge[1],2]], color='r')
            # Set plot axes properties
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(0, 1.0)
            # Setup legend
            ax.plot([0], [0], [0], color='b', label='Initial')
            ax.plot([0], [0], [0], color='g', label='Final')
            ax.plot([0], [0], [0], color='r', label='Predicted')
            ax.legend()
            # Title
            ax.set_title(f'Reconstruction for {num_nodes} nodes tree')
            plt.show()


if __name__ == '__main__':

    cfg = OmegaConf.load('cfg/test_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))

    test_graph_list = []
    for test_data in cfg.test_data_name:
        test_data_path = os.path.join(cfg.data_root, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list += test_graphs
    test_graph_list = utils.to_fully_connected(test_graph_list)
    test_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=cfg.test.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LearnedSimulator(hidden_size=cfg.model.hidden_size, num_IN_layers=cfg.model.num_IN_layers).to(device)
    # load checkpoint
    model.load_state_dict(torch.load(cfg.checkpoint_path))
    criterion = torch.nn.MSELoss()
    test_loss = test(model, criterion, test_loader, device)
    print(f'Test loss: {test_loss}')
    if cfg.test.visualize:
        test_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=1, shuffle=True)
        visualize(model, test_loader, device)        

    

