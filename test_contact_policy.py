import os
from omegaconf import OmegaConf
import pickle
from utils import utils
from tqdm import tqdm
import torch
from model import GNNSimulator, PointNet
import torch_scatter 

def validate(model, criterion_reg, criterion_cls, data_loader, lam, device):
    model.eval()
    running_loss_reg = 0
    running_loss_cls = 0
    running_loss = 0
    num_correct = 0
    with torch.no_grad():
        with tqdm(data_loader) as tepoch:
            for num_batch, batch in enumerate(tepoch):
                batch.to(device)
                # Predictions
                node_selection_logits, contact_force_pred = model(batch)
                # Get idx of node with max node_selection_logits for each graph
                _, contact_node_idx = torch_scatter.scatter_max(node_selection_logits, batch.batch, dim=0)
                # Index into contact_force_red using contact_node_idx
                contact_force_pred = contact_force_pred[contact_node_idx].reshape(-1, 3)
        
                # Ground Truth
                contact_node_gt = batch.contact_node.float()
                contact_force_gt = batch.contact_force.reshape(-1, 3)
                
                # Compute loss
                loss_cls = torch.cat([criterion_cls(node_selection_logits[batch.batch==i], contact_node_gt[batch.batch==i]).unsqueeze(0) for i in range(batch.num_graphs)]).mean()
                loss_reg = torch.sqrt(criterion_reg(contact_force_pred, contact_force_gt))
                loss = loss_reg*(1-lam) + loss_cls*lam

                # Logging
                running_loss += loss.item()
                running_loss_reg += loss_reg.item()
                running_loss_cls += loss_cls.item()
                for i in range(batch.num_graphs):
                    num_correct+=torch.sum(torch.argmax(node_selection_logits[batch.batch==i])==torch.argmax(contact_node_gt[batch.batch==i]))                #loss = loss_reg*(1-lam) + loss_cls*lam
                tepoch.set_postfix(loss=(running_loss/(num_batch+1)))
        val_loss = running_loss/len(data_loader)
        val_loss_reg = running_loss_reg/len(data_loader)
        val_loss_cls = running_loss_cls/len(data_loader)
        accuracy = (num_correct/len(data_loader.dataset)).item()
    return val_loss, val_loss_reg, val_loss_cls, accuracy

if __name__ == '__main__':
    
    cfg = OmegaConf.load('cfg/test_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    
    test_graph_list = []
    for test_data in cfg.test_data_name:
        test_data_path = os.path.join(cfg.data_root, cfg.mode, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list += test_graphs[:len(test_graphs)//2]
    if cfg.randomize_target:
        test_graph_list = utils.set_random_target_configuration(test_graph_list)
    if cfg.fully_connected:
        test_graph_list = utils.preprocess_graphs_to_fully_connected(test_graph_list)
    else:
        test_graph_list = utils.preprocess_graphs(test_graph_list)
    validate_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=cfg.test.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.policy=='gnn':
        model = GNNSimulator(hidden_size=cfg.model.hidden_size, 
                             num_IN_layers=cfg.model.num_IN_layers,
                             forward_model=False).to(device)
    elif cfg.policy=='pointnet':
        model = PointNet(forward_model=False).to(device)
    model.load_state_dict(torch.load(cfg.contact_policy_ckpt_path))

    criterion_reg = torch.nn.MSELoss()
    criterion_cls = torch.nn.CrossEntropyLoss()

    lam = 0.9
    val_loss, val_loss_reg, val_loss_cls, accuracy = validate(model, criterion_reg, criterion_cls, validate_loader, lam, device)
    print(f'Validation Loss: {val_loss:.4f}\nValidation Loss Reg: {val_loss_reg:.4f}\nValidation Loss Cls: {val_loss_cls:.4f}\nAccuracy: {accuracy:.4f}')
        

