import os
from omegaconf import OmegaConf
import pickle
import utils
import copy
from tqdm import tqdm
import torch
from model import GNNSimulator, PointNet
import torch_scatter
import wandb
import time

def train(model, optimizer, criterion_reg, criterion_cls, train_loader, epoch, lam, device):
    model.train()
    running_loss_reg = 0
    running_loss_cls = 0
    running_loss = 0
    with tqdm(train_loader) as tepoch:
        for num_batch, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
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

            # Backprop
            loss.backward()
            optimizer.step()
            
            # Logging
            running_loss += loss.item()
            running_loss_reg += loss_reg.item()
            running_loss_cls += loss_cls.item()
            # Metric: Maximum node distance error per tree
            tepoch.set_postfix(loss=(running_loss/(num_batch+1)))
    train_loss = running_loss/len(train_loader)
    train_loss_reg = running_loss_reg/len(train_loader)
    train_loss_cls = running_loss_cls/len(train_loader)
    return train_loss, train_loss_reg, train_loss_cls

def validate(model, criterion_reg, criterion_cls, data_loader, epoch, lam, device):
    model.eval()
    running_loss_reg = 0
    running_loss_cls = 0
    running_loss = 0
    num_correct = 0
    with torch.no_grad():
        for num_batch, batch in enumerate(data_loader):
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
                num_correct+=torch.sum(torch.argmax(node_selection_logits[batch.batch==i])==torch.argmax(contact_node_gt[batch.batch==i]))         
        val_loss = running_loss/len(data_loader)
        val_loss_reg = running_loss_reg/len(data_loader)
        val_loss_cls = running_loss_cls/len(data_loader)
        accuracy = (num_correct/len(data_loader.dataset)).item()
    return val_loss, val_loss_reg, val_loss_cls, accuracy 

if __name__ == '__main__':
    
    cfg = OmegaConf.load('cfg/train_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))

    time_str = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/{time_str}_{cfg.name}"
    os.makedirs(output_dir, exist_ok=True)

    if cfg.logging.wandb:
        wandb.login()
        run = wandb.init(
            project="tree_is_all_you_need_v2_cp",
            name=cfg.name,
            config=OmegaConf.to_container(cfg)
        )
    
    train_graph_list = []    
    for train_data in cfg.train_data_name:
        train_data_path = os.path.join(cfg.data_root, cfg.mode, train_data)
        with open(train_data_path, 'rb') as f:
            train_graphs = pickle.load(f)
        train_graph_list += train_graphs[:len(train_graphs)]
    test_graph_list = []
    for test_data in cfg.test_data_name:
        test_data_path = os.path.join(cfg.data_root, cfg.mode, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list += test_graphs[:len(test_graphs)//2]


    if cfg.fully_connected:    
        train_graph_list = utils.preprocess_graphs_to_fully_connected(train_graph_list)
        test_graph_list = utils.preprocess_graphs_to_fully_connected(test_graph_list)
    else:
        train_graph_list = utils.preprocess_graphs(train_graph_list)
        test_graph_list = utils.preprocess_graphs(test_graph_list)
    
    train_loader = utils.nx_to_pyg_dataloader(train_graph_list, batch_size=cfg.train.batch_size, shuffle=True)
    validate_loader = utils.nx_to_pyg_dataloader(test_graph_list, batch_size=cfg.train.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.policy=='pointnet':
        model = PointNet(forward_model=False).to(device)
    elif cfg.policy=='gnn':
        model = GNNSimulator(hidden_size=cfg.model.hidden_size, 
                             num_IN_layers=cfg.model.num_IN_layers,
                             forward_model=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion_reg = torch.nn.MSELoss()
    criterion_cls = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, min_lr=5e-4)

    train_loss_history = []
    val_loss_history = []
    best_loss = 1e9
    for epoch in range(1, cfg.train.epochs+1):   
        train_loss, train_loss_reg, train_loss_cls = train(model, optimizer, criterion_reg, criterion_cls, train_loader, epoch, cfg.train.lam, device)
        val_loss, val_loss_reg, val_loss_cls, accuracy  = validate(model, criterion_reg, criterion_cls, validate_loader, epoch, cfg.train.lam, device)
        if cfg.logging.wandb:
            wandb.log(
                {'train_loss': train_loss,
                'val_loss': val_loss,
                'train_loss_regression': train_loss_reg,
                'train_loss_classification': train_loss_cls,
                'val_loss_regression': val_loss_reg,
                'val_loss_classification': val_loss_cls,
                'val_accuracy': accuracy,}
            )
        if val_loss<best_loss:
            best_loss=val_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            print(f"Best model saved at epoch {epoch}")
        scheduler.step(best_loss)

        

