import os
from omegaconf import OmegaConf
import pickle
import utils
import copy
from tqdm import tqdm
import torch
from model import LearnedSimulator
import wandb
import time

def train(model, optimizer, criterion, train_loader, epoch, device):
    model.train()
    running_loss = 0
    max_distance_errors = []
    with tqdm(train_loader) as tepoch:
        for num_batch, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            batch.to(device)
            out = model(batch)
            prediction = batch.initial_position+out
            ground_truth = batch.final_position
            loss = criterion(prediction, ground_truth)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Metric: Maximum node distance error per tree
            distance_error = torch.norm(prediction-ground_truth, dim=-1)
            max_distance_error = torch.stack([torch.max(distance_error[batch.batch==i]) for i in range(batch.num_graphs)])
            max_distance_errors.append(max_distance_error)

            tepoch.set_postfix(loss=(running_loss/(num_batch+1)))
    train_loss = running_loss/len(train_loader)
    max_distance_errors = torch.cat(max_distance_errors)
    max_distance_error_mean = max_distance_errors.mean()
    max_distance_error_std = max_distance_errors.std()
    return train_loss, max_distance_error_mean, max_distance_error_std

def validate(model, criterion, data_loader, epoch, device):
    model.eval()
    running_loss = 0
    max_distance_errors = []
    with torch.no_grad():
        for num_batch, batch in enumerate(data_loader):
            batch.to(device)
            out = model(batch)
            prediction = batch.initial_position+out
            ground_truth = batch.final_position
            loss = criterion(prediction, ground_truth)
            running_loss += loss.item()

            # Metric: Maximum node distance error per tree
            distance_error = torch.norm(prediction-ground_truth, dim=-1)
            max_distance_error = torch.stack([torch.max(distance_error[batch.batch==i]) for i in range(batch.num_graphs)])
            max_distance_errors.append(max_distance_error)
        val_loss = running_loss/len(data_loader)
    max_distance_errors = torch.cat(max_distance_errors)
    max_distance_error_mean = max_distance_errors.mean()
    max_distance_error_std = max_distance_errors.std()
    return val_loss, max_distance_error_mean, max_distance_error_std

if __name__ == '__main__':

    cfg = OmegaConf.load('cfg/train_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))

    time_str = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/{time_str}_{cfg.name}"
    os.makedirs(output_dir, exist_ok=True)

    if cfg.logging.wandb:
        wandb.login()
        run = wandb.init(
            project="tree_is_all_you_need_v2",
            name=cfg.name,
            config=OmegaConf.to_container(cfg)
        )

    train_graph_list = []    
    for train_data in cfg.train_data_name:
        train_data_path = os.path.join(cfg.data_root, train_data)
        with open(train_data_path, 'rb') as f:
            train_graphs = pickle.load(f)
        train_graph_list += train_graphs
    test_graph_list = []
    for test_data in cfg.test_data_name:
        test_data_path = os.path.join(cfg.data_root, test_data)
        with open(test_data_path, 'rb') as f:
            test_graphs = pickle.load(f)
        test_graph_list += test_graphs[:len(test_graphs)//2]

    #train_graph_list, stats = utils.normalizer(train_graph_list)
    #test_graph_list, _ = utils.normalizer(test_graph_list, stats=stats)

    fully_connected = True
    if fully_connected:    
        train_loader = utils.nx_to_pyg_dataloader(utils.preprocess_graphs_to_fully_connected(train_graph_list), 
                                                  batch_size=cfg.train.batch_size)
        validate_loader = utils.nx_to_pyg_dataloader(utils.preprocess_graphs_to_fully_connected(test_graph_list), 
                                                     batch_size=cfg.train.batch_size, shuffle=False)
    else:
        train_loader = utils.nx_to_pyg_dataloader(utils.preprocess_graphs(train_graph_list), 
                                                batch_size=cfg.train.batch_size)
        validate_loader = utils.nx_to_pyg_dataloader(utils.preprocess_graphs(test_graph_list), 
                                                    batch_size=cfg.train.batch_size, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LearnedSimulator(hidden_size=cfg.model.hidden_size, num_IN_layers=cfg.model.num_IN_layers).to(device)
    #model = ForwardModel(graph_feat_size=3, node_feat_size=4, edge_feat_size=4).to(device)
    #model = IterativeForwardModel(graph_feat_size=3, node_feat_size=4, edge_feat_size=4).to(device)
    #model = BaselineMLP(input_size=51, hidden_size=256, output_size=36, layers=5).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, min_lr=5e-4)

    train_loss_history = []
    val_loss_history = []
    best_loss = 1e9
    
    for epoch in range(1, cfg.train.epochs+1):   
        train_loss, train_dist_err_mean, train_dist_error_std = train(model, optimizer, criterion, train_loader, epoch, device)
        val_loss, val_dist_err_mean, val_dist_err_std = validate(model, criterion, validate_loader, epoch, device)
        if cfg.logging.wandb:
            wandb.log(
                {'train_loss': train_loss,
                'val_loss': val_loss,
                'train_max_distance_error_mean': train_dist_err_mean,
                'train_max_distance_error_std': train_dist_error_std,
                'val_max_distance_error_mean': val_dist_err_mean,
                'val_max_distance_error_std': val_dist_err_std,}
            )
        if val_loss<best_loss:
            best_loss=val_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        scheduler.step(best_loss)

        

