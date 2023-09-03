import torch
import torch_scatter
from torch_geometric.nn import MessagePassing

class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class InteractionNetwork(MessagePassing):
    """Interaction Network as proposed in this paper: 
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 3+3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2+3, hidden_size, hidden_size, layers)

    def forward(self, x, edge_index, edge_feature, graph_feature, batch):
        # Repeat graph feature so that it has same first dimension as edge_feature
        graph_feature = graph_feature.reshape(-1, 3)
        # Compute number of nodes and edges per graph using batch info
        num_nodes_per_graph = torch.bincount(batch)
        num_edges_per_graph = torch.bincount(batch[edge_index[0]])
        # Repeat graph_feature for each node and edge according to batch
        graph_feature_node = torch.repeat_interleave(graph_feature, num_nodes_per_graph, dim=0)
        graph_feature_edge = torch.repeat_interleave(graph_feature, num_edges_per_graph, dim=0)
        
        # Compute edge_features and aggregate messages        
        edge_out, aggr = self.propagate(edge_index, x=(x,x), edge_feature=edge_feature, graph_feature=graph_feature_edge)
        node_out = self.lin_node(torch.cat((x, aggr, graph_feature_node), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature, graph_feature):
        x = torch.cat((x_i, x_j, edge_feature, graph_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)


class _GNNSimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        hidden_size,
        num_IN_layers, # number of GNN layers
        forward_model,
    ):
        super().__init__()
        self.forward_model = forward_model
        self.hidden_size = hidden_size
        if self.forward_model:
            self.node_in = MLP(1, hidden_size, hidden_size, 3) 
        else:
            self.node_in = MLP(3, hidden_size, hidden_size, 3) 
            self.node_selector = MLP(hidden_size, hidden_size, 1, 3)         
        self.edge_in = MLP(7, hidden_size, hidden_size, 3) 
        self.node_action = MLP(hidden_size, hidden_size, 3, 3) 

        self.num_IN_layers = num_IN_layers
        self.IN_layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(num_IN_layers)])

    def forward(self, data):
        if self.forward_model:
            x = data.contact_node.unsqueeze(-1).float()
            graph_feature = data.contact_force
        else:
            x = data.final_position - data.initial_position
            graph_feature = torch.zeros_like(data.contact_force)

        edge_attr = torch.cat((
                            data.initial_edge_delta, 
                            data.length.unsqueeze(-1),
                            data.parent2child.unsqueeze(-1),
                            data.branch.unsqueeze(-1),
                            data.stiffness.unsqueeze(-1),                            
                            ), dim=-1).float()
        node_feature = self.node_in(x)
        edge_feature = self.edge_in(edge_attr)

        # stack of GNN layers
        for i in range(self.num_IN_layers):
            node_feature, edge_feature = self.IN_layers[i](x=node_feature, 
                                                        edge_index=data.edge_index, 
                                                        edge_feature=edge_feature, 
                                                        graph_feature=graph_feature, 
                                                        batch=data.batch)
        action = self.node_action(node_feature)
        if self.forward_model:
            return action
        else:        
            node_selection_logits = self.node_selector(node_feature)
            return node_selection_logits.flatten(), action

class GNNSimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        hidden_size,
        num_IN_layers, # number of GNN layers
        forward_model,
    ):
        super().__init__()
        self.forward_model = forward_model
        self.hidden_size = hidden_size
        if self.forward_model:
            self.node_in = MLP(1, hidden_size, hidden_size, 3) 
        else:
            self.node_in = MLP(3, hidden_size, hidden_size, 3) 
            self.node_selector = MLP(hidden_size, hidden_size, 1, 3)         
        self.edge_in = MLP(6, hidden_size, hidden_size, 3) 
        self.node_action = MLP(hidden_size, hidden_size, 3, 3) 

        self.num_IN_layers = num_IN_layers
        self.IN_layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(num_IN_layers)])

    def forward(self, data):
        if self.forward_model:
            x = data.contact_node.unsqueeze(-1).float()
            graph_feature = data.contact_force
        else:
            x = data.final_position - data.initial_position
            graph_feature = torch.zeros_like(data.contact_force)

        edge_attr = torch.cat((
                            data.initial_edge_delta, 
                            data.length.unsqueeze(-1),
                            data.parent2child.unsqueeze(-1),
                            ###data.branch.unsqueeze(-1),
                            data.stiffness.unsqueeze(-1),                            
                            ), dim=-1).float()
        node_feature = self.node_in(x)
        edge_feature = self.edge_in(edge_attr)

        # stack of GNN layers
        for i in range(self.num_IN_layers):
            node_feature, edge_feature = self.IN_layers[i](x=node_feature, 
                                                        edge_index=data.edge_index, 
                                                        edge_feature=edge_feature, 
                                                        graph_feature=graph_feature, 
                                                        batch=data.batch)
        action = self.node_action(node_feature)
        if self.forward_model:
            return action
        else:        
            node_selection_logits = self.node_selector(node_feature)
            return node_selection_logits.flatten(), action

class PointNet(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        forward_model=True,
    ):
        super().__init__()
        self.forward_model = forward_model
        input_size = 4 if self.forward_model else 3
        self.local_feature_mlp = MLP(input_size, 64, 64, 2) 
        self.global_feature_mlp = MLP(64, 128, 1024, 3) 
        # MAX POOL to get Global Feature
        # Concatenate Global Feature with Local Feature
        self.point_feature_mlp = MLP(1088, 512, 128, 3)
        self.force_regressor = MLP(128, 128, 3, 3)
        if not self.forward_model: 
            self.node_selector = MLP(128, 128, 1, 3) 
        
    def forward(self, data):
        # Repeat graph feature so that it corresponds to each node according to batch
        if self.forward_model:
            graph_feature = data.contact_force.reshape(-1, 3)
            # Compute number of nodes and edges per graph using batch info
            num_nodes_per_graph = torch.bincount(data.batch)
            # Repeat graph_feature for each node and edge according to batch
            graph_feature_node = torch.repeat_interleave(graph_feature, num_nodes_per_graph, dim=0)
            # Concatenate initial position, contact node, and graph feature
            #x = torch.cat((data.initial_position, data.contact_node.unsqueeze(-1), graph_feature_node), dim=-1) # Maybe take initial_position out for fairness
            x = torch.cat((data.contact_node.unsqueeze(-1), graph_feature_node), dim=-1) # Maybe take initial_position out for fairness
        else:
            #x = torch.cat((data.initial_position, data.final_position), dim=-1)
            x = data.final_position - data.initial_position
        local_feature = self.local_feature_mlp(x)
        global_feature = self.global_feature_mlp(local_feature)
        # MAX POOL to get Global Feature per graph
        global_feature = torch_scatter.scatter_max(global_feature, data.batch, dim=0)[0]
        # Repeat global feature for each node according to batch
        global_feature = torch.repeat_interleave(global_feature, torch.bincount(data.batch), dim=0)
        # Concatenate Global Feature with Local Feature
        point_feature = torch.cat((local_feature, global_feature), dim=-1)
        # MLP to get point features
        point_feature = self.point_feature_mlp(point_feature)
        # MLP to get node position predictions
        contact_force = self.force_regressor(point_feature)
        if self.forward_model:
            return contact_force
        else:
            node_selection_logits = self.node_selector(point_feature)
            return node_selection_logits.flatten(), contact_force
        
class HeuristicBaseline(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode # random, greedy, or root

    def forward(self, data):
        
        # Compute contact trajectory first based on the actual displacement
        node_displacement = data.final_position - data.initial_position
        # Subtract the gripper palm protrusion
        palm_protrusion = 0.02
        # Get the distance of each vector
        node_displacement_norm = torch.norm(node_displacement, dim=-1)
        # Get the ratio of palm protrusion to contact force
        palm_protrusion_ratio = palm_protrusion/node_displacement_norm
        contact_trajectory = node_displacement*(1-palm_protrusion_ratio.unsqueeze(-1))

        # Node selection methods
        '''
        # OPTION 1: Just get from ground truth
        node_selection_onehot = data.contact_node
        '''
        if self.mode=='random':
            # OPTION 2: Randomly select a node to be the contact node except the root node
            # For each batch, randomly select a node to be the contact node except the root node using batch.data information
            node_selection_onehot = torch.zeros_like(data.contact_node)
            for i in range(torch.max(data.batch)+1):
                # Get the indices of nodes that are not the root node
                idx = torch.where(data.batch==i)[0]
                idx = idx[idx!=0]
                # Randomly select one node to be the contact node
                node_selection_onehot[idx[torch.randint(0, len(idx), (1,))]] = 1
        elif self.mode=='greedy':
            # OPTION 3: Order it from the largest displaced node to the smallest
            node_selection_onehot = node_displacement_norm    
        elif self.mode=='root':
            # OPTION 4: Always select the root node as the contact node. This is only for random target mode
            node_selection_onehot = data.contact_node
        return node_selection_onehot, contact_trajectory

if __name__=='__main__':
    simulator = GNNSimulator()
    print(simulator)