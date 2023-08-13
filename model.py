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

class LearnedSimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        hidden_size,
        num_IN_layers, # number of GNN layers
        dim=3, # dimension of the world, typical 2D or 3D
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_in = MLP(4, hidden_size, hidden_size, 3) 
        self.edge_in = MLP(2, hidden_size, hidden_size, 3) 
        self.node_out = MLP(hidden_size, hidden_size, dim, 3) 

        self.num_IN_layers = num_IN_layers
        self.IN_layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(num_IN_layers)])

    def forward(self, data):
        # pre-processing
        x = torch.cat((data.initial_position, data.contact_node.unsqueeze(-1)), dim=-1)
        edge_attr = torch.cat((
                            #data.initial_edge_delta, 
                            #data.length.unsqueeze(-1),
                            data.parent2child.unsqueeze(-1),
                            data.branch.unsqueeze(-1)
                            ), dim=-1).float()
        node_feature = self.node_in(x)
        edge_feature = self.edge_in(edge_attr)

        # stack of GNN layers
        for i in range(self.num_IN_layers):
            node_feature, edge_feature = self.IN_layers[i](x=node_feature, 
                                                        edge_index=data.edge_index, 
                                                        edge_feature=edge_feature, 
                                                        graph_feature=data.contact_force, 
                                                        batch=data.batch)
        out = self.node_out(node_feature)
        return out

class LearnedPolicy(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        hidden_size,
        num_IN_layers, # number of GNN layers
        dim=3, # dimension of the world, typical 2D or 3D
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_in = MLP(6, hidden_size, hidden_size, 3) 
        self.edge_in = MLP(2, hidden_size, hidden_size, 3) 
        self.node_selector = MLP(hidden_size, hidden_size, 1, 3) 
        self.force_regressor = MLP(hidden_size, hidden_size, 3, 3) 

        self.num_IN_layers = num_IN_layers
        self.IN_layers_node = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(num_IN_layers)])
        self.IN_layers_force = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(num_IN_layers)])

    def forward(self, data):
        # pre-processing
        x = torch.cat((data.initial_position, 
                       data.final_position,
                       ), dim=-1)
        edge_attr = torch.cat((
                            #data.initial_edge_delta, 
                            #data.length.unsqueeze(-1),
                            data.parent2child.unsqueeze(-1),
                            data.branch.unsqueeze(-1)
                            ), dim=-1).float()
        node_feature = self.node_in(x)
        edge_feature = self.edge_in(edge_attr)
        graph_feature = torch.zeros_like(data.contact_force)

        # stack of GNN layers
        for i in range(self.num_IN_layers):
            node_feature, edge_feature = self.IN_layers_node[i](x=node_feature, 
                                                        edge_index=data.edge_index, 
                                                        edge_feature=edge_feature, 
                                                        graph_feature=graph_feature, 
                                                        batch=data.batch)
        # Predict nodes after the first interaction network layer
        node_selection_logits = self.node_selector(node_feature)
        
        for i in range(self.num_IN_layers):
            node_feature, edge_feature = self.IN_layers_force[i](x=node_feature, 
                                                        edge_index=data.edge_index, 
                                                        edge_feature=edge_feature, 
                                                        graph_feature=graph_feature, 
                                                        batch=data.batch)
        # Predict contact force after the last interaction network layer
        contact_force = self.force_regressor(node_feature)

        # Apply softmax on node_selection based on data.batch
        # Index into contact_force using contact_node_idx
        #_, contact_node_idx = torch_scatter.scatter_max(node_selection_logits, data.batch, dim=0)
        #contact_force = contact_force[contact_node_idx].reshape(-1, 3)
        return node_selection_logits.flatten(), contact_force

if __name__=='__main__':
    simulator = LearnedSimulator()
    print(simulator)