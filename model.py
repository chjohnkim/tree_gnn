import math
import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import MessagePassing


class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

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
        hidden_size=128,
        num_IN_layers=30, # number of GNN layers
        dim=3, # dimension of the world, typical 2D or 3D
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_in = torch.nn.Sequential(
            torch.nn.Linear(4, self.hidden_size),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(self.hidden_size,self.hidden_size),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        ) 

        self.edge_in = torch.nn.Sequential(
            torch.nn.Linear(2, self.hidden_size),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(self.hidden_size,self.hidden_size),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        ) 

        self.node_out = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(self.hidden_size,self.hidden_size),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(self.hidden_size, dim),
        ) 

        self.num_IN_layers = num_IN_layers
        self.IN_layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(num_IN_layers)])

    def forward(self, data):
        # pre-processing
        x = torch.cat((data.initial_position, data.contact_node.unsqueeze(-1)), dim=-1)
        #edge_attr = torch.cat((data.initial_edge_delta, 
        #                       data.length.unsqueeze(-1),
        #                       data.parent2child.unsqueeze(-1),
        #                       data.branch.unsqueeze(-1)
        #                       ), dim=-1)
        edge_attr = torch.cat((data.parent2child.unsqueeze(-1),
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

if __name__=='__main__':
    simulator = LearnedSimulator()
    print(simulator)


# class BaselineMLP(torch.nn.Module):
#     """Multi-Layer perceptron"""
#     def __init__(self, input_size, hidden_size, output_size, layers):
#         super().__init__()
#         self.layers = torch.nn.ModuleList()
#         for i in range(layers):
#             self.layers.append(torch.nn.Linear(
#                 input_size if i == 0 else hidden_size,
#                 output_size if i == layers - 1 else hidden_size,
#             ))
#             if i != layers - 1:
#                 self.layers.append(torch.nn.ReLU())

#     def forward(self, data):
#         x = torch.cat((data.initial_position, data.contact_node.unsqueeze(-1)), dim=-1)
#         x = x.reshape(-1, 4*12)
#         x = torch.cat((x, data.contact_force.reshape(-1, 3)), dim=-1)
#         for layer in self.layers:
#             x = layer(x)
#         x = x.reshape(-1, 3)
#         return x
    
# class ForwardModel(torch.nn.Module):
#     """Graph Network-based Simulators(GNS)"""
#     def __init__(
#         self, graph_feat_size, node_feat_size, edge_feat_size):
#         super().__init__()
#         self.GN1 = GNBlock(graph_feat_size, node_feat_size, edge_feat_size)
#         self.GN2 = GNBlock(graph_feat_size*2, node_feat_size*2, edge_feat_size*2)

#         self.node_out = torch.nn.Sequential(
#             torch.nn.Linear(node_feat_size*2, node_feat_size*2),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(node_feat_size*2, node_feat_size*2),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(node_feat_size*2, 3)
#         )

#     def forward(self, data):
#         node_feature_in = torch.cat((data.initial_position, data.contact_node.unsqueeze(-1)), dim=-1)
#         edge_feature_in = torch.cat((data.initial_edge_delta, data.parent2child.unsqueeze(-1)), dim=-1)
#         graph_feature_in = data.contact_force
#         # Repeat graph feature so that it has same first dimension as edge_feature
#         graph_feature_in = graph_feature_in.reshape(-1, 3)
#         edge_index = data.edge_index

#         node_feature = torch.clone(node_feature_in)
#         edge_feature = torch.clone(edge_feature_in)
#         graph_feature = torch.clone(graph_feature_in)

#         node_feature, edge_feature, graph_feature = self.GN1(node_feature, edge_feature, graph_feature, edge_index)

#         # Concatenate features
#         node_feature = torch.cat((node_feature, node_feature_in), dim=-1)
#         edge_feature = torch.cat((edge_feature, edge_feature_in), dim=-1)
#         graph_feature = torch.cat((graph_feature, graph_feature_in), dim=-1)

#         node_feature, edge_feature, graph_feature = self.GN2(node_feature, edge_feature, graph_feature, edge_index)

#         out = self.node_out(node_feature)
#         return out



# class GNBlock(MessagePassing):
#     def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
#         super().__init__()
#         self.graph_feat_size = graph_feat_size
#         self.node_feat_size = node_feat_size
#         self.edge_feat_size = edge_feat_size
#         self.hidden_size = 256
#         self.mlp_edge = torch.nn.Sequential(
#             torch.nn.Linear(graph_feat_size + 2 * node_feat_size + edge_feat_size, self.hidden_size),
#             torch.nn.ReLU(inplace = True),
#             torch.nn.Linear(self.hidden_size,self.hidden_size),
#             torch.nn.ReLU(inplace = True),
#             torch.nn.Linear(self.hidden_size, self.hidden_size),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(self.hidden_size,edge_feat_size),
#         ) 

#         self.mlp_node = torch.nn.Sequential(
#             torch.nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, self.hidden_size),
#             torch.nn.ReLU(inplace = True),
#             torch.nn.Linear(self.hidden_size, self.hidden_size),
#             torch.nn.ReLU(inplace = True),
#             torch.nn.Linear(self.hidden_size, self.hidden_size),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(self.hidden_size, node_feat_size),
#         )

#         self.mlp_graph = torch.nn.Sequential(
#             torch.nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, self.hidden_size),
#             torch.nn.ReLU(inplace = True),
#             torch.nn.Linear(self.hidden_size, self.hidden_size),
#             torch.nn.ReLU(inplace = True),
#             torch.nn.Linear(self.hidden_size, self.hidden_size),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(self.hidden_size, graph_feat_size),
#         )

#     def forward(self, node_feature, edge_feature, graph_feature, edge_index):
   
#         #====================================================================================
#         num_edges_per_graph = edge_index.shape[1] // graph_feature.shape[0]
#         num_nodes_per_graph = node_feature.shape[0] // graph_feature.shape[0]
#         graph_feature_edge = graph_feature.repeat_interleave(num_edges_per_graph, dim=0)
#         graph_feature_node = graph_feature.repeat_interleave(num_nodes_per_graph, dim=0)
        
#         # Compute edge_features and aggregate messages
#         edge_feature, aggr = self.propagate(edge_index, x=(node_feature,node_feature), edge_feature=edge_feature, graph_feature=graph_feature_edge)
#         # Compute node-wise features
#         node_feature = self.mlp_node(torch.cat((node_feature, aggr, graph_feature_node), dim=-1))
        
#         # Aggregate all edges per graph
#         edge_feature_aggr = edge_feature.reshape(-1, num_edges_per_graph, self.edge_feat_size)
#         edge_feature_aggr = edge_feature_aggr.mean(dim=1)
#         # Aggregate all nodes per graph
#         node_feature_aggr = node_feature.reshape(-1, num_nodes_per_graph, self.node_feat_size)
#         node_feature_aggr = node_feature_aggr.mean(dim=1)
#         # Compute global features
#         graph_feature = self.mlp_graph(torch.cat((node_feature_aggr, edge_feature_aggr, graph_feature), dim=-1))
#         return node_feature, edge_feature, graph_feature

#     def message(self, x_i, x_j, edge_feature, graph_feature):
#         new_edge_feature = torch.cat((x_i, x_j, edge_feature, graph_feature), dim=-1)
#         new_edge_feature = self.mlp_edge(new_edge_feature)
#         return new_edge_feature

#     def aggregate(self, inputs, index, dim_size=None):
#         out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
#         return (inputs, out)
    

# class IterativeForwardModel(torch.nn.Module):
#     """Graph Network-based Simulators(GNS)"""
#     def __init__(
#         self, graph_feat_size, node_feat_size, edge_feat_size):
#         super().__init__()


#         self.n_mp_layers = 6
#         self.layers = torch.nn.ModuleList([GNBlock(graph_feat_size, node_feat_size, edge_feat_size) for _ in range(self.n_mp_layers)])

#         self.node_out = torch.nn.Sequential(
#             torch.nn.Linear(node_feat_size, 128),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(128, 128),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(128, 3)
#         )

#     def forward(self, data):
#         node_feature_in = torch.cat((data.initial_position, data.contact_node.unsqueeze(-1)), dim=-1)
#         edge_feature_in = torch.cat((data.initial_edge_delta, data.parent2child.unsqueeze(-1)), dim=-1)
#         graph_feature_in = data.contact_force
#         # Repeat graph feature so that it has same first dimension as edge_feature
#         graph_feature_in = graph_feature_in.reshape(-1, 3)
#         edge_index = data.edge_index

#         node_feature = torch.clone(node_feature_in)
#         edge_feature = torch.clone(edge_feature_in)
#         graph_feature = torch.clone(graph_feature_in)

#         for i in range(self.n_mp_layers):
#             node_feature, edge_feature, graph_feature = self.layers[i](node_feature, edge_feature, graph_feature, edge_index)
#         # post-processing
#         out = self.node_out(node_feature)
#         return out
