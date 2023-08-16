import networkx as nx 
from urdfpy import URDF
from torch_geometric.loader import DataLoader
import numpy as np 
from tqdm.contrib import tzip
from tqdm import tqdm
from collections import defaultdict
from typing import Any, List, Optional, Union
import torch_geometric
import torch
from torch import Tensor
from torch_geometric.data import Data

def parse_urdf_graph(urdf_path):
    '''
    JOHN:
    Parses the urdf file to get the graph structure. 
    Returns a networkx DiGraph object
    '''
    robot = URDF.load(urdf_path)        
    edge_list = []
    num_nodes = 0
    for link in robot.links:
        if link.name.startswith('node'):
            num_nodes += 1
        if link.name.startswith('link') and ('y' in link.name):
            parent_node_idx = link.name.split('_')[1]
            child_node_idx = link.name.split('_')[3]
            edge_list.append([parent_node_idx, child_node_idx]) 
    
    DiG = nx.DiGraph()
    DiG.add_edges_from(edge_list)

    for edge in edge_list:
        for joint in robot.joints:
            joint_name = f'joint_{edge[0]}_x_{edge[1]}'
            if joint.name == joint_name:
                DiG.edges[edge]['stiffness'] = joint.dynamics.friction
                break
    return DiG

def nx_to_pyg_dataloader(graph_list, batch_size, shuffle=True):
    pyg_list = []
    for g in tqdm(graph_list):
        g = from_networkx(g)
        pyg_list.append(g)
    data_loader = DataLoader(pyg_list, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def tensor_to_nx(node_pos, edge_index, edge_stiffness):
    '''
    Converts node_pos and edge_index to networkx DiGraph object
    '''
    edge_list = []
    for edge in edge_index:
        edge_list.append((edge[0], edge[1]))
    DiG = nx.DiGraph()
    DiG.add_edges_from(edge_list)
    for i, node in enumerate(node_pos):
        DiG.nodes[i]['position'] = node
    # Add edge length to edges
    for i, edge in enumerate(DiG.edges):
        DiG.edges[edge]['length'] = np.linalg.norm(DiG.nodes[edge[1]]['position'] - DiG.nodes[edge[0]]['position'])
        DiG.edges[edge]['stiffness'] = edge_stiffness[i]
    return DiG


# Preprocess list of graphs to list of fully connected graphs
def preprocess_graphs_to_fully_connected(graph_list):
    graph_list_new = [g.copy() for g in graph_list]
    for g, g_new in tzip(graph_list, graph_list_new):
        for parent in g.nodes:
            for child in g.nodes:
                if parent != child:
                    if g.has_edge(parent, child):
                        pass
                    else:
                        # Add edge and populate features
                        g_new.add_edge(parent, child)
                        g_new.edges[parent, child]['initial_edge_delta'] = g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']
                        g_new.edges[parent, child]['final_edge_delta'] = g.nodes[child]['final_position'] - g.nodes[parent]['final_position']
                        # if child is descendant of parent, then parent2child is 1, else -1
                        g_new.edges[parent, child]['parent2child'] = 1 if nx.has_path(g, parent, child) else -1
                        g_new.edges[parent, child]['branch'] = 1 if g.has_edge(child, parent) else 0
                        g_new.edges[parent, child]['stiffness'] = g.edges[child, parent]['stiffness'] if g.has_edge(child, parent) else 0
                    g_new.edges[parent, child]['length'] = np.linalg.norm(g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']) 
    return graph_list_new

# Preprocess list of graphs to add edges from child to parent
def preprocess_graphs(graph_list):
    graph_list_new = [g.copy() for g in graph_list]
    for g, g_new in tzip(graph_list, graph_list_new):
        for parent in g.nodes:
            for child in g.nodes:
                if parent != child:
                    if g.has_edge(parent, child):
                        # Add edge feature to indicate that this edge is a branch
                        g_new.edges[parent, child]['length'] = np.linalg.norm(g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']) 

                        # Add edge and populate features
                        g_new.add_edge(child, parent)
                        g_new.edges[child, parent]['initial_edge_delta'] = g.nodes[parent]['initial_position'] - g.nodes[child]['initial_position']
                        g_new.edges[child, parent]['final_edge_delta'] = g.nodes[parent]['final_position'] - g.nodes[child]['final_position']
                        # if child is descendant of parent, then parent2child is 1, else -1
                        g_new.edges[child, parent]['parent2child'] = -1 
                        g_new.edges[child, parent]['branch'] = 1
                        g_new.edges[child, parent]['stiffness'] = g.edges[parent, child]['stiffness']
                        g_new.edges[child, parent]['length'] = np.linalg.norm(g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']) 
    return graph_list_new

def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], all]] = None,
    group_edge_attrs: Optional[Union[List[str], all]] = None,
) -> 'torch_geometric.data.Data':
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.

    Examples:

        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> g = to_networkx(data)
        >>> # A `Data` object is returned
        >>> from_networkx(g)
        Data(edge_index=[2, 6], num_nodes=4)
    """

    G = G.to_directed() if not nx.is_directed(G) else G

    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = src
        edge_index[1, i] = dst

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i in range(G.number_of_nodes()):
        feat_dict = G.nodes[i]
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for edge in edge_index.T:
        feat_dict = G.edges[edge[0].item(), edge[1].item()]
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


def normalizer(graph_list, stats=None):
    # Get keys of node features
    node_feat_keys = graph_list[0].nodes[next(iter(graph_list[0].nodes))].keys()
    # Get keys of edge features
    edge_feat_keys = graph_list[0].edges[next(iter(graph_list[0].edges))].keys()
    # Get keys of graph features
    graph_feat_keys = graph_list[0].graph.keys()

    if stats is None:
        stats = {'node': {}, 'edge': {}, 'graph': {}}
        for key in node_feat_keys:
            data = []
            for graph in graph_list:
                for node_idx in graph.nodes:
                    data.append(graph.nodes[node_idx][key])
            mean = np.array(data, dtype=float).mean(axis=0)
            std = np.array(data, dtype=float).std(axis=0)
            stats['node'][key] = {'mean': mean, 'std': std}
            for graph in graph_list:
                for node_idx in graph.nodes:
                    graph.nodes[node_idx][key] = ((graph.nodes[node_idx][key] - mean) / std).astype(np.float32)

            print(f'Node feature {key} mean: {mean} std: {std}')
        
        for key in edge_feat_keys:
            data = []
            for graph in graph_list:
                for edge in graph.edges:
                    data.append(graph.edges[edge][key])
            mean = np.array(data, dtype=float).mean(axis=0)
            std = np.array(data, dtype=float).std(axis=0)
            stats['edge'][key] = {'mean': mean, 'std': std}
            for graph in graph_list:
                for edge in graph.edges:
                    graph.edges[edge][key] = ((graph.edges[edge][key] - mean) / std).astype(np.float32)
            print(f'Edge feature {key} mean: {mean} std: {std}')
        
        for key in graph_feat_keys:
            data = []
            for graph in graph_list:
                data.append(graph.graph[key])
            mean = np.array(data, dtype=float).mean(axis=0)
            std = np.array(data, dtype=float).std(axis=0)
            stats['graph'][key] = {'mean': mean, 'std': std}
            for graph in graph_list:
                graph.graph[key] = ((graph.graph[key] - mean) / std).astype(np.float32)
            print(f'Graph feature {key} mean: {mean} std: {std}')
        return graph_list, stats

    else:
        for key in node_feat_keys:
            mean = stats['node'][key]['mean']
            std = stats['node'][key]['std']
            for graph in graph_list:
                for node_idx in graph.nodes:
                    graph.nodes[node_idx][key] = ((graph.nodes[node_idx][key] - mean) / std).astype(np.float32)
        for key in edge_feat_keys:
            mean = stats['edge'][key]['mean']
            std = stats['edge'][key]['std']
            for graph in graph_list:
                for edge in graph.edges:
                    graph.edges[edge][key] = ((graph.edges[edge][key] - mean) / std).astype(np.float32)
        for key in graph_feat_keys:
            mean = stats['graph'][key]['mean']
            std = stats['graph'][key]['std']
            for graph in graph_list:
                graph.graph[key] = ((graph.graph[key] - mean) / std).astype(np.float32)
        return graph_list, stats