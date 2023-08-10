import networkx as nx 
from urdfpy import URDF
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import numpy as np 
from tqdm.contrib import tzip
from tqdm import tqdm

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
    return DiG

def nx_to_pyg_dataloader(graph_list, batch_size, shuffle=True):
    pyg_list = []
    for g in tqdm(graph_list):
        g = from_networkx(g)
        pyg_list.append(g)
    data_loader = DataLoader(pyg_list, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def tensor_to_nx(node_pos, edge_index):
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
    for edge in DiG.edges:
        DiG.edges[edge]['length'] = np.linalg.norm(DiG.nodes[edge[1]]['position'] - DiG.nodes[edge[0]]['position'])
    return DiG

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

# Preprocess list of graphs to list of fully connected graphs
def preprocess_graphs_to_fully_connected(graph_list):
    graph_list_new = [g.copy() for g in graph_list]
    for g, g_new in tzip(graph_list, graph_list_new):
        for parent in g.nodes:
            for child in g.nodes:
                if parent != child:
                    if g_new.has_edge(parent, child):
                        # Add edge feature to indicate that this edge is a branch
                        g_new.edges[parent, child]['branch'] = 1
                    else:
                        # Add edge and populate features
                        g_new.add_edge(parent, child)
                        g_new.edges[parent, child]['initial_edge_delta'] = g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']
                        g_new.edges[parent, child]['final_edge_delta'] = g.nodes[child]['final_position'] - g.nodes[parent]['final_position']
                        # if child is descendant of parent, then parent2child is 1, else -1
                        g_new.edges[parent, child]['parent2child'] = 1 if nx.has_path(g, parent, child) else -1
                        g_new.edges[parent, child]['branch'] = 0
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
                        g_new.edges[parent, child]['branch'] = 1
                        g_new.edges[parent, child]['length'] = np.linalg.norm(g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']) 

                        # Add edge and populate features
                        g_new.add_edge(child, parent)
                        g_new.edges[child, parent]['initial_edge_delta'] = g.nodes[parent]['initial_position'] - g.nodes[child]['initial_position']
                        g_new.edges[child, parent]['final_edge_delta'] = g.nodes[parent]['final_position'] - g.nodes[child]['final_position']
                        # if child is descendant of parent, then parent2child is 1, else -1
                        g_new.edges[child, parent]['parent2child'] = -1 
                        g_new.edges[child, parent]['branch'] = 1
                        g_new.edges[child, parent]['length'] = np.linalg.norm(g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']) 
    return graph_list_new

    
