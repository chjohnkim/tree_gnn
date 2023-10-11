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
    
    for link in robot.links:
        if link.name.startswith('link') and ('z' in link.name):
            parent_node_idx = link.name.split('_')[1]
            child_node_idx = link.name.split('_')[3]
            DiG.edges[parent_node_idx, child_node_idx]['radius'] = link.collisions[0].geometry.cylinder.radius
            DiG.edges[parent_node_idx, child_node_idx]['length'] = link.collisions[0].geometry.cylinder.length
            
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

def tensor_to_nx(node_pos, edge_index, edge_stiffness, edge_radius):
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
        DiG.edges[edge]['radius'] = edge_radius[i]
    return DiG

def preprocess_graphs_to_single_target(graph_list):
    # For each graph, randomly select a node excluding root node
    # Get the displacement between initial and final configuration and set it as a graph feature for target node displacement
    # Add node feature to randomly selected node to indicate that it is the target node. 1 if target node, 0 otherwise
    graph_list_new = [g.copy() for g in graph_list]
    for g, g_new in tzip(graph_list, graph_list_new):
        node_displacements = []
        for node_idx in range(g.number_of_nodes()):
            node_displacements.append(g.nodes[node_idx]['final_position'] - g.nodes[node_idx]['initial_position'])
        node_distances = np.linalg.norm(node_displacements, axis=1)
        target_node_idx = np.argmax(node_distances)
        g_new.graph['target_node_delta'] = node_displacements[target_node_idx]
        for node_idx in g.nodes:
            g_new.nodes[node_idx]['target_node'] = 0
        g_new.nodes[target_node_idx]['target_node'] = 1
    return graph_list_new

# Preprocess list of graphs to list of fully connected graphs
def preprocess_graphs_to_fully_connected(graph_list):
    graph_list_new = [g.copy() for g in graph_list]
    for g, g_new in tzip(graph_list, graph_list_new):
        for parent in g.nodes:
            for child in g.nodes:
                if parent != child:
                    # If edge already exists, do nothing
                    if g.has_edge(parent, child):
                        pass
                    # If edge does not exist, add edge and populate features
                    else:
                        # Add edge and populate features
                        g_new.add_edge(parent, child)
                        g_new.edges[parent, child]['initial_edge_delta'] = g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']
                        g_new.edges[parent, child]['final_edge_delta'] = g.nodes[child]['final_position'] - g.nodes[parent]['final_position']
                        # if child is descendant of parent, then parent2child is 1, else -1
                        g_new.edges[parent, child]['parent2child'] = 1 if nx.has_path(g, parent, child) else -1
                        g_new.edges[parent, child]['branch'] = 1 if g.has_edge(child, parent) else 0
                        g_new.edges[parent, child]['stiffness'] = g.edges[child, parent]['stiffness'] if g.has_edge(child, parent) else 0
                        g_new.edges[parent, child]['radius'] = 0 
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
                        #g_new.edges[parent, child]['length'] = np.linalg.norm(g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']) 
                        # Add edge and populate features
                        g_new.add_edge(child, parent)
                        g_new.edges[child, parent]['initial_edge_delta'] = g.nodes[parent]['initial_position'] - g.nodes[child]['initial_position']
                        g_new.edges[child, parent]['final_edge_delta'] = g.nodes[parent]['final_position'] - g.nodes[child]['final_position']
                        # if child is descendant of parent, then parent2child is 1, else -1
                        g_new.edges[child, parent]['parent2child'] = -1 
                        g_new.edges[child, parent]['branch'] = 1
                        g_new.edges[child, parent]['stiffness'] = g.edges[parent, child]['stiffness']
                        g_new.edges[child, parent]['radius'] = g.edges[parent, child]['radius']
                        g_new.edges[child, parent]['length'] = g.edges[parent, child]['length'] 
    return graph_list_new

def set_random_target_configuration(graph_list):
    '''
    For each graph, randomly select a node exluding the root node. 
    Find all descendents of the randomly selected node.
    Arbitrarily set a reasonable displacement angle for the subtree rooted at the randomly selected node.
    Set that as the new target configuration.
    '''
    graph_list_new = [g.copy() for g in graph_list]
    for g, g_new in tzip(graph_list, graph_list_new):
        random_root_node = np.random.randint(1, g.number_of_nodes())
        parent_node = list(g.predecessors(random_root_node))[0]
        # Get subtree nodes rooted at the randomly selected node
        sub_tree_nodes = [random_root_node] + list(nx.descendants(g, random_root_node))
        # Theta: Random angle between 0 and 20 degrees
        theta = np.random.uniform(15, 30) * np.pi / 180
        primary_vector = g.nodes[random_root_node]['initial_position'] - g.nodes[parent_node]['initial_position']
        primary_vector_normalized = primary_vector / np.linalg.norm(primary_vector)
        # Random orthogonal vector to primary_vector
        secondary_vector = np.random.randn(3)
        secondary_vector = secondary_vector - np.dot(secondary_vector, primary_vector_normalized) * primary_vector_normalized
        secondary_vector_normalized = secondary_vector / np.linalg.norm(secondary_vector)
        # Rotate primary_vector by theta degrees around secondary_vector
        rotation_matrix = np.array([[np.cos(theta) + secondary_vector_normalized[0]**2 * (1 - np.cos(theta)),
                                    secondary_vector_normalized[0] * secondary_vector_normalized[1] * (1 - np.cos(theta)) - secondary_vector_normalized[2] * np.sin(theta),
                                    secondary_vector_normalized[0] * secondary_vector_normalized[2] * (1 - np.cos(theta)) + secondary_vector_normalized[1] * np.sin(theta)],
                                    [secondary_vector_normalized[1] * secondary_vector_normalized[0] * (1 - np.cos(theta)) + secondary_vector_normalized[2] * np.sin(theta),
                                    np.cos(theta) + secondary_vector_normalized[1]**2 * (1 - np.cos(theta)),
                                    secondary_vector_normalized[1] * secondary_vector_normalized[2] * (1 - np.cos(theta)) - secondary_vector_normalized[0] * np.sin(theta)],
                                    [secondary_vector_normalized[2] * secondary_vector_normalized[0] * (1 - np.cos(theta)) - secondary_vector_normalized[1] * np.sin(theta),
                                    secondary_vector_normalized[2] * secondary_vector_normalized[1] * (1 - np.cos(theta)) + secondary_vector_normalized[0] * np.sin(theta),
                                    np.cos(theta) + secondary_vector_normalized[2]**2 * (1 - np.cos(theta))]])        
        for node_idx in g.nodes:
            if node_idx in sub_tree_nodes:
                # If node is in subtree, rotate it by theta degrees around secondary_vector
                g_new.nodes[node_idx]['final_position'] = g.nodes[node_idx]['initial_position'] - g.nodes[parent_node]['initial_position']
                g_new.nodes[node_idx]['final_position'] = np.dot(rotation_matrix, g_new.nodes[node_idx]['final_position'])
                g_new.nodes[node_idx]['final_position'] = g_new.nodes[node_idx]['final_position'] + g.nodes[parent_node]['initial_position']
                g_new.nodes[node_idx]['final_position'] = g_new.nodes[node_idx]['final_position'].astype(np.float32)
            else:
                g_new.nodes[node_idx]['final_position'] = g.nodes[node_idx]['initial_position']
            # Set the contact_node to be node 0
            g_new.nodes[node_idx]['contact_node'] = 1 if node_idx == random_root_node else 0
        for edge in g.edges:
            g_new.edges[edge]['final_edge_delta'] = g_new.nodes[edge[1]]['final_position'] - g_new.nodes[edge[0]]['final_position']
            g_new.edges[edge]['final_edge_delta'] = g_new.edges[edge]['final_edge_delta'].astype(np.float32)
        # Set contact force to be 0
        g_new.graph['contact_force'] = np.zeros((3,)).astype(np.float32)

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


def get_quat_from_vec(vec, branch_vec, gripper_axis):
    '''
    Returns quaternion that rotates z axis to vec and x axis to a vector orthogonal to vec and branch_vec
    '''
    if gripper_axis == 'x':
        # Compute orthonormal basis
        x_axis = vec/torch.norm(vec, dim=-1, keepdim=True)
        branch_vec = branch_vec/torch.norm(branch_vec, dim=1, keepdim=True)
        z_axis = torch.cross(x_axis, branch_vec, dim=-1)
        z_axis = -z_axis/torch.norm(z_axis, dim=-1, keepdim=True) # NOTE: Switched the sign of this for motion planning version to work
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis/torch.norm(y_axis, dim=-1, keepdim=True)
    elif gripper_axis == 'z':
        z_axis = vec/torch.norm(vec, dim=-1, keepdim=True)
        branch_vec = branch_vec/torch.norm(branch_vec, dim=1, keepdim=True)
        y_axis = torch.cross(z_axis, branch_vec, dim=-1)
        y_axis = y_axis/torch.norm(y_axis, dim=-1, keepdim=True)
        x_axis = torch.cross(y_axis, z_axis, dim=-1)
        x_axis = x_axis/torch.norm(x_axis, dim=-1, keepdim=True)
    # Given three orthonormal basis, the quaternion is given by
    # q = [w, x, y, z] = [sqrt(1+trace(R))/2, (R21-R12)/(4w), (R02-R20)/(4w), (R10-R01)/(4w)]
    # where R is the rotation matrix
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    trace = x_axis[:,0] + y_axis[:,1] + z_axis[:,2]
    w = torch.sqrt(1+trace)/2
    x = (y_axis[:,2] - z_axis[:,1])/(4*w)
    y = (z_axis[:,0] - x_axis[:,2])/(4*w)
    z = (x_axis[:,1] - y_axis[:,0])/(4*w)
    quat = torch.stack((x,y,z, w), dim=-1)
    return quat

