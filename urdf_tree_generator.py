from xml.dom import minidom
import os
import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R

class URDFTreeGenerator(object):
    def __init__(self, nx_graph, urdf_name, trunk_radius, asset_path):
        
        # XML URDF initalization
        self.urdf = minidom.Document()
        self.robot = self.urdf.createElement('robot')
        self.robot.setAttribute('name', urdf_name)
        self.urdf.appendChild(self.robot)
        self.generate_color_definitions()
        self.nx_graph = nx_graph
        # Generate tree from graph
        self.generate_tree(nx_graph, trunk_radius)

        urdf_string = self.urdf.toprettyxml(indent='    ')
        self.save_file_name = urdf_name+'.urdf'
        with open(os.path.join(asset_path, self.save_file_name), "w") as f:
            f.write(urdf_string)
        
    def generate_tree(self, nx_graph, trunk_radius):

        #edges, self.node_positions, edge_radii, _edge_length, edge_rpy = self._get_tree_info(nx_graph)
        edges = nx_graph.edges
        num_nodes = len(nx_graph.nodes)
        self.node_positions = np.array([nx_graph.nodes[i]['position'] for i in range(num_nodes)])
        _edge_length = np.array([nx_graph.edges[edge]['length'] for edge in edges])
        
        # Compute edge rpy
        edge_rpy = []
        rotation_history = {'0': R.from_matrix(np.eye(3))}
        # The following is a way to compute roll pitch yaw from a vector in world coordinates for building the urdf
        # The URDF is a tree structure so we need to keep track of the orientation of each node such that the origin of the URDF joint is appropriately oriented
        for edge in edges:
            # Get edge direction
            edge_direction = self.node_positions[edge[1]] - self.node_positions[edge[0]]
            edge_direction = edge_direction / np.linalg.norm(edge_direction)
            r_local = R.align_vectors([rotation_history[f'{edge[0]}'].as_matrix().T@edge_direction], [[0,0,1]])[0]
            r_parent =  rotation_history[f'{edge[0]}'] * r_local
            rpy = r_local.as_euler('xyz', degrees=False)
            edge_rpy.append(rpy)
            rotation_history[f'{edge[1]}'] = r_parent
        edge_rpy = np.array(edge_rpy)
        
        # Compute edge radius
        for edge in edges:
            nx_graph.edges[edge]['weight'] = np.linalg.norm(self.node_positions[edge[0]] - self.node_positions[edge[1]])
        # Weight is equal to the length of the longest path in subtree rooted at edge parent node
        # radius = trunk_radius * (edge_weight / trunk_weight)
        for edge_idx, edge in enumerate(edges):
            if edge[0]==0:
                nx_graph.edges[edge[0], edge[1]]['radius'] = trunk_radius

                path_lengths_dict = nx.shortest_path_length(nx_graph, source=edge[0], weight='weight') 
                trunk_weight = max(path_lengths_dict.values())
            else:

                path_lengths_dict = nx.shortest_path_length(nx_graph, source=edge[0], weight='weight') 
                edge_weight =  max(path_lengths_dict.values())
                edge_radius = trunk_radius * (edge_weight / trunk_weight)
                nx_graph.edges[edge[0], edge[1]]['radius'] = max([edge_radius, 0.01]) # TODO: Check if this is working
        edge_radii = np.array([nx_graph.edges[edge]['radius'] for edge in edges])
        
        # Generate root node
        self.generate_node_link(0, radius=edge_radii[0], color='green')
        # Generate tree
        for i, (parent_idx, child_idx) in enumerate(edges):
            edge_length = _edge_length[i]
            self.generate_edge_link(parent_idx, child_idx, edge_radii[i], edge_length, 'brown')
            self.generate_node_link(child_idx)
            damping = 100000
            effort_xyz = [99999999, 99999999, 99999999] 
            self.generate_spherical_joint(parent_idx, child_idx, edge_rpy[i], edge_length, damping, effort_xyz) 
            
    def generate_color_definitions(self):
        """
        generates color definition section of urdf
        """
        colors = [
            ("green", "0 0.6 0 0.8"),
            ("brown", "0.3 0.15 0.05 1.0")
        ]
        for name, rgba in colors:
            material = self.urdf.createElement('material')
            material.setAttribute('name', name)
            self.robot.appendChild(material)
            color = self.urdf.createElement('color')
            color.setAttribute('rgba', rgba)
            material.appendChild(color)

    def generate_node_link(self, 
                           node_idx,  
                           radius=None, # Set visuals for root node and leaf nodes 
                           color=None): # Otherwise leave as None OR set to brown maybe
        # create link element        
        link = self.urdf.createElement('link') 
        link.setAttribute('name', f'node_{node_idx}') 
        
        if color:
            # Add visual properties
            visual = self.urdf.createElement('visual') 

            # Add origin
            origin = self.urdf.createElement('origin') 
            origin.setAttribute('xyz', '0 0 0')
            origin.setAttribute('rpy', '0 0 0')
            visual.appendChild(origin)

            # Add geometry sphere
            geometry = self.urdf.createElement('geometry')
            sphere = self.urdf.createElement('sphere')
            sphere.setAttribute('radius', str(radius)) 
            geometry.appendChild(sphere)
            visual.appendChild(geometry)

            # Add material
            material = self.urdf.createElement('material')
            material.setAttribute('name', 'green')
            visual.appendChild(material)

            link.appendChild(visual)
        self.robot.appendChild(link)

    def generate_edge_link(self, 
                           parent_idx, 
                           child_idx, 
                           radius, # Radius of branch cylinder connecting parent and child 
                           length, # Length of branch cylinder connecting parent and child
                           color):
        # create x link element
        link = self.urdf.createElement('link') 
        link.setAttribute('name', f'link_{parent_idx}_x_{child_idx}') 
        self.robot.appendChild(link)
        # create y link element
        link = self.urdf.createElement('link') 
        link.setAttribute('name', f'link_{parent_idx}_y_{child_idx}') 
        self.robot.appendChild(link)
        # create z link element with visuals
        link = self.urdf.createElement('link') 
        link.setAttribute('name', f'link_{parent_idx}_z_{child_idx}') 
        
        # Add visual and collision properties
        visual = self.urdf.createElement('visual') 
        collision = self.urdf.createElement('collision')

        # Add origin
        origin = self.urdf.createElement('origin') 
        origin.setAttribute('xyz', f'0 0 {0.5*length}') # z is half the length
        origin.setAttribute('rpy', '0 0 0')
        visual.appendChild(origin)
        origin = self.urdf.createElement('origin') 
        origin.setAttribute('xyz', f'0 0 {0.5*length}') # z is half the length
        origin.setAttribute('rpy', '0 0 0')
        collision.appendChild(origin)

        # Add geometry cylinder
        geometry = self.urdf.createElement('geometry')
        cylinder = self.urdf.createElement('cylinder')
        cylinder.setAttribute('length', str(length)) 
        cylinder.setAttribute('radius', str(radius)) 
        geometry.appendChild(cylinder)
        visual.appendChild(geometry)
        geometry = self.urdf.createElement('geometry')
        cylinder = self.urdf.createElement('cylinder')
        cylinder.setAttribute('length', str(length)) 
        cylinder.setAttribute('radius', str(radius)) 
        geometry.appendChild(cylinder)
        collision.appendChild(geometry) 

        # Add material
        material = self.urdf.createElement('material')
        material.setAttribute('name', color)
        visual.appendChild(material)
        
        # Add visual and collision properties to link element
        link.appendChild(visual)
        link.appendChild(collision) 
        
        # Add link element to robot
        self.robot.appendChild(link)
        

    def generate_spherical_joint(self, 
                                 parent_idx, 
                                 child_idx, 
                                 rpy, # Angle of branch joining the parent and child
                                 length, # Length of branch joining the parent and child
                                 damping, 
                                 effort, 
                                 friction=3.0, 
                                 lower=-3.1416, 
                                 upper=3.1416, 
                                 velocity=3.0):
        # TODO: The dynamic parameters should be different for each axis
        # Create x joint element
        #if parent_idx==0:
        #    lower = -0.01
        #    upper = 0.01
        joint = self.urdf.createElement('joint') 
        joint.setAttribute('name', f'joint_{parent_idx}_x_{child_idx}') 
        joint.setAttribute('type', 'revolute') 
        # Add parent and child
        parent = self.urdf.createElement('parent')
        parent.setAttribute('link', f'node_{parent_idx}')
        joint.appendChild(parent)
        child = self.urdf.createElement('child')
        child.setAttribute('link', f'link_{parent_idx}_x_{child_idx}')
        joint.appendChild(child)
        # Add origin
        origin = self.urdf.createElement('origin')
        origin.setAttribute('xyz', f'0 0 0')
        origin.setAttribute('rpy', f'{rpy[0]} {rpy[1]} {rpy[2]}') # NOTE: Direction of next branch here
        joint.appendChild(origin)
        # Add axis
        axis = self.urdf.createElement('axis')
        axis.setAttribute('xyz', '1 0 0')
        joint.appendChild(axis)
        # Add dynamics
        dynamics = self.urdf.createElement('dynamics')
        dynamics.setAttribute('damping', str(damping))
        dynamics.setAttribute('friction', str(friction))
        joint.appendChild(dynamics)
        # Add limit
        limit = self.urdf.createElement('limit')
        limit.setAttribute('effort', str(effort[0]))
        limit.setAttribute('lower', str(lower/2.0))
        limit.setAttribute('upper', str(upper/2.0))
        limit.setAttribute('velocity', str(velocity))
        joint.appendChild(limit)
        # Append joint to robot
        self.robot.appendChild(joint)

        # Create y joint element
        joint = self.urdf.createElement('joint')
        joint.setAttribute('name', f'joint_{parent_idx}_y_{child_idx}')
        joint.setAttribute('type', 'revolute')
        # Add parent and child
        parent = self.urdf.createElement('parent')
        parent.setAttribute('link', f'link_{parent_idx}_x_{child_idx}')
        joint.appendChild(parent)
        child = self.urdf.createElement('child')
        child.setAttribute('link', f'link_{parent_idx}_y_{child_idx}')
        joint.appendChild(child)
        # Add origin
        origin = self.urdf.createElement('origin')
        origin.setAttribute('xyz', f'0 0 0')
        origin.setAttribute('rpy', f'0 0 0')
        joint.appendChild(origin)
        # Add axis
        axis = self.urdf.createElement('axis')
        axis.setAttribute('xyz', '0 1 0')
        joint.appendChild(axis)
        # Add dynamics
        dynamics = self.urdf.createElement('dynamics')
        dynamics.setAttribute('damping', str(damping))
        dynamics.setAttribute('friction', str(friction))
        joint.appendChild(dynamics)
        # Add limit
        limit = self.urdf.createElement('limit')
        limit.setAttribute('effort', str(effort[1]))
        limit.setAttribute('lower', str(lower/2.0))
        limit.setAttribute('upper', str(upper/2.0))
        limit.setAttribute('velocity', str(velocity))
        joint.appendChild(limit)
        # Append joint to robot
        self.robot.appendChild(joint)

        # Create z joint element
        joint = self.urdf.createElement('joint')
        joint.setAttribute('name', f'joint_{parent_idx}_z_{child_idx}')
        joint.setAttribute('type', 'revolute')
        # Add parent and child
        parent = self.urdf.createElement('parent')
        parent.setAttribute('link', f'link_{parent_idx}_y_{child_idx}')
        joint.appendChild(parent)
        child = self.urdf.createElement('child')
        child.setAttribute('link', f'link_{parent_idx}_z_{child_idx}')
        joint.appendChild(child)
        # Add origin
        origin = self.urdf.createElement('origin')
        origin.setAttribute('xyz', f'0 0 0')
        origin.setAttribute('rpy', f'0 0 0')
        joint.appendChild(origin)
        # Add axis
        axis = self.urdf.createElement('axis')
        axis.setAttribute('xyz', '0 0 1')
        joint.appendChild(axis)
        # Add dynamics
        dynamics = self.urdf.createElement('dynamics')
        dynamics.setAttribute('damping', str(damping))
        dynamics.setAttribute('friction', str(friction))
        joint.appendChild(dynamics)
        # Add limit
        limit = self.urdf.createElement('limit')
        limit.setAttribute('effort', str(effort[2]))
        limit.setAttribute('lower', str(lower*2))
        limit.setAttribute('upper', str(upper*2))
        limit.setAttribute('velocity', str(velocity))
        joint.appendChild(limit)
        # Append joint to robot
        self.robot.appendChild(joint)

        # Create primary joint element
        joint = self.urdf.createElement('joint')
        joint.setAttribute('name', f'joint_{parent_idx}_{child_idx}')
        joint.setAttribute('type', 'fixed')
        # Add parent and child
        parent = self.urdf.createElement('parent')
        parent.setAttribute('link', f'link_{parent_idx}_z_{child_idx}')
        joint.appendChild(parent)
        child = self.urdf.createElement('child')
        child.setAttribute('link', f'node_{child_idx}')
        joint.appendChild(child)
        # Add origin
        origin = self.urdf.createElement('origin')
        origin.setAttribute('xyz', f'0 0 {length}') # NOTE: Length of next branch here
        origin.setAttribute('rpy', f'0 0 0')
        joint.appendChild(origin)
        # Append joint to robot
        self.robot.appendChild(joint)