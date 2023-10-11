import open3d as o3d
import numpy as np
import argparse
import cv2
import networkx as nx
from azure_kinect_module import AzureKinectModule

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      #zoom=0.3412,
                                      #front=[0.4257, -0.2125, -0.8795],
                                      #lookat=[2.6172, 2.0475, 1.532],
                                      #up=[-0.0694, -0.9768, 0.2024]
                                      )

class TreeGraphExtractor:
    def __init__(self, camera, max_depth=0.6):
        self.camera = camera
        self.max_depth = max_depth
        self.youngs_modulous = 10_000_000_000

    def extract_tree_graph(self):
        '''
        Description: Extract tree graph from the point cloud
        Input: None
        Output: 
            - mst_simple: networkx graph object representing the simplified tree graph.
                Node attributes: position, radius
                Edge attributes: length, stiffness
        '''
        VISUALIZE = False
        color, depth = self.camera.get_frame()
        # Compute discontinuity map
        discontinuity_map = self.extract_depth_discontuinities(depth, disc_rat_thresh=100)
        # Mask out discontinuity and depth high pass filter
        depth = depth * ~np.array(discontinuity_map, dtype=bool)
        pcd_very_raw = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), self.camera.intrinsic)
        
        if VISUALIZE:
            R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            pcd_very_raw.rotate(R, center=(0, 0, 0))
            self.visualize_o3d([pcd_very_raw])

        depth[depth > self.max_depth] = 0.0
        # Convert depth image to point cloud
        pcd_raw = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), self.camera.intrinsic)


        # Statistical outlier removal
        _, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        self.pcd_filtered = pcd_raw.select_by_index(ind)

        if VISUALIZE:
            R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            pcd_raw.rotate(R, center=(0, 0, 0))
            self.visualize_o3d([pcd_raw])

        # Rotate points about x-axis by +90 degrees
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        self.pcd_filtered.rotate(R, center=(0, 0, 0))
        if VISUALIZE:
            self.visualize_o3d([self.pcd_filtered])

        # Laplacian smoothing
        smoothed_points = self.laplacian_smoothing(np.asarray(self.pcd_filtered.points), 0.02)
        pcd_smoothed = o3d.geometry.PointCloud()
        pcd_smoothed.points = o3d.utility.Vector3dVector(smoothed_points)
        if VISUALIZE:
            self.visualize_o3d([pcd_smoothed])

        # Statistical oulier removal and voxel downsampling
        _, ind = pcd_smoothed.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        self.pcd_processed = pcd_smoothed.select_by_index(ind)
        self.pcd_processed = self.pcd_processed.voxel_down_sample(voxel_size=0.005)
        if VISUALIZE:
            self.visualize_o3d([self.pcd_processed])

        # Compute MST
        mst = self.minimum_spanning_tree(np.asarray(self.pcd_processed.points))
        root_node = self.get_root_node(mst)
        directed_mst = nx.dfs_tree(mst, root_node)    
        
        # Simplify graph
        mst_simple = self.simplify_graph(directed_mst, np.asarray(self.pcd_processed.points))
        root_node = [node for node, degree in mst_simple.in_degree if degree == 0][0]

        # Populate the node and edge attributes
        mst_simple = self.populate_node_attributes(mst_simple)
        mst_simple = self.populate_edge_attributes(mst_simple)
        mst_simple = self.reindex_nodes(mst_simple)
        # Center the tree by setting root to be at 0
        mst_simple, offset = self.center_tree(mst_simple)


        if True:
            root_node = [node for node, degree in mst_simple.in_degree if degree == 0][0]
            ground_pcd = o3d.geometry.PointCloud()
            ground_pcd.points = o3d.utility.Vector3dVector([mst_simple.nodes[root_node]['position']])
            line_set = o3d.geometry.LineSet()
            # Get points from mst_simple
            points = []
            for i in range(len(mst_simple.nodes)):
                points.append(mst_simple.nodes[i]['position'])
            line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
            line_set.lines = o3d.utility.Vector2iVector(np.asarray(mst_simple.edges()))
            self.visualize_o3d([line_set, ground_pcd, self.pcd_processed])
            #self.visualize_o3d([line_set, ground_pcd])
        return mst_simple
    
    @staticmethod
    def center_tree(mst):
        '''
        Description: Center the tree by setting root to be at 0
        Input: 
            - mst: networkx graph object
        Output:
            - mst: networkx graph object
        '''
        # Get the root node, which has no parent node
        root_node = [node for node, degree in mst.in_degree if degree == 0][0]
        # Get the root node position
        offset = mst.nodes[root_node]['position']
        # Subtract the root node position from all node positions
        for node in mst.nodes:
            mst.nodes[node]['initial_position'] = mst.nodes[node]['position'] - offset
            mst.nodes[node]['initial_position'] = mst.nodes[node]['initial_position'].astype(np.float32)
        return mst, offset

    @staticmethod
    def reindex_nodes(mst):
        '''
        Description: Reindex the nodes of the graph so that the node indices are continuous starting from 0, where 0 is the root 
        Input: 
            - mst: networkx graph object
        Output:
            - mst: networkx graph object
        '''
        # Get the root node, which has no parent node
        root_node = [node for node, degree in mst.in_degree if degree == 0][0]
        # Starting from root node, iterate through the graph in a depth first search manner
        # Sort the nodes in the order of depth first search
        dfs_nodes = list(nx.dfs_preorder_nodes(mst, root_node))
        # Reindex the nodes
        mapping = {dfs_nodes[i]: i for i in range(len(dfs_nodes))}
        mst = nx.relabel_nodes(mst, mapping)
        return mst
    
    def populate_node_attributes(self, mst_simple):
        # Populate the node position attributes
        for node in mst_simple.nodes:
            mst_simple.nodes[node]['position'] = np.asarray(self.pcd_processed.points)[node]

        # Populate radius of each node by referring to pcd_filtered
        kd_tree = o3d.geometry.KDTreeFlann(self.pcd_filtered)
        for node in mst_simple.nodes:
            # Get the node position
            node_position = mst_simple.nodes[node]['position']
            # Get the node radius
            [k, idxs, _] = kd_tree.search_knn_vector_3d(node_position, knn=400)
            neighbors = np.asarray(self.pcd_filtered.points)[idxs]
            # Compute the aspect ratio of the neighboring points
            neighbors_centered = neighbors - np.mean(neighbors, axis=0)
            covariance_matrix = (neighbors_centered.T@neighbors_centered)/k
            u, s, vh = np.linalg.svd(covariance_matrix)
            # Align the centered neighbors along the principal axes
            neighbors_aligned = neighbors_centered@u
            mask = np.logical_and(neighbors_aligned[:,0] < 0.005, neighbors_aligned[:,0] > -0.005)
            neighbors_aligned = neighbors_aligned[mask]   
            d_max = np.max(neighbors_aligned[:,2])
            d_min = np.min(neighbors_aligned[:,2])
            estimated_radius = (d_max-d_min)/2
            mst_simple.nodes[node]['radius'] = estimated_radius.astype(np.float32)
        return mst_simple

    def populate_edge_attributes(self, mst_simple):
        # Populate edge attributes including length and stiffness
        for edge in mst_simple.edges:
            # Get the edge nodes
            node1, node2 = edge
            # Get the edge length
            edge_length = np.linalg.norm(mst_simple.nodes[node1]['position'] - mst_simple.nodes[node2]['position'])
            mst_simple.edges[edge]['length'] = edge_length.astype(np.float32)
            # Get the edge stiffness
            edge_stiffness = self.youngs_modulous*np.pi*mst_simple.nodes[node1]['radius']**4/(4*edge_length)
            mst_simple.edges[edge]['stiffness'] = edge_stiffness.astype(np.float32)
            mst_simple.edges[edge]['radius'] = mst_simple.nodes[node1]['radius']
            mst_simple.edges[edge]['parent2child'] = 1
            mst_simple.edges[edge]['branch'] = 1
            mst_simple.edges[edge]['initial_edge_delta'] = mst_simple.nodes[node2]['position'] - mst_simple.nodes[node1]['position']
            mst_simple.edges[edge]['initial_edge_delta'] = mst_simple.edges[edge]['initial_edge_delta'].astype(np.float32)
        # Delete radius from node attributes
        for node in mst_simple.nodes:
            del mst_simple.nodes[node]['radius']
        return mst_simple

    @staticmethod
    def visualize_o3d(o3d_entity_list):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries(o3d_entity_list + [mesh_frame],
                                      zoom=0.7,
                                      front=[ -0.7450501521572539, -0.63417959618329034, 0.20667972942514667 ],
                                      lookat=[ -0.011800197549886149, 0.29700001192092895, -0.02884005742568474 ],
                                      up=[ 0.16296161319842123, 0.12740107204693898, 0.97837236237797454 ],
                                      )



    @staticmethod
    def get_root_node(mst):
        # Get the root node, which has no parent node
        min_height = np.inf
        for node in mst.nodes:
            # Get the node position
            node_position = mst.nodes[node]['position']
            # If the node position is lower than the current max height, update the max height and the ground node
            if node_position[2] < min_height:
                min_height = node_position[2]
                root_node = node
        return root_node

    @staticmethod
    def extract_depth_discontuinities(depth_img, disc_rat_thresh):
        '''
        Description: Extract depth discontinuities from depth image
        Input: 
            - depth_img: numpy array of shape (height, width)
            - disc_rat_thresh: float
        Output:
            - discontinuity_map: numpy array of shape (height, width)
        '''
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilation = cv2.dilate(depth_img, element)
        erosion = cv2.erode(depth_img, element)
        dilation -= depth_img
        erosion = depth_img - erosion
        max_image = np.max((dilation, erosion), axis=0)
        ratio_image = max_image / depth_img
        _, discontinuity_map = cv2.threshold(ratio_image, disc_rat_thresh, 1.0, cv2.THRESH_BINARY)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        discontinuity_map = cv2.morphologyEx(discontinuity_map, cv2.MORPH_CLOSE, element)
        return discontinuity_map

    @staticmethod
    def laplacian_smoothing(points, search_radius):
        '''
        Description: Apply laplacian smoothing to the input point cloud
        Input: 
            - points: numpy array of shape (num_points, 3)
            - search_radius: float
        Output: 
            - new_points: numpy array of shape (num_points, 3)
        '''
        # Construct kd-tree
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        new_points = []
        for point in points:
            [k, idxs, _] = kd_tree.search_knn_vector_3d(point, knn=20)
            neighbors = points[idxs]
            # Compute the aspect ratio of the neighboring points
            neighbors_centered = neighbors - np.mean(neighbors, axis=0)
            covariance_matrix = (neighbors_centered.T@neighbors_centered)/k
            u, s, vh = np.linalg.svd(covariance_matrix)
            aspect_ratio = np.sqrt(s[1]/s[0])
            # Find neighbors using search radius as a function of aspect ratio
            [k, idxs, _] = kd_tree.search_radius_vector_3d(point, radius=0.001+search_radius*aspect_ratio)
            neighbors = points[idxs]
            # Compute the new point as the mean of the neighbors
            new_point = np.mean(neighbors, axis=0)
            new_points.append(new_point)
        return np.array(new_points)

    @staticmethod
    def minimum_spanning_tree(points):
        '''
        Description: Construct a minimum spanning tree from the input points
        Input: 
            - points: numpy array of shape (num_points, 3)
        Output:
            - T: networkx graph object representing the minimum spanning tree
        '''
        num_nodes = len(points)
        fully_connected_weighted_adj_mat = np.zeros((num_nodes, num_nodes))        
        # Compute the distance between all pairs of points in vectorized form
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x1 = x.reshape(-1, 1)
        x2 = x.reshape(1, -1)
        y1 = y.reshape(-1, 1)
        y2 = y.reshape(1, -1)
        z1 = z.reshape(-1, 1)
        z2 = z.reshape(1, -1)
        dist_mat = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        # Construct fully connected weighted adjacency matrix
        fully_connected_weighted_adj_mat[np.triu_indices(num_nodes, k=1)] = dist_mat[np.triu_indices(num_nodes, k=1)]
        fully_connected_weighted_adj_mat[np.tril_indices(num_nodes, k=-1)] = dist_mat[np.tril_indices(num_nodes, k=-1)]

        # Construct MST
        G = nx.from_numpy_array(fully_connected_weighted_adj_mat)
        T = nx.minimum_spanning_tree(G)

        # Add node attributes: points, radii and scores
        for i, node in enumerate(T.nodes):
            T.nodes[node]['position'] = points[i]
        return T

    @staticmethod
    def simplify_graph(G, points):
        '''
        Description: Simplify the graph by removing nodes that are too close to each other.
                    Start from root node and find the next descendant node with degree greater than 2. 
                    If the distance between the root node and the descendant node is greater than 0.2, 
                    subdivide the edge between the root node and the descendant node
        Input:
            - G: networkx graph object
            - points: numpy array of shape (num_points, 3)
        Output:
            - G: networkx graph object
        '''
        MAX_EDGE_LENGTH = 0.2
        MIN_EDGE_LENGTH = 0.05
        MAX_NODES = 50
        # Get the root node, which has no parent node
        root_node = [node for node, degree in G.in_degree if degree == 0][0]
        # Starting from root node, iterate through the graph in a depth first search manner
        # Sort the nodes in the order of depth first search
        dfs_nodes = list(nx.dfs_preorder_nodes(G, root_node))
        line_segments = []
        line_segment = []
        for node in dfs_nodes:
            line_segment.append(node)
            if G.out_degree(node)==0:
                line_segments.append(line_segment)
                line_segment = []
            if G.out_degree(node)==2:
                line_segments.append(line_segment)
                line_segment = [node]

        simple_line_segments = set()
        for line_segment in line_segments:
            distance = np.linalg.norm(points[line_segment[0]] - points[line_segment[-1]])
            num_line_segments = int(np.ceil(distance/MAX_EDGE_LENGTH))
            simple_line_segments.add(line_segment[0])
            simple_line_segments.add(line_segment[-1])
            if num_line_segments>1:
                step = len(line_segment)//(num_line_segments)
                for i in range(1, num_line_segments):
                    simple_line_segments.add(line_segment[i*step])
        
        # Start Here
        simple_line_segments = list(simple_line_segments)
        G_copy = G.copy()
        for node, degree in G_copy.degree:
            if node not in simple_line_segments and degree==2:
                #Get the two incident nodes
                incident_nodes = list(G.predecessors(node)) + list(G.successors(node)) 
                #Remove the node
                G.remove_node(node)
                #Add the new edge
                G.add_edge(incident_nodes[0], incident_nodes[1])
            elif degree==0:
                G.remove_node(node)
        
        # Beginning from root node, if the distance between parent node and child node is smaller than 0.05m, 
        # remove the child node and connect child node's child node to parent node
        flag = False
        connected = False
        while not flag:
            dfs_nodes = list(nx.dfs_preorder_nodes(G, root_node))
            for node in dfs_nodes:
                child_nodes = list(G.successors(node))
                for child_node in child_nodes:
                    if np.linalg.norm(points[node] - points[child_node])<MIN_EDGE_LENGTH: # Hyperparameter
                        grandchild_nodes = list(G.successors(child_node))                    
                        G.remove_node(child_node)
                        for grandchild_node in grandchild_nodes:
                            G.add_edge(node, grandchild_node)
                        connected = True
                        break
                if connected:
                    break
            if not connected:
                flag = True
            connected = False  

        # If number of nodes in G is greater than max_nodes, prune leaf nodes that are farthest from root node until number of nodes is equal to max_nodes
        root_node = [node for node, degree in G.in_degree if degree == 0][0]
        while G.number_of_nodes() > MAX_NODES:
            leaf_nodes = [node for node, degree in G.out_degree if degree==0]
            leaf_nodes_distances = []
            for leaf_node in leaf_nodes:
                leaf_nodes_distances.append(np.linalg.norm(points[root_node] - points[leaf_node]))
            leaf_node_to_prune = leaf_nodes[np.argmax(leaf_nodes_distances)]
            G.remove_node(leaf_node_to_prune)
        return G        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perception Module.')
    parser.add_argument('--config', type=str, help='input json kinect config')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('-a',
                        '--align_depth_to_color',
                        action='store_true',
                        help='enable align depth image to color')
    args = parser.parse_args()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0
    camera = AzureKinectModule(config, device, args.align_depth_to_color)
    perception_module = TreeGraphExtractor(camera)
    tree_graph = perception_module.extract_tree_graph()

    # Print the node and edge attributes of the tree graph
    print('Node attributes:')
    for node in tree_graph.nodes:
        print(node, tree_graph.nodes[node])
    print('Edge attributes:')
    for edge in tree_graph.edges:
        print(edge, tree_graph.edges[edge])
