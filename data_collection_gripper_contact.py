"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Apply Forces (apply_forces.py)
----------------------------
This example shows how to apply forces and torques to rigid bodies using the tensor API.
"""

from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import os
import numpy as np
import torch
from copy import deepcopy
import utils
from omegaconf import OmegaConf
import networkx as nx

class DataCollectionGym:
    def __init__(self, cfg):
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # Load some config file parameters
        self.cfg = cfg
        self.headless = self.cfg.task.headless
        self.max_episode_length = self.cfg["env"]["steadyStateLength"]
        self.visualize_actions = False #self.cfg["env"]["visualizeActions"] # For visualizing target node and force vector
        # Load asset information
        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"]["assetRoot"])
        tree_asset_path = self.cfg["env"]["asset"]["assetPathTrees"]
        tree_group_suffx = self.cfg["env"]["asset"]["assetGroupSuffix"]
        self.num_tree_types = self.cfg["env"]["trees"]["numTrees"]
        self.num_envs = self.cfg["env"]["numEnvs"]
        # Check if numEnvs is divisible by num_tree_types and raise error if not
        if self.num_envs % self.num_tree_types != 0:
            raise ValueError("Number of environments must be divisible by number of tree types. Adjust in cfg yaml file.")

        # Iteratively load urdf path and graph information
        self.tree_urdf_paths = []
        self.DiGs_by_id = []
        self.edge_tensor_list = []
        for i in range(self.num_tree_types):
            tree_urdf_path = os.path.join(tree_asset_path, tree_group_suffx + f'_{i}.urdf')
            self.tree_urdf_paths.append(tree_urdf_path)
            DiG_by_id = utils.parse_urdf_graph(os.path.join(self.asset_root, tree_urdf_path))
            self.DiGs_by_id.append(DiG_by_id)
            edge_tensor = torch.Tensor([[int(parent), int(child)] for parent, child in DiG_by_id.edges()]).long() # shape: (num_edges, 2) 
            self.edge_tensor_list.append(edge_tensor)
        # Initialize tree graph information
        self.num_nodes = self.DiGs_by_id[0].number_of_nodes() # Including root node (0)
        self.num_edges = self.DiGs_by_id[0].number_of_edges()
        self.up_axis = "z"
        self.up_axis_idx = 2
        self.num_actors = 2 # Tree, gripper
        self.create_sim()

        # Acquire gym tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
                
        # Dof state slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.tree_dof_state = self.dof_state.view(self.num_envs, -1, 2)

        # Default tree dof pos: tree is passive, hence the default dof pos is 0
        self.tree_default_dof_pos = torch.zeros_like(self.dof_state, device=self.device)

        # (N, num_bodies, 13)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        
        # (N, 3, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
                        
        # Dof targets. This is the target dof pos for the tree, not the goal pos
        self.dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        # Global inds
        self._global_indices = torch.arange(
            self.num_envs * self.num_actors, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

        # Reset all environments
        self.gym.refresh_actor_root_state_tensor(self.sim) 
        self.gym.refresh_dof_state_tensor(self.sim) 
        self.gym.refresh_rigid_body_state_tensor(self.sim) 
        
        # Had to make it into function because indexing seems to make copy instead of reference
        node_id2positions = self.get_node_id2positions()
        
        self.default_node_id2positions_list = []
        for i in range(self.num_tree_types):
            default_node_id2positions = {}
            for node_id in range(self.num_nodes):
                default_node_id2positions[node_id] = deepcopy(node_id2positions[node_id][i])
            self.default_node_id2positions_list.append(default_node_id2positions)

    def get_node_id2positions(self):
        # nodes and goal position slices 
        node_id2positions = {} 
        for node_id in range(self.num_nodes):
            # NOTE: This part is probably not stable when num_envs is not divisible by num_tree_types. 
            # Hence we do an assert during initialization to make sure num_envs is divisible by num_tree_types
            node_handles = [self.rb_node_id2handles_list[i][node_id] for i in range(self.num_tree_types)]*(self.num_envs//self.num_tree_types)
            node_id2positions[node_id] = self.rigid_body_state[torch.arange(self.num_envs), node_handles, :3]
        return node_id2positions

    def create_sim(self):
        
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        sim_params.substeps = self.cfg["sim"]["substeps"]
        sim_params.physx.solver_type = self.cfg["sim"]["physx"]["solver_type"]
        sim_params.physx.num_position_iterations = self.cfg["sim"]["physx"]["num_position_iterations"]
        sim_params.physx.num_velocity_iterations = self.cfg["sim"]["physx"]["num_velocity_iterations"]
        sim_params.physx.num_threads = self.cfg["sim"]["physx"]["num_threads"]
        sim_params.physx.use_gpu = self.cfg["sim"]["physx"]["use_gpu"]

        sim_params.use_gpu_pipeline = self.cfg["sim"]["use_gpu_pipeline"]
        args = gymutil.parse_arguments()
        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        #self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        self.gym.prepare_sim(self.sim)
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.02
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # Create tree asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.005
        asset_options.collapse_fixed_joints = False 
        asset_options.armature = 0.01
        asset_options.max_angular_velocity = 40.
        asset_options.max_linear_velocity = 100.
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.density = 500
        asset_options.override_com = True
        asset_options.override_inertia = True
        tree_assets = []
        for tree_urdf_path in self.tree_urdf_paths:
            tree_asset = self.gym.load_asset(self.sim, self.asset_root, tree_urdf_path, asset_options)
            tree_assets.append(tree_asset)

        gripper_asset = self.gym.load_asset(self.sim, self.asset_root, "zero_dof_gripper.urdf", asset_options)

        self.num_bodies = self.gym.get_asset_rigid_body_count(tree_asset) + self.gym.get_asset_rigid_body_count(gripper_asset)
        self.num_tree_dofs = self.gym.get_asset_dof_count(tree_asset)
        self.num_dofs = self.num_tree_dofs

        print("num tree bodies: ", self.num_bodies)
        print("num tree dofs: ", self.num_tree_dofs)

        self.tree_dof_props_per_asset = []
        for tree_asset in tree_assets:
            # Set stick dof props by reading directly from URDF
            tree_dof_props = self.gym.get_asset_dof_properties(tree_asset)
            tree_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS) 
            # TODO: Seems like the stiffness parameter is not read from URDF, so we need to set it manually.
            # TODO: The damping parameter is read from URDF, but it is too small. We need to set it manually.
            tree_dof_props['effort'].fill(np.inf)
            tree_dof_props['velocity'].fill(9999999999)
            tree_dof_props['armature'].fill(0.1)
            for i in range(self.num_tree_dofs):
                tree_dof_props['stiffness'][i] = tree_dof_props['friction'][i]    
            tree_dof_props['friction'].fill(0.0)
            self.tree_dof_props_per_asset.append(tree_dof_props)

        # Set start pose of tree and gripper
        tree_start_pose = gymapi.Transform()
        tree_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        tree_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        gripper_start_pose = gymapi.Transform()
        gripper_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        gripper_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size, results in error if incorrect
        num_tree_bodies = self.gym.get_asset_rigid_body_count(tree_asset)
        num_tree_shapes = self.gym.get_asset_rigid_shape_count(tree_asset)
        num_gripper_bodies = self.gym.get_asset_rigid_body_count(gripper_asset)
        num_gripper_shapes = self.gym.get_asset_rigid_shape_count(gripper_asset)
        self.max_agg_bodies =  num_tree_bodies + num_gripper_bodies
        max_agg_shapes =  num_tree_shapes + num_gripper_shapes
        
        # Cache actors and envs
        self.trees = []
        self.envs = []
        self.grippers = []
        # Create environments
        for i in range(self.num_envs):
            if (i+1)%self.num_tree_types == 0:
                print("\tCreating env ", i+1, " of ", self.num_envs)
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, max_agg_shapes, True)
                        
            # Create tree actor
            # Here we load different tree types by using different assets
            # Setting collision filter to 1 seems to make the tree more stable (visual observation)
            tree_actor = self.gym.create_actor(env_ptr, tree_assets[i%self.num_tree_types], tree_start_pose, "tree", i, 1, 0) 
            self.gym.set_actor_dof_properties(env_ptr, tree_actor, self.tree_dof_props_per_asset[i%self.num_tree_types])
            
            # Create robot actor
            gripper_actor = self.gym.create_actor(env_ptr, gripper_asset, gripper_start_pose, "gripper", i, 0, 0)
            
            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.trees.append(tree_actor)
            self.grippers.append(gripper_actor)
        
        # Store the rigid body node handles dictionary into list for each tree type
        self.rb_node_id2handles_list = []
        for i in range(self.num_tree_types):
            rb_node_id2handles = {}
            for node_id in range(self.num_nodes):
                rb_node_id2handles[node_id] = self.gym.find_actor_rigid_body_handle(self.envs[i], self.trees[i], f"node_{node_id}")
            self.rb_node_id2handles_list.append(rb_node_id2handles)
        self.env_tree_ind = self.gym.get_actor_index(env_ptr, tree_actor, gymapi.DOMAIN_ENV)
        self.gripper_ind = self.gym.get_actor_index(env_ptr, gripper_actor, gymapi.DOMAIN_ENV)

    def simulate(self):
        frame_count=0
        graph_data = []
        begin_action_frame = 20
        penetration_threshold = 10
        penetrated_env = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        while not self.gym.query_viewer_has_closed(self.viewer):
            # RESET
            if frame_count%self.max_episode_length == 0:
                # Reset all environments
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.tree_default_dof_pos))
                # Randomly generate contact node based on num_nodes using numpy
                contact_nodes = np.random.randint(low=1, high=self.num_nodes, size=(self.num_envs, ))
                # Generate random trajectory vector to be applied on contact node between (-1, 1)
                trajectory_distance = torch.rand((self.num_envs, ), device=self.device)*self.cfg["env"]["trajectoryScale"] 
                t_x = torch.rand((self.num_envs, ), device=self.device)*2 - 1
                t_y = torch.rand((self.num_envs, ), device=self.device)*2 - 1
                t_z = torch.rand((self.num_envs, ), device=self.device)*2 - 1
                # Concatenate force vector
                trajectory_vector = torch.stack((t_x, t_y, t_z), dim=-1)
                # Normalize force vector
                trajectory_vector_norm = trajectory_vector/torch.linalg.norm(trajectory_vector, dim=-1, keepdim=True)
                trajectory_vector = trajectory_distance.unsqueeze(-1)*trajectory_vector_norm

                # Compute initial contact node position and parent node position
                parent_nodes = [int(next(self.DiGs_by_id[i%self.num_tree_types].predecessors(str(contact_nodes[i]))))  
                    for i in range(self.num_envs)]
                parent_node_positions = torch.stack([self.default_node_id2positions_list[i%self.num_tree_types][node_id] for i, node_id in enumerate(parent_nodes)])
                contact_node_positions = torch.stack([self.default_node_id2positions_list[i%self.num_tree_types][node_id] for i, node_id in enumerate(contact_nodes)])
                # Compute branch vectors
                branch_vectors = contact_node_positions - parent_node_positions
                branch_vectors_norm = branch_vectors/torch.linalg.norm(branch_vectors, dim=-1, keepdim=True)
                
                # Compute end-effector trajectory
                trajectory_num_waypoints = self.max_episode_length-begin_action_frame
                self.gripper_trajectory = torch.zeros((self.num_envs, trajectory_num_waypoints, 3), dtype=torch.float32, device=self.device, requires_grad=False)
                # NOTE: Slightly offsetting along branch vector and force vector to avoid losing contact with branch
                self.gripper_trajectory[:, 0] = contact_node_positions - 0.01*branch_vectors_norm - 0.1*trajectory_vector_norm
                for i in range(1, trajectory_num_waypoints):
                    self.gripper_trajectory[:, i] = self.gripper_trajectory[:, 0] + i*((0.1+trajectory_distance).unsqueeze(-1)*trajectory_vector_norm/trajectory_num_waypoints)
                # Orient the end-effector such that z axis aligns with force vector
                self.gripper_quat = utils.get_quat_from_vec(trajectory_vector_norm, branch_vectors_norm, gripper_axis='z')

            multi_env_ids_int32 = self._global_indices[:, self.gripper_ind].flatten().contiguous()
                
            if frame_count%self.max_episode_length>begin_action_frame:
                self.root_state_tensor[:, self.gripper_ind, :3] = self.gripper_trajectory[:, frame_count-begin_action_frame] 
            else: 
                self.root_state_tensor[:, self.gripper_ind, :3] = self.gripper_trajectory[:, 0] 
            self.root_state_tensor[:, self.gripper_ind, 3:7] = self.gripper_quat

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(multi_env_ids_int32), 
                len(multi_env_ids_int32))

            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim) 
            
            # If the gripper is penetrating the tree, update penetrated_env using torch logical_or
            penetrated_env = torch.logical_or(penetrated_env, self.tree_dof_state[:, :, 1].abs().max(dim=-1)[0]>penetration_threshold)

            if frame_count%self.max_episode_length==begin_action_frame//2:
                initial_node_position = self.compute_observations()

            if not self.headless and frame_count%self.max_episode_length==1:
                # For visualizing the actions
                self.gym.clear_lines(self.viewer)
                for env_id in range(self.num_envs):
                    # determine contact node position
                    contact_node_pos = self.rigid_body_state[env_id, self.rb_node_id2handles_list[env_id%self.num_tree_types][contact_nodes[env_id]], :3]
                    # Add Lines for visualization
                    trajectory = trajectory_vector[env_id]
                    line_vertices = torch.cat((contact_node_pos, contact_node_pos + trajectory), dim=0).detach().cpu().numpy()
                    line_colors = [1,0,0] 
                    num_lines = 1
                    self.gym.add_lines(self.viewer, self.envs[env_id], num_lines, line_vertices, line_colors)
            
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            if not self.headless:
                # update the viewer
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

            
            # Record final state if we are at the end of the episode
            if frame_count%self.max_episode_length == self.max_episode_length-1:
                frame_count = 0
                final_node_position = self.compute_observations()
                # Compile to graph
                graphs = self.compile_to_graph(initial_node_position, final_node_position, contact_nodes, trajectory_vector)
                # Exclude the environments that have penetrated the tree
                # This must be done after compiling to graph as compile_to_graph depends on tree order in envs
                graphs = [graphs[i] for i in range(self.num_envs) if not penetrated_env[i]]
                graph_data.extend(graphs)
                penetrated_env = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
                
                print(f'Collected {len(graph_data)} graphs out of {self.cfg["task"]["num_data_collect"]} graphs')
                if len(graph_data) > self.cfg["task"]["num_data_collect"]:
                    graph_data = graph_data[:self.cfg["task"]["num_data_collect"]]
                    print("Saving data...")
                    import pickle
                    out_path = os.path.join(self.cfg["task"]["data_root"], self.cfg["task"]["data_name"])
                    with open(out_path, 'wb') as f:
                        pickle.dump(graph_data, f)
                    break
            else:
                frame_count += 1
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def compile_to_graph(self, initial_node_position, final_node_position, contact_nodes, force_vector):
        # Generate an nx graph for each environment
        graphs = []
        for i in range(self.num_envs):
            g = nx.DiGraph()
            parent2child_edges = self.DiGs_by_id[i%self.num_tree_types].edges()
            parent2child_edges = [(int(parent), int(child)) for parent, child in parent2child_edges]
            #child2parent_edges = [(child, parent) for parent, child in parent2child_edges]
            edges = parent2child_edges #+ child2parent_edges
            g.add_edges_from(edges)            
            
            # Node attributes
            for node_id in range(self.num_nodes):
                g.nodes[node_id]['initial_position'] = initial_node_position[i, node_id].detach().cpu().numpy()
                g.nodes[node_id]['final_position'] = final_node_position[i, node_id].detach().cpu().numpy()
                g.nodes[node_id]['contact_node'] = 1 if node_id == contact_nodes[i] else 0
            # Graph attributes
            g.graph['contact_force'] = force_vector[i].detach().cpu().numpy()
            # Edge attributes
            for edge_idx, (parent, child) in enumerate(parent2child_edges):
                g[parent][child]['initial_edge_delta'] = g.nodes[child]['initial_position'] - g.nodes[parent]['initial_position']
                g[parent][child]['final_edge_delta'] = g.nodes[child]['final_position'] - g.nodes[parent]['final_position']
                g[parent][child]['parent2child'] = 1 
                g[parent][child]['branch'] = 1               
                #g[parent][child]['stiffness'] = self.tree_dof_props_per_asset[i%self.num_tree_types][edge_idx*3]['stiffness']
                g[parent][child]['stiffness'] = self.DiGs_by_id[i%self.num_tree_types].edges[str(parent), str(child)]['stiffness']
            graphs.append(g)
        return graphs


    def compute_observations(self):
        # Get node positions
        node_id2positions = self.get_node_id2positions()
        node_ids = np.arange(0, self.num_nodes, dtype=int)
        node_positions = torch.cat([node_id2positions[node_id] for node_id in node_ids], dim=-1).reshape(-1, self.num_nodes, 3) # shape: (batch_size, num_nodes, 3)
        return node_positions
        
if __name__ == "__main__":
    cfg = OmegaConf.load('cfg/data_collect_cfg.yaml')  
    print(OmegaConf.to_yaml(cfg))
    env = DataCollectionGym(cfg)
    env.simulate()
