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
from utils import utils
from utils import math_utils
from omegaconf import OmegaConf
import networkx as nx

class DataCollectionGym:
    def __init__(self, cfg):
        # initialize gym
        self.gym = gymapi.acquire_gym()
        # Load some config file parameters
        self.cfg = cfg
        self.headless = self.cfg.task.headless
        self.begin_action_frame = self.cfg["env"]["beginActionFrame"]
        self.trajectory_length = self.cfg["env"]["trajectoryLength"]
        self.settling_length = self.cfg["env"]["settlingLength"]
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

        # TESTING
        #min_per_tree = []
        #max_per_tree = []
        #mean_per_tree = []
        #minr_per_tree = []
        #maxr_per_tree = []
        #meanr_per_tree = []
        #for g in self.DiGs_by_id:
        #    stiffness = nx.get_edge_attributes(g, 'stiffness').values()
        #    min_per_tree.append(min(stiffness))
        #    max_per_tree.append(max(stiffness))
        #    mean_per_tree.append(sum(stiffness)/len(stiffness))
        #    radius = nx.get_edge_attributes(g, 'radius').values()
        #    minr_per_tree.append(min(radius))
        #    maxr_per_tree.append(max(radius))
        #    meanr_per_tree.append(sum(radius)/len(radius))
        #print('asset type', self.tree_urdf_paths)
        #print('min per tree:', min(min_per_tree),'-', max(min_per_tree),';', min(minr_per_tree),'-', max(minr_per_tree))
        #print('max per tree:', min(max_per_tree),'-', max(max_per_tree),';', min(maxr_per_tree),'-', max(maxr_per_tree))
        #print('mean per tree:', min(mean_per_tree),'-', max(mean_per_tree), ';', min(meanr_per_tree),'-', max(meanr_per_tree))
        #import sys; sys.exit()
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
        rb_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        rb_contact_force_tensor = gymtorch.wrap_tensor(rb_contact_force_tensor)
        self.rb_contact_force = rb_contact_force_tensor.view(self.num_envs, -1, 3)

        # Dof state slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.tree_dof_state = self.dof_state.view(self.num_envs, -1, 2)

        self.dof_pos = self.dof_state[:, 0].view(self.num_envs, -1, 1) # CHANGED
        self.pos_action = torch.zeros_like(self.dof_pos, device=self.device).squeeze(-1) # CHANGED
    
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
        self.gym.refresh_net_contact_force_tensor(self.sim)

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
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.0005
        sim_params.physx.contact_collection = gymapi.ContactCollection(2)
        sim_params.use_gpu_pipeline = self.cfg["sim"]["use_gpu_pipeline"]
        args = gymutil.parse_arguments()
        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        #self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        self.gym.prepare_sim(self.sim)
        cam_props = gymapi.CameraProperties()
        cam_props.use_collision_geometry = True
        self.viewer = self.gym.create_viewer(self.sim, cam_props)
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
        asset_options.thickness = 0.001
        asset_options.collapse_fixed_joints = False 
        asset_options.armature = 0.01
        asset_options.max_angular_velocity = 40.
        asset_options.max_linear_velocity = 100.
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.density = 1000
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        #asset_options.vhacd_params.resolution = 30_000_000
        #asset_options.vhacd_params.max_convex_hulls = 10
        #asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.replace_cylinder_with_capsule = True
        asset_options.slices_per_cylinder = 100

        tree_assets = []
        for tree_urdf_path in self.tree_urdf_paths:
            tree_asset = self.gym.load_asset(self.sim, self.asset_root, tree_urdf_path, asset_options)
            tree_assets.append(tree_asset)
        asset_options.replace_cylinder_with_capsule = False
        gripper_asset = self.gym.load_asset(self.sim, self.asset_root, "zero_dof_gripper.urdf", asset_options)

        self.num_bodies = self.gym.get_asset_rigid_body_count(tree_asset) + self.gym.get_asset_rigid_body_count(gripper_asset)
        self.num_tree_dofs = self.gym.get_asset_dof_count(tree_asset)
        self.num_dofs = self.num_tree_dofs + self.gym.get_asset_dof_count(gripper_asset) # CHANGED

        print("num bodies: ", self.num_bodies)
        print("num tree dofs: ", self.num_tree_dofs)

        self.tree_dof_props_per_asset = []
        for tree_asset in tree_assets:
            # Set stick dof props by reading directly from URDF
            tree_dof_props = self.gym.get_asset_dof_properties(tree_asset)
            tree_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS) 
            # TODO: Seems like the stiffness parameter is not read from URDF, so we need to set it manually.
            # TODO: The damping parameter is read from URDF, but it is too small. We need to set it manually.
            tree_dof_props['effort'].fill(np.inf)
            tree_dof_props['velocity'].fill(0.01)
            tree_dof_props['armature'].fill(0.1)
            for i in range(self.num_tree_dofs):
                tree_dof_props['stiffness'][i] = tree_dof_props['friction'][i]
            tree_dof_props['friction'].fill(0.0)
            self.tree_dof_props_per_asset.append(tree_dof_props)
        gripper_dof_props = self.gym.get_asset_dof_properties(gripper_asset)
        gripper_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        gripper_dof_props['effort'].fill(2000)
        gripper_dof_props['velocity'].fill(0.05)
        gripper_dof_props['armature'].fill(0.01)
        gripper_dof_props['stiffness'].fill(100_000)
        gripper_dof_props['damping'].fill(100)
        gripper_dof_props['friction'].fill(0.0)

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
        self.num_gripper_bodies = self.gym.get_asset_rigid_body_count(gripper_asset)
        num_gripper_shapes = self.gym.get_asset_rigid_shape_count(gripper_asset)
        self.max_agg_bodies =  num_tree_bodies + self.num_gripper_bodies
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
            gripper_actor = self.gym.create_actor(env_ptr, gripper_asset, gripper_start_pose, "gripper", i, 0, 0) # TODO: Change collision group back
            self.gym.set_actor_dof_properties(env_ptr, gripper_actor, gripper_dof_props)
            
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
        penetration_threshold = 10
        penetrated_env = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim) 
            self.gym.refresh_net_contact_force_tensor(self.sim)
            
            #print(self.dof_state[:, 0].view(self.num_envs, -1)[:,:(self.num_nodes-1)*3].sum())
            #print(self.dof_state[:, 1].view(self.num_envs, -1)[])
            
            # RESET
            if frame_count==0: # CHANGED
                # Reset all environments
                #self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.tree_default_dof_pos))
                # Randomly generate contact node based on num_nodes using numpy
                contact_nodes = np.random.randint(low=1, high=self.num_nodes, size=(self.num_envs, ))
                # Compute initial contact node position and parent node position
                parent_nodes = [int(next(self.DiGs_by_id[i%self.num_tree_types].predecessors(str(contact_nodes[i]))))  
                    for i in range(self.num_envs)]
                parent_node_positions = torch.stack([self.default_node_id2positions_list[i%self.num_tree_types][node_id] for i, node_id in enumerate(parent_nodes)])
                contact_node_positions = torch.stack([self.default_node_id2positions_list[i%self.num_tree_types][node_id] for i, node_id in enumerate(contact_nodes)])
                # Compute branch vectors
                branch_vectors = contact_node_positions - parent_node_positions
                branch_vectors_norm = branch_vectors/torch.linalg.norm(branch_vectors, dim=-1, keepdim=True)
                
                # Generate random trajectory vector to be applied on contact node 
                scale_min =self.cfg["env"]["trajectoryScale"][0]
                scale_max =self.cfg["env"]["trajectoryScale"][1]
                trajectory_distance = torch.rand((self.num_envs, ), device=self.device)*(scale_max-scale_min) + scale_min
                # Random trajectory vector
                t_x = torch.rand((self.num_envs, ), device=self.device)*2 - 1
                t_y = torch.rand((self.num_envs, ), device=self.device)*2 - 1
                t_z = torch.rand((self.num_envs, ), device=self.device)*2 - 1
                trajectory_vector = torch.stack((t_x, t_y, t_z), dim=-1)
                # Project trajectory vector to be orthogonal to branch vector
                trajectory_vector = trajectory_vector - torch.sum(trajectory_vector*branch_vectors_norm, dim=-1, keepdim=True)*branch_vectors_norm
                trajectory_vector_norm = trajectory_vector/torch.linalg.norm(trajectory_vector, dim=-1, keepdim=True)
                # Apply random tilt to the trajectory vector with respect to the plane perpendicular to the branch vector
                # Random tilt angle constrained to be between +-20 degrees
                random_tilt_angle = (torch.rand((self.num_envs, ), device=self.device)*90 - 45)*3.1415/180
                # Compute axis_angle representation of rotation
                rotation_axis = torch.cross(branch_vectors_norm, trajectory_vector_norm, dim=-1)
                rotation_axis_norm = rotation_axis/torch.linalg.norm(rotation_axis, dim=-1, keepdim=True)
                axis_angle = rotation_axis_norm * random_tilt_angle.unsqueeze(-1)
                # Compute rotation matrix
                rotation_matrix = math_utils.axis_angle_to_matrix(axis_angle)
                # Rotate trajectory vector
                trajectory_vector = torch.bmm(rotation_matrix, trajectory_vector.unsqueeze(-1)).squeeze(-1)
                # Normalize trajectory vector
                trajectory_vector_norm = trajectory_vector/torch.linalg.norm(trajectory_vector, dim=-1, keepdim=True)
                trajectory_vector = trajectory_distance.unsqueeze(-1)*trajectory_vector_norm


                # Compute end-effector trajectory
                #self.gripper_trajectory = torch.zeros((self.num_envs, self.trajectory_length, 3), dtype=torch.float32, device=self.device, requires_grad=False)
                # NOTE: Slightly offsetting along branch vector and force vector to avoid losing contact with branch
                #self.gripper_trajectory[:, 0] = contact_node_positions - 0.01*branch_vectors_norm - 0.1*trajectory_vector_norm
                #for i in range(1, self.trajectory_length):
                #    self.gripper_trajectory[:, i] = self.gripper_trajectory[:, 0] + i*((0.1+trajectory_distance).unsqueeze(-1)*trajectory_vector_norm/self.trajectory_length)
                # Orient the end-effector such that z axis aligns with force vector
                self.gripper_quat = utils.get_quat_from_vec(trajectory_vector_norm, branch_vectors_norm, gripper_axis='z')

                multi_env_ids_int32 = self._global_indices[:, self.gripper_ind].flatten().contiguous()                    
                #self.root_state_tensor[:, self.gripper_ind, :3] = self.gripper_trajectory[:, 0] 
                self.root_state_tensor[:, self.gripper_ind, :3] = contact_node_positions - 0.01*branch_vectors_norm - 0.05*trajectory_vector_norm
                self.root_state_tensor[:, self.gripper_ind, 3:7] = self.gripper_quat
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                    gymtorch.unwrap_tensor(multi_env_ids_int32), 
                    len(multi_env_ids_int32))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.tree_default_dof_pos))
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(torch.zeros_like(self.pos_action)))
            elif frame_count > self.begin_action_frame and frame_count < self.begin_action_frame + self.trajectory_length:
                self.pos_action[:, -1] = (trajectory_distance+0.05)*(frame_count - self.begin_action_frame)/self.trajectory_length
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

            
            # If the gripper is penetrating the tree, update penetrated_env using torch logical_or
            penetrated_env = torch.logical_or(penetrated_env, self.tree_dof_state[:, :-1, 1].abs().max(dim=-1)[0]>penetration_threshold) # CHANGED -1

            if frame_count==self.begin_action_frame-1:
                # Record initial state
                initial_node_position = self.compute_observations()
                # Record those that are in contact with the gripper
                initial_in_contact = self.rb_contact_force[:, -self.num_gripper_bodies:].abs().sum(dim=-1).sum(dim=-1) > 0.0 
                if not self.headless:
                    for env_id in range(self.num_envs):
                        # determine contact node position
                        contact_node_pos = self.rigid_body_state[env_id, self.rb_node_id2handles_list[env_id%self.num_tree_types][contact_nodes[env_id]], :3]
                        # Add Lines for visualization
                        trajectory = trajectory_vector[env_id]
                        line_vertices = torch.cat((contact_node_pos, contact_node_pos + trajectory), dim=0).detach().cpu().numpy()
                        line_colors = [1,0,0] 
                        num_lines = 1
                        self.gym.add_lines(self.viewer, self.envs[env_id], num_lines, line_vertices, line_colors)
            
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
            if frame_count==self.begin_action_frame + self.trajectory_length + self.settling_length: # Episode length
                if not self.headless:
                    self.gym.clear_lines(self.viewer)
                frame_count = 0
                final_node_position = self.compute_observations()
                # If the gripper is not in contact with the tree, exclude from data
                final_not_in_contact = self.rb_contact_force[:, -self.num_gripper_bodies:].abs().sum(dim=-1).sum(dim=-1) == 0.0
                # If initial position and final position is the same, exclude from data
                print('-----------------------------------------------')
                #print('fast dof:', penetrated_env.sum().item())
                #displacement = torch.linalg.norm(initial_node_position - final_node_position, dim=-1)                 
                #print('tree not moving:', torch.all(displacement<0.01, dim=-1).sum().item())
                #penetrated_env = torch.logical_or(penetrated_env, torch.all(displacement<0.01, dim=-1))
                #print('num_tree out:', penetrated_env.sum().item())
                print(f'fast dof {penetrated_env.sum().item()}')
                penetrated_env = torch.logical_or(penetrated_env, initial_in_contact)
                print(f'initial contact {initial_in_contact.sum().item()}')
                penetrated_env = torch.logical_or(penetrated_env, final_not_in_contact)
                print(f'final contact {final_not_in_contact.sum().item()}')
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
                g[parent][child]['radius'] = self.DiGs_by_id[i%self.num_tree_types].edges[str(parent), str(child)]['radius']
                g[parent][child]['length'] = self.DiGs_by_id[i%self.num_tree_types].edges[str(parent), str(child)]['length']
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
