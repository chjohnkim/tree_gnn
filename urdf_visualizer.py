import os
import sys
import pickle
import numpy as np
from urdf_tree_generator import URDFTreeGenerator
from isaacgym import gymapi, gymutil, gymtorch
import torch
import utils
from copy import deepcopy
import cv2
import pyautogui
import time

class URDFVisualizer:
    def __init__(self, args, graph_initial, graph_final, graph_predicted, 
                 contact_node_gt, contact_force_gt, contact_node=None, contact_force=None, auto_close=np.inf):
        self.headless = False
        self.args = args
        self.graph_initial = graph_initial
        self.graph_final = graph_final
        self.graph_predicted = graph_predicted
        self.contact_node_gt = contact_node_gt
        self.contact_force_gt = contact_force_gt
        self.contact_node = contact_node
        self.contact_force = contact_force
        
        self.mode = args.mode
        self.begin_action_frame = 50
        self.trajectory_length = 100
        self.settling_length = auto_close 
        self.auto_close = auto_close
        
        self.use_gripper = self.mode=='gripper_contact' and self.contact_node is not None

        self.node_analysis = False
        if self.contact_node is not None:
            if self.contact_node.shape!=contact_node_gt.shape:
                self.node_analysis = True

        self.num_actors = 3 
        if self.use_gripper:
            self.num_actors = 4
        self.num_nodes = len(self.graph_initial[0].nodes)
        if self.node_analysis:
            self.num_envs = self.num_nodes
            self.contact_node_sort = torch.argsort(self.contact_node, descending=True)
            # Copy the contents graph_initial, graph_final, and graph_predicted for each node
            self.graph_initial = [deepcopy(self.graph_initial[0]) for _ in range(self.num_envs)]
            self.graph_final = [deepcopy(self.graph_final[0]) for _ in range(self.num_envs)]
            self.graph_predicted = [deepcopy(self.graph_predicted[0]) for _ in range(self.num_envs)]
            # Copy the contents of contact_node_gt and contact_force_gt for each node as tensors
            self.contact_node_gt = torch.stack([self.contact_node_gt for _ in range(self.num_envs)])
            self.contact_force_gt = torch.stack([self.contact_force_gt for _ in range(self.num_envs)])
        else:
            self.num_envs = len(self.graph_predicted)
        self.gym = gymapi.acquire_gym()
        self.initialize_sim()

        self.asset_root = '.'

        # Generate URDFs for initial, final, and predicted trees
        self.assets_initial = []
        self.assets_final = []
        self.assets_predicted = []
        self.DiGs_by_id = []
        for i in range(self.num_envs):
            tree_urdf = URDFTreeGenerator(self.graph_initial[i], f'temp_tree_urdf', asset_path=self.asset_root)
            DiG_by_id = utils.parse_urdf_graph(os.path.join(self.asset_root, tree_urdf.save_file_name))
            self.DiGs_by_id.append(DiG_by_id)
            self.assets_initial.append(self.load_asset(tree_urdf.save_file_name))
            
            tree_urdf = URDFTreeGenerator(self.graph_final[i], f'temp_tree_urdf', asset_path=self.asset_root)
            self.assets_final.append(self.load_asset(tree_urdf.save_file_name))
            
            tree_urdf = URDFTreeGenerator(self.graph_predicted[i], f'temp_tree_urdf', asset_path=self.asset_root)
            self.assets_predicted.append(self.load_asset(tree_urdf.save_file_name))
        self.create_env()
        self.create_viewer()
        self.visualize()

    def initialize_sim(self):
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        sim_params.substeps = 2
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 12
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = True
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.0005
        sim_params.use_gpu_pipeline = True

        #sim_params.dt = dt = 1.0 / 60.0
        #sim_params.physx.contact_offset = 0.0
        #sim_params.physx.rest_offset = 0.0
        #sim_params.physx.bounce_threshold_velocity = 0.001
        #sim_params.physx.max_depenetration_velocity = 0.001
        #sim_params.physx.default_buffer_size_multiplier = 5.0
        #sim_params.physx.max_gpu_contact_pairs = 1048576
        #sim_params.physx.num_subscenes = 4
        #sim_params.physx.contact_collection = 0
        
        if self.args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")  
        self.device = self.args.sim_device if args.use_gpu_pipeline else 'cpu'
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
    
    def create_viewer(self):
        # create viewer
        camera_props = gymapi.CameraProperties()
        #camera_props.horizontal_fov = 90.0
        #camera_props.width = 1920
        #camera_props.height = 1080
        #camera_props.supersampling_horizontal = 1
        #camera_props.supersampling_vertical = 1
        camera_props.use_collision_geometry = True
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")

        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        # position the camera
        cam_pos = gymapi.Vec3(0.8, 0.8, 0.8)
        cam_target = gymapi.Vec3(-1, -1, 0.3)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def load_asset(self, asset_file):
        # load asset
        #asset_file = urdf_tree.save_file_name

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
        asset_options.replace_cylinder_with_capsule = True
        asset_options.slices_per_cylinder = 100

        print("Loading asset '%s' from '%s'" % (asset_file, self.asset_root))
        asset = self.gym.load_asset(self.sim, self.asset_root, asset_file, asset_options)
        urdf_file_path = os.path.join(self.asset_root, asset_file)
        if os.path.exists(urdf_file_path):
            os.remove(urdf_file_path)
        return asset
    
    def create_env(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.02
        #gym.add_ground(sim, plane_params)


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
        asset_options.replace_cylinder_with_capsule = False
        asset_options.slices_per_cylinder = 100
        gripper_asset = self.gym.load_asset(self.sim, self.asset_root, os.path.join("assets", "zero_dof_gripper.urdf"), asset_options)

        # create an array of DOF states that will be used to update the actors
        tree_num_dofs = self.gym.get_asset_dof_count(self.assets_initial[0])
        self.num_rigid_bodies = self.gym.get_asset_rigid_body_count(self.assets_initial[0])*3
        if self.use_gripper:
            num_dofs = tree_num_dofs + self.gym.get_asset_dof_count(gripper_asset)
            self.num_rigid_bodies += self.gym.get_asset_rigid_body_count(gripper_asset)
        else:
            num_dofs = tree_num_dofs


        # get array of DOF properties
        self.tree_dof_props_per_asset = []
        for tree_asset in self.assets_initial:
            dof_props = self.gym.get_asset_dof_properties(tree_asset)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS) 
            dof_props['effort'].fill(np.inf)
            dof_props['velocity'].fill(0.01)
            dof_props['armature'].fill(0.1)
            for i in range(tree_num_dofs):
                dof_props['stiffness'][i] = dof_props['friction'][i]    
            dof_props['friction'].fill(0.0)
            self.tree_dof_props_per_asset.append(dof_props)
        # Setup gripper dof props
        gripper_dof_props = self.gym.get_asset_dof_properties(gripper_asset)
        gripper_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        gripper_dof_props['effort'].fill(2000)
        gripper_dof_props['velocity'].fill(0.05)
        gripper_dof_props['armature'].fill(0.01)
        gripper_dof_props['stiffness'].fill(100_000)
        gripper_dof_props['damping'].fill(100)
        gripper_dof_props['friction'].fill(0.0) 

        # set up the env grid
        if self.node_analysis:
            num_per_row = self.num_nodes+1
            spacing = 0.5
        else:
            num_per_row = int(np.sqrt(self.num_envs))
            spacing = 2.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        print("Creating %d environments" % self.num_envs)
        self.envs = []
        self.initial_actors = []
        self.final_actors = []
        self.predicted_actors = []
        self.grippers = []
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            # add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            pose.r = gymapi.Quat()
            dof_props = self.tree_dof_props_per_asset[i]
            # Initial tree actor
            actor_handle = self.gym.create_actor(env, self.assets_initial[i], pose, "actor", 0, 1, 0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)
            self.initial_actors.append(actor_handle) 

            # Final tree actor
            actor_handle = self.gym.create_actor(env, self.assets_final[i], pose, "actor", 1, 1, 0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)
            self.final_actors.append(actor_handle) 
            color = gymapi.Vec3(0.0, 1.0, 0.0)
            rb_names = self.gym.get_actor_rigid_body_names(env, actor_handle)
            for rb_name in rb_names:
                rb_idx = self.gym.find_actor_rigid_body_index(env, actor_handle, rb_name, gymapi.DOMAIN_ACTOR)
                self.gym.set_rigid_body_color(env, actor_handle, rb_idx, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # Predicted tree actor
            actor_handle = self.gym.create_actor(env, self.assets_predicted[i], pose, "actor", 2, 1, 0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)
            self.predicted_actors.append(actor_handle)     
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            rb_names = self.gym.get_actor_rigid_body_names(env, actor_handle)
            for rb_name in rb_names:
                rb_idx = self.gym.find_actor_rigid_body_index(env, actor_handle, rb_name, gymapi.DOMAIN_ACTOR)
                self.gym.set_rigid_body_color(env, actor_handle, rb_idx, gymapi.MESH_VISUAL_AND_COLLISION, color)
            if self.use_gripper:
                # Create robot actor
                gripper_start_pose = gymapi.Transform()
                gripper_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
                gripper_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                gripper_actor = self.gym.create_actor(env, gripper_asset, gripper_start_pose, "gripper", 2, 0, 0)
                self.gym.set_actor_dof_properties(env, gripper_actor, gripper_dof_props)
                self.grippers.append(gripper_actor)
        self.gym.prepare_sim(self.sim)

        # Acquire gym tensor
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.dof_pos = self.dof_state[:, 0].view(self.num_envs, -1, 1) # CHANGED
        self.pos_action = torch.zeros_like(self.dof_pos, device=self.device) # CHANGED
        # Default tree dof pos: tree is passive, hence the default dof pos is 0
        self.tree_default_dof_pos = torch.zeros_like(self.dof_state, device=self.device)

        # Global inds
        self._global_indices = torch.arange(
            self.num_envs * self.num_actors, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

        self.refresh()

        self.asset_rb_handles_dict_list = []
        for i in range(self.num_envs):
            asset_rb_handles_dict = {
                'initial': {},
                'final': {},
                'predicted': {}
            }
            for node in self.graph_initial[i].nodes:
                rb_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.initial_actors[i], f"node_{node}")
                asset_rb_handles_dict['initial'][node] = rb_handle
            for node in self.graph_final[i].nodes:
                rb_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.final_actors[i], f"node_{node}")
                asset_rb_handles_dict['final'][node] = rb_handle
            for node in self.graph_predicted[i].nodes:
                rb_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.predicted_actors[i], f"node_{node}")
                asset_rb_handles_dict['predicted'][node] = rb_handle            
            self.asset_rb_handles_dict_list.append(asset_rb_handles_dict)
        if self.use_gripper:
            self.gripper_ind = self.gym.get_actor_index(env, gripper_actor, gymapi.DOMAIN_ENV)

        node_id2positions = self.get_node_id2positions()
        self.default_node_id2positions_list = []
        for i in range(self.num_envs):
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
            node_handles = [self.asset_rb_handles_dict_list[i]['initial'][node_id] for i in range(self.num_envs)]
            node_id2positions[node_id] = self.rigid_body_state[torch.arange(self.num_envs), node_handles, :3]
        return node_id2positions

    def visualize(self):
        frame = 0
        if self.mode=='virtual_force':
            visual_scaling_factor = 0.001
        elif self.mode=='gripper_contact':
            visual_scaling_factor = 1
        rb_force_tensor = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), device=self.device, dtype=torch.float)
        # Initialize wireframe sphere geometry for visualizing nodes
        contact_node_geom = gymutil.WireframeSphereGeometry(0.025, 10, 10, gymapi.Transform(), color=(1, 0, 0))
        contact_node_gt_geom = gymutil.WireframeSphereGeometry(0.02, 10, 10, gymapi.Transform(), color=(0, 1, 0))
        # variables to detect object penetration
        #begin_action_frame = 20
        #trajectory_num_waypoints = 100 if self.auto_close==np.inf else self.auto_close-begin_action_frame
        penetration_threshold = 10
        penetrated_env = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        
        if self.args.record_video:
            left, top, width, height = 470, 112, 840, 840 #self.gym.get_viewer_size(self.viewer).x, self.gym.get_viewer_size(self.viewer).y 
            # Create a video writer
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            fps = 20.0
            timestr = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join('videos', f'{self.num_nodes}_nodes', f'{self.num_nodes}_{timestr}.mp4')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))


        
        while not self.gym.query_viewer_has_closed(self.viewer):
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.refresh()

            if self.use_gripper:
                if frame==0:
                    # Compute initial contact node position and parent node position
                    if self.node_analysis:
                        contact_nodes = self.contact_node_sort
                        gripper_trajectory = self.contact_force[contact_nodes]
                    else:
                        contact_nodes = torch.tensor(self.contact_node)
                        gripper_trajectory = self.contact_force
                    parent_nodes = []
                    for i in range(self.num_envs):
                        if contact_nodes[i].item()==0:
                            parent_nodes.append(-1)
                        else: 
                            parent_nodes.append(int(next(self.DiGs_by_id[i].predecessors(str(contact_nodes[i].item())))))
                    
                    parent_node_positions = torch.stack([self.default_node_id2positions_list[i%self.num_envs][node_id] if node_id!=-1 
                                                         else torch.zeros((3,), dtype=torch.float32, device=self.device) 
                                                         for i, node_id in enumerate(parent_nodes)])
                    contact_node_positions = torch.stack([self.default_node_id2positions_list[i%self.num_envs][node_id.item()] for i, node_id in enumerate(contact_nodes)])
                    # Compute branch vectors
                    branch_vectors = contact_node_positions - parent_node_positions
                    branch_vectors_norm = branch_vectors/torch.linalg.norm(branch_vectors, dim=-1, keepdim=True)
                    trajectory_vector_norm = gripper_trajectory / torch.linalg.norm(gripper_trajectory, dim=-1, keepdim=True)   
                    trajectory_distance = torch.linalg.norm(gripper_trajectory, dim=-1, keepdim=True)
                    # Compute end-effector trajectory
                    #self.gripper_trajectory = torch.zeros((self.num_envs, self.trajectory_length, 3), dtype=torch.float32, device=self.device, requires_grad=False)
                    # NOTE: Slightly offsetting along branch vector and force vector to avoid losing contact with branch
                    #self.gripper_trajectory[:, 0] = contact_node_positions - 0.01*branch_vectors_norm - 0.05*trajectory_vector_norm

                    #for i in range(1, self.trajectory_length):
                    #    self.gripper_trajectory[:, i] = self.gripper_trajectory[:, 0] + i*((0.05+trajectory_distance)*trajectory_vector_norm/self.trajectory_length)
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
                elif frame > self.begin_action_frame and frame < self.begin_action_frame + self.trajectory_length:
                    self.pos_action[:, -1] = (trajectory_distance+0.05)*(frame - self.begin_action_frame)/self.trajectory_length
                    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
                

            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                # Visualize GT contact node position
                contact_node_gt_pos = self.rigid_body_state[i, self.asset_rb_handles_dict_list[i]['initial'][self.contact_node_gt[i].item()], :3]
                geom_pose = gymapi.Transform()
                geom_pose.p = gymapi.Vec3(contact_node_gt_pos[0], contact_node_gt_pos[1], contact_node_gt_pos[2])
                gymutil.draw_lines(contact_node_gt_geom, self.gym, self.viewer, self.envs[i], geom_pose)
                # Visualize GT contact force
                rb_force_scaled = (self.contact_force_gt[i]*visual_scaling_factor).flatten().detach().cpu().numpy()
                line_vertices = np.stack((contact_node_gt_pos.detach().cpu().numpy(), contact_node_gt_pos.detach().cpu().numpy() + rb_force_scaled), axis=0)
                line_color = [0,1,0] 
                num_lines = 1
                self.gym.add_lines(self.viewer, self.envs[i], num_lines, line_vertices, line_color)

                if self.contact_node is not None: # If we are visualizing control policy predictions            
                    # Visualize predicted contact node position
                    if self.node_analysis:
                        contact_node = self.contact_node_sort
                        contact_force = self.contact_force[contact_node]
                    else:
                        contact_node = self.contact_node
                        contact_force = self.contact_force
                    contact_node_pos = self.rigid_body_state[i, self.asset_rb_handles_dict_list[i]['initial'][contact_node[i].item()], :3]
                    rb_force_scaled = (contact_force[i]*visual_scaling_factor).flatten().detach().cpu().numpy()
                    geom_pose.p = gymapi.Vec3(contact_node_pos[0], contact_node_pos[1], contact_node_pos[2])
                    gymutil.draw_lines(contact_node_geom, self.gym, self.viewer, self.envs[i], geom_pose)                    
                    # Visualize predicted contact force
                    line_vertices = np.stack((contact_node_pos.detach().cpu().numpy(), contact_node_pos.detach().cpu().numpy() + rb_force_scaled), axis=0)
                    line_color = [1,0,0] 
                    num_lines = 1
                    self.gym.add_lines(self.viewer, self.envs[i], num_lines, line_vertices, line_color)
                    # Simulate predicted contact force deformation    
                    if frame>self.begin_action_frame:
                        rb_force_tensor[i, self.asset_rb_handles_dict_list[i]['predicted'][contact_node[i].item()], :3] = contact_force[i]

            if self.mode=='virtual_force':
                self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(rb_force_tensor), None, gymapi.ENV_SPACE)
                
            
            if not self.headless:
                # update the viewer
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)
            else: 
                self.gym.poll_viewer_events(self.viewer)

            if frame>self.begin_action_frame + self.trajectory_length + self.settling_length:
                import json
                if self.node_analysis:
                    initial_tree_rb_handles = list(self.asset_rb_handles_dict_list[0]['initial'].values())
                    predicted_tree_rb_handles = list(self.asset_rb_handles_dict_list[0]['predicted'].values())
                    target_tree_rb_handles = list(self.asset_rb_handles_dict_list[0]['final'].values())
                    initial_node_pos = self.rigid_body_state[:, initial_tree_rb_handles, :3]
                    predicted_node_pos = self.rigid_body_state[:, predicted_tree_rb_handles, :3]
                    target_node_pos = self.rigid_body_state[:, target_tree_rb_handles, :3]
                    # Compute metrics
                    dist_error = torch.norm(predicted_node_pos-target_node_pos, dim=-1)
                    mean_dist_error = torch.mean(dist_error, dim=-1).cpu().numpy().tolist()
                    max_dist_error = torch.max(dist_error, dim=-1)[0].cpu().numpy().tolist()
                    node_probs = self.contact_node[self.contact_node_sort].cpu().numpy().tolist()
                    node_indices = self.contact_node_sort.cpu().numpy().tolist()
                    data = {'mean_dist_error': mean_dist_error, 'max_dist_error': max_dist_error, 'node_probs': node_probs, 'node_indices': node_indices}
                else:
                    # Compute the maximum node distance between the predicted and target tree
                    # and the maximum node displacement between the initial and target tree
                    max_node_displacements = []
                    max_node_dist_errors = []
                    mean_node_displacements = []
                    mean_node_dist_errors = []
                    for i in range(self.num_envs):
                        asset_rb_handles_dict = self.asset_rb_handles_dict_list[i]
                        initial_tree_rb_handles = list(asset_rb_handles_dict['initial'].values())
                        predicted_tree_rb_handles = list(asset_rb_handles_dict['predicted'].values())
                        target_tree_rb_handles = list(asset_rb_handles_dict['final'].values())
                        initial_node_pos = self.rigid_body_state[i, initial_tree_rb_handles, :3]
                        predicted_node_pos = self.rigid_body_state[i, predicted_tree_rb_handles, :3]
                        target_node_pos = self.rigid_body_state[i, target_tree_rb_handles, :3]
                        node_dist_error = torch.norm(predicted_node_pos-target_node_pos, dim=-1)
                        node_displacement = torch.norm(target_node_pos-initial_node_pos, dim=-1)
                        max_node_displacement = torch.max(node_displacement).item()
                        max_node_dist_error = torch.max(node_dist_error).item()
                        max_node_displacements.append(max_node_displacement)
                        max_node_dist_errors.append(max_node_dist_error)
                        mean_node_displacement = torch.mean(node_displacement).item()
                        mean_node_dist_error = torch.mean(node_dist_error).item()
                        mean_node_displacements.append(mean_node_displacement)
                        mean_node_dist_errors.append(mean_node_dist_error)
                    data = {'max_node_displacement': max_node_displacements, 'max_node_dist_error': max_node_dist_errors,
                            'mean_node_displacement': mean_node_displacements, 'mean_node_dist_error': mean_node_dist_errors}
                serialized_data = json.dumps(data)
                print("DATA_START")
                print(serialized_data)
                print("DATA_END")
                sys.stdout.flush()
                break
            else:
                frame+=1
                if self.args.record_video:
                    # Capture a screenshot of the selected window
                    if frame%(60//fps)==0:
                        screenshot = pyautogui.screenshot(region=(left, top, width, height))
                        f = np.array(screenshot)
                        # Convert BGR to RGB format (OpenCV uses BGR)
                        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        # Write the frame to the video file
                        out.write(f)  
        # Release the video writer and close the window
        if self.args.record_video:
            out.release()
        self.destroy()

    def refresh(self):
        # refresh all environments
        self.gym.refresh_actor_root_state_tensor(self.sim) 
        self.gym.refresh_dof_state_tensor(self.sim) 
        self.gym.refresh_rigid_body_state_tensor(self.sim) 

    def destroy(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        #print("Safely destroyed viewer and sim!")

if __name__ == "__main__":
    args = gymutil.parse_arguments(
        description="Visualize URDF",
        custom_parameters=[
            {"name": "--trunk_radius", "type": float, "default": 0.02, "help": "Radius of trunk"},
            {"name": "--temp_file", "type": str, "help": "Path to temp file"},
            {"name": "--mode", "type": str, "help": "Either virtual force or gripper contact"},
            {"name": "--record_video", "type": bool, "default": False, "help": "Whether to save video"}
        ])

    # Deserialize the data from temp file using pickle
    with open(args.temp_file, 'rb') as temp_file:
        serialized_data = temp_file.read()
        data = pickle.loads(serialized_data)
    visualizer = URDFVisualizer(args, *data)

