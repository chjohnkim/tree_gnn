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
import robot_utils
import motion_planning_utils as mpu
from isaacgym.torch_utils import to_torch
import math_utils

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
        
        self.per_node_affordance = contact_node
        self.per_node_vector = contact_force

        self.begin_action_frame = 50
        self.init_traj_length = 100
        self.buffer_length = 50
        self.trajectory_length = 100
        self.settling_length = auto_close 
        self.auto_close = auto_close
        

        self.num_actors = 3
        self.num_nodes = len(self.graph_initial[0].nodes)
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
        
        self.initialize_robot()
        self.create_env()
        self.create_viewer()
        self.visualize()

    def initialize_robot(self):
        self.robot_kinematics = robot_utils.RobotKinematics('ur5')
        self.robot_asset_path = "assets/ur5_description/ur5.urdf"
        self.ee_link_name = "wrist_3_link"
        # Initialize motion planner
        mp_robot = mpu.URDFAsset(os.path.join(self.asset_root, self.robot_asset_path), 
                                        pos=[0,0,0], rot=[0,0,0], base_link_name='base_link')
        c_space = mpu.ObstacleCSpace(eps=1e-1)
        c_space.add_robot(mp_robot)

        tree_urdf = URDFTreeGenerator(self.graph_initial[0], f'temp_tree_urdf', asset_path=self.asset_root)
        mp_tree = mpu.URDFAsset(os.path.join(self.asset_root, tree_urdf.save_file_name), 
                                        pos=[0,0,0], rot=[0,0,0], base_link_name='node_0')
        c_space.add_obstacle(mp_tree)
        self.planner = mpu.MotionPlanner(c_space)
        self.load_asset(tree_urdf.save_file_name)

        # Default robot dof pos
        self.robot_default_dof_pos = to_torch([(1/2)*np.pi, (-2/3)*np.pi, (4/5)*np.pi, (-1/7)*np.pi, (1/2)*np.pi, 0.0], device=self.device)
        self.robot_default_dof_pos = self.robot_default_dof_pos.repeat(self.num_envs, 1)
        # Default robot root state
        self.robot_default_pos = to_torch([[0.5, 0.0, 0.1]], device=self.device)
        self.robot_default_quat = to_torch([[0.0, 0.0, 0.0, 1.0]], device=self.device)

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
        self.gym.add_ground(self.sim, plane_params)


        asset_options = gymapi.AssetOptions()
        '''
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
        '''
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        robot_asset = self.gym.load_asset(self.sim, self.asset_root, self.robot_asset_path, asset_options)

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        # arm
        robot_dof_props["stiffness"].fill(40000.0)
        robot_dof_props["damping"].fill(40.0)
        robot_dof_props["effort"].fill(800.0)
        robot_dof_props["lower"].fill(-np.pi*4)
        robot_dof_props["upper"].fill(np.pi*4)

        robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        print(robot_link_dict)
        self.robot_ee_index = robot_link_dict[self.ee_link_name]




        # create an array of DOF states that will be used to update the actors
        tree_num_dofs = self.gym.get_asset_dof_count(self.assets_initial[0])
        self.num_rigid_bodies = self.gym.get_asset_rigid_body_count(self.assets_initial[0])*2
        #num_dofs = tree_num_dofs + self.gym.get_asset_dof_count(robot_asset)
        self.num_rigid_bodies += self.gym.get_asset_rigid_body_count(robot_asset)


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
        # set up the env grid
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 2.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        print("Creating %d environments" % self.num_envs)
        self.envs = []
        self.initial_actors = []
        self.final_actors = []
        self.predicted_actors = []
        self.robots = []
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
            #actor_handle = self.gym.create_actor(env, self.assets_initial[i], pose, "actor", 0, 1, 0)
            #self.gym.set_actor_dof_properties(env, actor_handle, dof_props)
            #self.initial_actors.append(actor_handle) 

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
            '''
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            rb_names = self.gym.get_actor_rigid_body_names(env, actor_handle)
            for rb_name in rb_names:
                rb_idx = self.gym.find_actor_rigid_body_index(env, actor_handle, rb_name, gymapi.DOMAIN_ACTOR)
                self.gym.set_rigid_body_color(env, actor_handle, rb_idx, gymapi.MESH_VISUAL_AND_COLLISION, color)
            '''
            # Create robot actor
            robot_start_pose = gymapi.Transform()
            robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            robot_actor = self.gym.create_actor(env, robot_asset, robot_start_pose, "robot", 2, 0, 0)
            self.gym.set_actor_dof_properties(env, robot_actor, robot_dof_props)
            self.robots.append(robot_actor)
        self.gym.prepare_sim(self.sim)




        # Acquire gym tensor
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self.j_eef = jacobian[:, self.robot_ee_index-1, :, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.dof_pos = self.dof_state[:, 0].view(self.num_envs, -1, 1) # CHANGED
        self.pos_action = torch.zeros_like(self.dof_pos, device=self.device) # CHANGED
        # Default tree dof pos: tree is passive, hence the default dof pos is 0
        self.tree_default_dof_pos = torch.zeros_like(self.dof_state, device=self.device)
        self.tree_default_dof_pos[-6:, 0] = self.robot_default_dof_pos.squeeze()

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
            #for node in self.graph_initial[i].nodes:
            #    rb_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.initial_actors[i], f"node_{node}")
            #    asset_rb_handles_dict['initial'][node] = rb_handle
            for node in self.graph_final[i].nodes:
                rb_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.final_actors[i], f"node_{node}")
                asset_rb_handles_dict['final'][node] = rb_handle
            for node in self.graph_predicted[i].nodes:
                rb_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.predicted_actors[i], f"node_{node}")
                asset_rb_handles_dict['predicted'][node] = rb_handle            
            self.asset_rb_handles_dict_list.append(asset_rb_handles_dict)
        self.robot_ind = self.gym.get_actor_index(env, robot_actor, gymapi.DOMAIN_ENV)

        node_id2positions = self.get_node_id2positions()
        self.default_node_id2positions_list = []
        for i in range(self.num_envs):
            default_node_id2positions = {}
            for node_id in range(self.num_nodes):
                default_node_id2positions[node_id] = deepcopy(node_id2positions[node_id][i])
            self.default_node_id2positions_list.append(default_node_id2positions)
        
        # Robot gripper
        self.ee_rb_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], self.ee_link_name)


    def get_node_id2positions(self):
        # nodes and goal position slices 
        node_id2positions = {} 
        for node_id in range(self.num_nodes):
            # NOTE: This part is probably not stable when num_envs is not divisible by num_tree_types. 
            # Hence we do an assert during initialization to make sure num_envs is divisible by num_tree_types
            node_handles = [self.asset_rb_handles_dict_list[i]['predicted'][node_id] for i in range(self.num_envs)]
            node_id2positions[node_id] = self.rigid_body_state[torch.arange(self.num_envs), node_handles, :3]
        return node_id2positions

    def visualize(self):
        frame = 0
        visual_scaling_factor = 1
        rb_force_tensor = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), device=self.device, dtype=torch.float)
        # Initialize wireframe sphere geometry for visualizing nodes
        contact_node_geom = gymutil.WireframeSphereGeometry(0.025, 10, 10, gymapi.Transform(), color=(1, 0, 0))
        contact_node_gt_geom = gymutil.WireframeSphereGeometry(0.02, 10, 10, gymapi.Transform(), color=(0, 1, 0))
        # variables to detect object penetration
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
            # Doing this to avoid the viewer from glitching
            robot_utils.control_ik(self.j_eef, torch.zeros(self.num_envs,6,1, device=self.device), self.num_envs)
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.refresh()
            if frame==0:
                # Get the sorted order index of the affordance values in decreasing order
                affordance_rank = torch.argsort(self.per_node_affordance, dim=-1, descending=True)
                for rank, node_idx in enumerate(affordance_rank):
                    trajectory_vector = self.per_node_vector[node_idx].unsqueeze(0)

                    contact_nodes = torch.tensor(node_idx).unsqueeze(0)
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
                    trajectory_vector_norm = trajectory_vector / torch.linalg.norm(trajectory_vector, dim=-1, keepdim=True)   
                    trajectory_distance = torch.linalg.norm(trajectory_vector, dim=-1, keepdim=True)
                    # Compute end-effector trajectory
                    #self.gripper_trajectory = torch.zeros((self.num_envs, self.trajectory_length, 3), dtype=torch.float32, device=self.device, requires_grad=False)
                    # NOTE: Slightly offsetting along branch vector and force vector to avoid losing contact with branch
                    #self.gripper_trajectory[:, 0] = contact_node_positions - 0.01*branch_vectors_norm - 0.05*trajectory_vector_norm

                    #for i in range(1, self.trajectory_length):
                    #    self.gripper_trajectory[:, i] = self.gripper_trajectory[:, 0] + i*((0.05+trajectory_distance)*trajectory_vector_norm/self.trajectory_length)
                    # Orient the end-effector such that z axis aligns with force vector
                    self.gripper_quat = utils.get_quat_from_vec(trajectory_vector_norm, branch_vectors_norm, gripper_axis='z')
                
                    multi_env_ids_int32 = self._global_indices[:, self.robot_ind].flatten().contiguous()
                    #self.root_state_tensor[:, self.gripper_ind, :3] = self.gripper_trajectory[:, 0]
                    #self.root_state_tensor[:, self.robot_ind, :3] = contact_node_positions - 0.01*branch_vectors_norm - 0.05*trajectory_vector_norm
                    #self.root_state_tensor[:, self.robot_ind, 3:7] = self.gripper_quat
                    self.root_state_tensor[:, self.robot_ind, :3] = self.robot_default_pos
                    self.root_state_tensor[:, self.robot_ind, 3:7] = self.robot_default_quat
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                        gymtorch.unwrap_tensor(multi_env_ids_int32), 
                        len(multi_env_ids_int32))
                    self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.tree_default_dof_pos))
                    self.pos_action[:,-6:] = self.robot_default_dof_pos.unsqueeze(-1)
                    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

                    # Declare variables for computing end-effector trajectory
                    self.found_motion_plan = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
                    self.plan_exhausted = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
                    
                    self.ee_traj_joint_angles = torch.clone(self.robot_default_dof_pos).unsqueeze(dim=1).repeat(1, self.init_traj_length, 1) 
                    while not torch.all(self.plan_exhausted):                    
                        # Compute end-effector trajectory
                        self.ee_traj_pos = torch.zeros((self.num_envs, self.trajectory_length, 3), dtype=torch.float32, device=self.device, requires_grad=False)
                        # NOTE: Slightly offsetting along branch vector and force vector to avoid losing contact with branch
                        force_vector_length = torch.linalg.norm(trajectory_vector, dim=1, keepdim=True)
                        unit_force_vector = trajectory_vector/force_vector_length
                        self.ee_traj_pos[:, 0] = contact_node_positions - 0.01*branch_vectors_norm - 0.05*unit_force_vector
                        for i in range(1, self.trajectory_length):
                            self.ee_traj_pos[:, i] = self.ee_traj_pos[:, 0] + i*((0.05+force_vector_length)*unit_force_vector/self.trajectory_length)
                        # Orient the end-effector such that z axis aligns with force vector
                        self.ee_init_quat = math_utils.get_quat_from_vec(trajectory_vector, branch_vectors, gripper_axis='x')
                        self.ee_traj_quat = math_utils.get_quat_from_vec(trajectory_vector, branch_vectors, gripper_axis='z')
                        
                        # Compute joint angles for end-effector trajectory using IKFast
                        # TODO: Apply transform to ee_traj_pos and ee_traj_quat to get end-effector trajectory in robot base frame 
                        # Robot pose is defined in world frame T_world_robot
                        # Trajectory is defined in world frame T_world_ee
                        # T_robot_ee = T_world_robot^-1 * T_world_ee            
                        T_world_robot = math_utils.get_transform_from_pose(self.robot_default_pos, self.robot_default_quat)
                        T_world_ee = math_utils.get_transform_from_pose(self.ee_traj_pos, 
                                                                        self.ee_init_quat.unsqueeze(dim=1).repeat(1, self.trajectory_length, 1))
                        T_robot_world = math_utils.transform_inverse(T_world_robot)
                        T_robot_ee = torch.matmul(T_robot_world.unsqueeze(1), T_world_ee).detach().cpu().numpy()
                        # Compute joint angles for end-effector trajectory using ee_traj_pos and ee_traj_quat
                        # <======================= MOTION PLANNING ======================= 
                        robot_base_euler = math_utils.quaternion_to_euler(self.robot_default_quat)
                        for i in range(self.num_envs):
                            if self.plan_exhausted[i]:
                                continue
                            q_guess = self.robot_default_dof_pos[i].detach().cpu().numpy()
                            self.planner.space.select_obstacle(i) # TODO: Handle cases when there are more than 128 envs
                            self.planner.space.robot.set_base_pose(pos=self.robot_default_pos[i].detach().cpu().numpy(), 
                                                                rot=robot_base_euler[i].detach().cpu().numpy())
                            start = self.robot_default_dof_pos[i].detach().cpu().numpy()
                            goals = self.robot_kinematics.get_ik(T_robot_ee[i, 0][:3, 3], T_robot_ee[i, 0][:3, :3], q_guess)
                            for goal in goals:
                                if self.planner.init_plan(start, goal):
                                    if self.planner.plan(steps=10):
                                        trajectory = self.planner.get_trajectory(dt=0.0166, n=self.init_traj_length)
                                        self.ee_traj_joint_angles[i] = torch.from_numpy(trajectory).to(self.device)
                                        self.found_motion_plan[i] = 1
                                        self.plan_exhausted[i] = 1
                                        break
                            self.plan_exhausted[i] = 1
                    print(f'Found motion plan: {torch.sum(self.found_motion_plan)}/{self.num_envs}, Plan exhausted: {torch.sum(torch.abs(self.plan_exhausted))}/{self.num_envs}')
                    if self.found_motion_plan[0]:
                        print(f'Node with rank {rank} has a motion plan!')
                        break
                        # ======================= MOTION PLANNING =======================>

            elif frame > self.begin_action_frame and frame < self.begin_action_frame + self.init_traj_length:
                #self.pos_action[:, -1] = (trajectory_distance+0.05)*(frame - self.begin_action_frame)/self.trajectory_length
                self.pos_action[:, -6:] = self.ee_traj_joint_angles[:, frame-self.begin_action_frame].unsqueeze(-1)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
            elif (frame >= self.begin_action_frame + self.init_traj_length + self.buffer_length) and (frame < self.begin_action_frame + self.init_traj_length + self.buffer_length +self.trajectory_length):
                i = frame - self.begin_action_frame - self.init_traj_length - self.buffer_length
                desired_pose = torch.cat((self.ee_traj_pos[:, i], self.ee_traj_quat), dim=1)
                current_pose = self.rigid_body_state[:, self.ee_rb_handle][:, :7].clone()   
                delta_pose = robot_utils.compute_error(desired_pose, current_pose)
                u_delta = robot_utils.control_ik(self.j_eef, delta_pose, self.num_envs)
                u_delta = u_delta*self.found_motion_plan.unsqueeze(dim=1).repeat(1, 6) # Mask out the env with no motion plan
                self.pos_action[:, -6:] = self.dof_state[:, 0].view(self.num_envs, -1)[:, -6:].unsqueeze(-1) + u_delta.unsqueeze(-1) 
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
            '''
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

                # Visualize predicted contact node position
                contact_node_pos = self.rigid_body_state[i, self.asset_rb_handles_dict_list[i]['initial'][contact_nodes[i].item()], :3]
                rb_force_scaled = (trajectory_vector[i]*visual_scaling_factor).flatten().detach().cpu().numpy()
                geom_pose.p = gymapi.Vec3(contact_node_pos[0], contact_node_pos[1], contact_node_pos[2])
                gymutil.draw_lines(contact_node_geom, self.gym, self.viewer, self.envs[i], geom_pose)                    
                # Visualize predicted contact force
                line_vertices = np.stack((contact_node_pos.detach().cpu().numpy(), contact_node_pos.detach().cpu().numpy() + rb_force_scaled), axis=0)
                line_color = [1,0,0] 
                num_lines = 1
                self.gym.add_lines(self.viewer, self.envs[i], num_lines, line_vertices, line_color)
            '''    
                
            
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
        self.gym.refresh_jacobian_tensors(self.sim)

    def destroy(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        #print("Safely destroyed viewer and sim!")

if __name__ == "__main__":
    args = gymutil.parse_arguments(
        description="Visualize URDF",
        custom_parameters=[
            {"name": "--temp_file", "type": str, "help": "Path to temp file"},
            {"name": "--record_video", "type": bool, "default": False, "help": "Whether to save video"}
        ])

    # Deserialize the data from temp file using pickle
    with open(args.temp_file, 'rb') as temp_file:
        serialized_data = temp_file.read()
        data = pickle.loads(serialized_data)
    visualizer = URDFVisualizer(args, *data)

