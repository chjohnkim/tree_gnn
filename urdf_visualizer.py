import os
import sys
import pickle
import numpy as np
from urdf_tree_generator import URDFTreeGenerator
from isaacgym import gymapi, gymutil, gymtorch
import torch

# TODO: There is a memory leak problem with this class
# The problem comes from launching and closing the viewer multiple times
# Related post: https://forums.developer.nvidia.com/t/possible-memory-leak/196747/3
# Fixed it by subprocessing the URDFVisualizer class
class URDFVisualizer:
    def __init__(self, args, graph_initial, graph_final, graph_predicted, 
                 contact_node_gt, contact_force_gt, contact_node=None, contact_force=None, auto_close=np.inf):
        self.args = args
        self.graph_initial = graph_initial
        self.graph_final = graph_final
        self.graph_predicted = graph_predicted
        self.contact_node_gt = contact_node_gt
        self.contact_force_gt = contact_force_gt
        self.contact_node = contact_node
        self.contact_force = contact_force
        
        self.auto_close = auto_close

        self.num_envs = len(self.graph_predicted)
        self.gym = gymapi.acquire_gym()
        self.initialize_sim()

        self.asset_root = '.'

        # Generate URDFs for initial, final, and predicted trees
        self.assets_initial = []
        self.assets_final = []
        self.assets_predicted = []
        for i in range(self.num_envs):
            tree_urdf = URDFTreeGenerator(self.graph_initial[i], f'temp_tree_urdf', trunk_radius=self.args.trunk_radius, asset_path=self.asset_root)
            self.assets_initial.append(self.load_asset(tree_urdf.save_file_name))
            
            tree_urdf = URDFTreeGenerator(self.graph_final[i], f'temp_tree_urdf', trunk_radius=self.args.trunk_radius, asset_path=self.asset_root)
            self.assets_final.append(self.load_asset(tree_urdf.save_file_name))
            
            tree_urdf = URDFTreeGenerator(self.graph_predicted[i], f'temp_tree_urdf', trunk_radius=self.args.trunk_radius, asset_path=self.asset_root)
            self.assets_predicted.append(self.load_asset(tree_urdf.save_file_name))

        self.create_env()
        self.create_viewer()
        self.visualize()

    def initialize_sim(self):
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = dt = 1.0 / 60.0
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        sim_params.substeps = 1
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 12
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True
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
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
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
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        asset_options.flip_visual_attachments = True
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

        # create an array of DOF states that will be used to update the actors
        num_dofs = self.gym.get_asset_dof_count(self.assets_initial[0])

        self.num_rigid_bodies = self.gym.get_asset_rigid_body_count(self.assets_initial[0])*3

        # get array of DOF properties
        self.tree_dof_props_per_asset = []
        for tree_asset in self.assets_initial:
            dof_props = self.gym.get_asset_dof_properties(tree_asset)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS) 
            dof_props['effort'].fill(np.inf)
            dof_props['velocity'].fill(9999999999)
            dof_props['armature'].fill(0.1)
            for i in range(num_dofs):
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
        self.gym.prepare_sim(self.sim)

        # Acquire gym tensor
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
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

    def visualize(self):
        frame = 0
        rb_force_tensor = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), device=self.device, dtype=torch.float)
        # Initialize wireframe sphere geometry for visualizing nodes
        contact_node_geom = gymutil.WireframeSphereGeometry(0.02, 10, 10, gymapi.Transform(), color=(1, 0, 0))
        contact_node_gt_geom = gymutil.WireframeSphereGeometry(0.02, 10, 10, gymapi.Transform(), color=(0, 1, 0))
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                # Visualize GT contact node position
                contact_node_gt_pos = self.rigid_body_state[i, self.asset_rb_handles_dict_list[i]['initial'][self.contact_node_gt[i].item()], :3]
                geom_pose = gymapi.Transform()
                geom_pose.p = gymapi.Vec3(contact_node_gt_pos[0], contact_node_gt_pos[1], contact_node_gt_pos[2])
                gymutil.draw_lines(contact_node_gt_geom, self.gym, self.viewer, self.envs[i], geom_pose)
                # Visualize GT contact force
                rb_force_scaled = (self.contact_force_gt[i]*0.01).flatten().detach().cpu().numpy()
                line_vertices = np.stack((contact_node_gt_pos.detach().cpu().numpy(), contact_node_gt_pos.detach().cpu().numpy() + rb_force_scaled), axis=0)
                line_color = [0,1,0] 
                num_lines = 1
                self.gym.add_lines(self.viewer, self.envs[i], num_lines, line_vertices, line_color)

                if self.contact_node is not None: # If we are visualizing control policy predictions            
                    # Visualize predicted contact node position
                    contact_node = self.contact_node
                    contact_force = self.contact_force
                    contact_node_pos = self.rigid_body_state[i, self.asset_rb_handles_dict_list[i]['initial'][contact_node[i].item()], :3]
                    rb_force_scaled = (contact_force[i]*0.01).flatten().detach().cpu().numpy()
                    geom_pose.p = gymapi.Vec3(contact_node_pos[0], contact_node_pos[1], contact_node_pos[2])
                    gymutil.draw_lines(contact_node_geom, self.gym, self.viewer, self.envs[i], geom_pose)                    
                    # Visualize predicted contact force
                    line_vertices = np.stack((contact_node_pos.detach().cpu().numpy(), contact_node_pos.detach().cpu().numpy() + rb_force_scaled), axis=0)
                    line_color = [1,0,0] 
                    num_lines = 1
                    self.gym.add_lines(self.viewer, self.envs[i], num_lines, line_vertices, line_color)
                    # Simulate predicted contact force deformation    
                    if frame>25:
                        rb_force_tensor[i, self.asset_rb_handles_dict_list[i]['predicted'][contact_node[i].item()], :3] = contact_force[i]

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(rb_force_tensor), None, gymapi.ENV_SPACE)
            self.refresh()
                
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
                
            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
            frame+=1
            if frame>self.auto_close:
                import json
                # Compute the maximum node distance between the predicted and target tree
                # and the maximum node displacement between the initial and target tree
                max_node_displacements = []
                max_node_dist_errors = []
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
                serialized_data = json.dumps({'max_node_displacement': max_node_displacements, 'max_node_dist_error': max_node_dist_errors})
                print("DATA_START")
                print(serialized_data)
                print("DATA_END")
                sys.stdout.flush()
                break
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
        ])

    # Deserialize the data from temp file using pickle
    with open(args.temp_file, 'rb') as temp_file:
        serialized_data = temp_file.read()
        data = pickle.loads(serialized_data)
    visualizer = URDFVisualizer(args, *data)

