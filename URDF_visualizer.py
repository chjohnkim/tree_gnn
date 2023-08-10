import os
import numpy as np
from isaacgym import gymapi, gymutil

from urdf_tree_generator import URDFTreeGenerator

# TODO: There is a memory leak problem with this class
# The problem comes from launching and closing the viewer multiple times
# Related post: https://forums.developer.nvidia.com/t/possible-memory-leak/196747/3

class URDFVisualizer:
    def __init__(self, graph_initial, graph_final, graph_predicted):
        self.args = gymutil.parse_arguments(
                description="Visualize URDF",
                custom_parameters=[
                    {"name": "--trunk_radius", "type": float, "default": 0.02, "help": "Radius of trunk"},
                ])
        
        # initialize gym
        self.gym = gymapi.acquire_gym()
        self.sim = self.initialize_sim()
        
        self.asset_root = '.'
        self.assets = []
        tree_urdf = URDFTreeGenerator(graph_initial, f'temp_tree_urdf', trunk_radius=self.args.trunk_radius, asset_path=self.asset_root)
        self.assets.append(self.load_asset(tree_urdf.save_file_name))
        
        tree_urdf = URDFTreeGenerator(graph_final, f'temp_tree_urdf', trunk_radius=self.args.trunk_radius, asset_path=self.asset_root)
        self.assets.append(self.load_asset(tree_urdf.save_file_name))
        
        tree_urdf = URDFTreeGenerator(graph_predicted, f'temp_tree_urdf', trunk_radius=self.args.trunk_radius, asset_path=self.asset_root)
        self.assets.append(self.load_asset(tree_urdf.save_file_name))

        self.create_viewer()
        self.create_env()
        self.visualize()

    def initialize_sim(self):
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = dt = 1.0 / 60.0
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = self.args.num_threads
        sim_params.physx.use_gpu = self.args.use_gpu
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        sim_params.use_gpu_pipeline = False

        if self.args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()
        return sim
    
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
        #asset_options.flip_visual_attachments = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
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
        num_dofs = self.gym.get_asset_dof_count(self.assets[0])
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

        # get array of DOF properties
        dof_props = self.gym.get_asset_dof_properties(self.assets[0])
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS) 
        dof_props['stiffness'].fill(600.)
        dof_props['damping'].fill(25.)
        dof_props['effort'].fill(np.inf)
        dof_props['velocity'].fill(9999999999)
        dof_props['friction'].fill(0.0)
        dof_props['armature'].fill(0.1)

        # set up the env grid
        num_envs = 1
        num_per_row = 1
        spacing = 2.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        print("Creating %d environments" % num_envs)
        self.envs = []
        for i in range(num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            # add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            pose.r = gymapi.Quat()
            for asset_idx in range(len(self.assets)):
                actor_handle = self.gym.create_actor(env, self.assets[asset_idx], pose, "actor", asset_idx, 1, 0)
                self.gym.set_actor_dof_properties(env, actor_handle, dof_props)
                self.gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

                # Set rigid body colors
                # Initial tree stays same, final tree is green, predicted tree is red
                if asset_idx==1:                
                    color = gymapi.Vec3(0.0, 1.0, 0.0)
                    rb_names = self.gym.get_actor_rigid_body_names(env, actor_handle)
                    for rb_name in rb_names:
                        rb_idx = self.gym.find_actor_rigid_body_index(env, actor_handle, rb_name, gymapi.DOMAIN_ACTOR)
                        self.gym.set_rigid_body_color(env, actor_handle, rb_idx, gymapi.MESH_VISUAL_AND_COLLISION, color)
                elif asset_idx==2:
                    color = gymapi.Vec3(1.0, 0.0, 0.0)
                    rb_names = self.gym.get_actor_rigid_body_names(env, actor_handle)
                    for rb_name in rb_names:
                        rb_idx = self.gym.find_actor_rigid_body_index(env, actor_handle, rb_name, gymapi.DOMAIN_ACTOR)
                        self.gym.set_rigid_body_color(env, actor_handle, rb_idx, gymapi.MESH_VISUAL_AND_COLLISION, color)


    def visualize(self):

        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
                
            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
        self.destroy()

    def destroy(self):
        self.gym.destroy_viewer(self.viewer)
        
        self.gym.destroy_sim(self.sim)
        # destroy environments
        for env in self.envs:
            self.gym.destroy_env(env)
        del self.assets
        del self.envs
        del self.viewer
        del self.sim
        del self.gym

        print("Safely destroyed viewer and sim!")

if __name__ == "__main__":
    URDFVisualizer()