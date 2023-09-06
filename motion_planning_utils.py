import numpy as np
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.model.trajectory import Trajectory

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene
from pykin.utils.kin_utils import ShellColors as sc
import trimesh
import time 

class URDFAsset:
    def __init__(self, urdf_path, pos, rot, base_link_name):
        self.asset = SingleArm(urdf_path, Transform(pos=pos, rot=rot))
        self.asset.setup_link_name(base_link_name)
        self.c_manager = CollisionManager(is_robot=True)
        self.c_manager.setup_robot_collision(self.asset, geom="collision")
        self.set_dof(np.array([0]*self.asset.arm_dof))

    def set_dof(self, dof_pos):
        fk = self.asset.forward_kin(dof_pos)
        for link, transform in fk.items():
            if link in self.c_manager._objs:
                collision_h_mat = np.dot(
                    transform.h_mat, self.asset.links[link].collision.offset.h_mat
                )
                self.c_manager.set_transform(name=link, h_mat=collision_h_mat)    

    def set_base_pose(self, pos, rot):
        self.asset.offset = Transform(pos=pos, rot=rot)
        self.asset._setup_kinematics()
        #self.asset._setup_init_fk() # TODO: check if this is necessary

    def collision_check(self, obstacle):
        result_internal, _ = self.c_manager.in_collision_internal(return_names=True)
        result_other, name = self.c_manager.in_collision_other(obstacle, return_names=True)
        if result_other:
        #    print(f"{sc.FAIL}Collide!! {sc.ENDC}{list(name)[0][0]} and {list(name)[0][1]}")
            if list(name)[0][0]=='ee_base': # This is specifically for my gripper
                result_other = not result_other
        return result_internal or result_other
    
    def get_joint_limits(self):
        limits = []
        for joint_name in self.asset.joint_limits.keys():
            limits.append(self.asset.joint_limits[joint_name])
        return limits
        
class ObstacleCSpace(CSpace):
    def __init__(self, eps=1e-1):
        CSpace.__init__(self)
        # set collision checking resolution
        self.eps = eps # Tradeoff between roughness of collision checking and speed of planner
        # set properties
        self.properties['euclidean'] = False
        self.properties['geodesic'] = False
        self.properties['metric'] = 'manhattan'

        # set obstacles here
        self.obstacles = []
        self.all_obstacles = []
        self.running_avg_collision_check_time = 0

    def add_robot(self, robot):
        self.robot = robot
        self.bound = self.robot.get_joint_limits()

    def add_obstacle(self, obstacle):
        self.all_obstacles.append(obstacle)
    
    def select_obstacle(self, idx):
        self.obstacles = [self.all_obstacles[idx]]

    def feasible(self, q):
        #bounds test
        if not CSpace.feasible(self,q): return False
        self.robot.set_dof(q)
        for o in self.obstacles:
            if self.robot.collision_check(o.c_manager): return False
        return True

    def visualize_space(self):
        scene = trimesh.Scene()
        scene = apply_robot_to_scene(trimesh_scene=scene, 
                                     robot=self.robot.asset, 
                                     geom=self.robot.c_manager.geom)
        for o in self.obstacles:
            scene = apply_robot_to_scene(trimesh_scene=scene, 
                                         robot=o.asset, 
                                         geom=o.c_manager.geom)
        scene.show()

class MotionPlanner:
    def __init__(self, space, planner_type='rrt*'):
        self.space = space
        if planner_type == 'prm':
            #PRM planner
            MotionPlan.setOptions(type="prm",knn=10,connectionThreshold=0.1)
            self.optimizingPlanner = False
        elif planner_type == 'fmm*':
            #FMM* planner
            MotionPlan.setOptions(type="fmm*")
            self.optimizingPlanner = True
        elif planner_type == 'rrt':
            #RRT planner
            MotionPlan.setOptions(type="rrt",perturbationRadius=0.25,bidirectional=True)
            self.optimizingPlanner = False
        elif planner_type == 'rrt*':
            #RRT* planner
            MotionPlan.setOptions(type="rrt*", connectionThreshold=4)
            self.optimizingPlanner = True
        elif planner_type == 'rrt_random_restart':        
            #random-restart RRT planner
            MotionPlan.setOptions(type="rrt",perturbationRadius=0.25,bidirectional=True,shortcut=True,restart=True,restartTermCond="{foundSolution:1,maxIters:1000}")
            self.optimizingPlanner = True

    def init_plan(self, start, goal):
        t0 = time.time()
        if not self.space.feasible(goal):   
            print(f'{sc.FAIL}{sc.BOLD}Goal configuration is infeasible{sc.ENDC} in {time.time()-t0:.3f} seconds')
            return False
        if not self.space.feasible(start):   
            print(f'{sc.FAIL}{sc.BOLD}Start configuration is infeasible{sc.ENDC} in {time.time()-t0:.3f} seconds')
            return False
        self.start = start
        self.goal = goal
        self.planner = MotionPlan(self.space)
        try:
            self.planner.setEndpoints(start, goal)
        except Exception as e:
            print(f'{sc.FAIL}{sc.BOLD}Failed to set start and goal{sc.ENDC}: {e} in {time.time()-t0:.3f} seconds')
            return False
        self.path = []
        print(f'Start-Goal configuration is feasible in {time.time()-t0:.3f} seconds')
        return True
    
    def plan(self, steps):
        t0 = time.time()
        if self.optimizingPlanner or not self.path:
            self.planner.planMore(steps)
            self.path = self.planner.getPath()
            self.G = self.planner.getRoadmap()
        if self.path:
            print(f'Planning {sc.COLOR_LIGHT_GREEN}{sc.BOLD}successful{sc.ENDC} in {time.time()-t0:.3f} seconds')
            return True
        else:
            print(f'Planning {sc.FAIL}{sc.BOLD}failed{sc.ENDC} in {time.time()-t0:.3f} seconds')
            return False
        
    def get_trajectory(self, dt, n=None):
        if self.path:
            traj = Trajectory(milestones=self.path)
            traj = traj.discretize_state(dt)
            if n is not None:
                # Resample traj.milestones into n points
                sampled_traj = []
                for i in range(n):
                    idx = int(i*len(traj.milestones)/n)
                    sampled_traj.append(traj.milestones[idx])
                return np.array(sampled_traj)                        
            else:
                return np.array(traj.milestones)
        else:
            return None

if __name__=='__main__':
    bound = [(-np.pi, np.pi), 
             (-np.pi, np.pi),
             (-np.pi, np.pi),
             (-np.pi, np.pi),
             (-np.pi, np.pi),
             (-np.pi, np.pi)]
    space = ObstacleCSpace(bound)
    #space.addObstacle()
    start = (0,0,0,0,0,0)
    goal = (1,1,1,1,1,1)
    planner = MotionPlanner(space)
    planner.init_plan(start, goal)
    planner.plan(steps=1000)
    print(planner.path)
    
    planner.init_plan(goal, start)
    planner.plan(steps=1000)
    print(planner.path)
