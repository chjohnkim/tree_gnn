import numpy as np
import torch
from math_utils import quat_conjugate, quat_mul

class RobotKinematics:
    def __init__(self, robot_name):
        if robot_name=='franka':
            from ikfast_franka_panda import get_fk, get_ik, get_dof, get_free_dof
            self.feasible_ranges = {
                                    'robot_joint_1' :  {'lower' : -2.8973, 'upper' : 2.8973}, 
                                    'robot_joint_2' :  {'lower' : -1.7628, 'upper' : 1.7628},
                                    'robot_joint_3' :  {'lower' : -2.8973, 'upper' : 2.8973},
                                    'robot_joint_4' :  {'lower' : -3.0718, 'upper' : -0.0698},
                                    'robot_joint_5' :  {'lower' : -2.8973, 'upper' : 2.8973},
                                    'robot_joint_6' :  {'lower' : -0.0175, 'upper' : 3.7525},
                                    'robot_joint_7' :  {'lower' : -2.8973, 'upper' : 2.8973},
                                    }
            self.free_joints = [6]        
        elif robot_name=='ur5':
            from ikfast_ur5 import get_fk, get_ik, get_dof, get_free_dof
            self.feasible_ranges = {'robot_joint_1' :  {'lower' : -3.14159*2, 'upper' : 3.14159*2}, 
                                    'robot_joint_2' :  {'lower' : -3.14159*2, 'upper' : 3.14159*2},
                                    'robot_joint_3' :  {'lower' : -3.14159, 'upper' : 3.14159},
                                    'robot_joint_4' :  {'lower' : -3.14159*2, 'upper' : 3.14159*2},
                                    'robot_joint_5' :  {'lower' : -3.141  *2, 'upper' : 3.14159*2},
                                    'robot_joint_6' :  {'lower' : -3.14159*2, 'upper' : 3.14159*2},
                                    }
            self.free_joints = [5]
        self.fk = get_fk
        self.ik = get_ik
        self.n_joints = get_dof()
        self.n_free_joints = get_free_dof()
        self.weights = [1]*self.n_joints

    def get_fk(self, q):
        """get the end effector pose given joint angles
        q : list of joint angles in radian
        """
        assert len(q) == self.n_joints
        pos, rot = self.fk(q)
        return pos, rot

    def get_ik(self, pos, rot, q_guess=None):
        """get solutions of joint angles given end effector pose
        pos : end effector position [x, y, z]
        rot : end effector 3x3 rotation matrix 
        q_guess : initial joint angles
        """
        #assert len(pos) == 3
        #assert len(rot) == 9
        #assert len(q_guess) == self.n_joints
        sols = self.ik(pos, rot, self.free_joints)
        if q_guess is None:
            return sols
        else:
            sols = self.best_sol(sols, q_guess)
            if sols is None:
                return np.array([q_guess])
            return sols
        
    def best_sol(self, sols, q_guess):
        """get the best solution based on UR's joint domain value and weighted joint diff
        modified from :
        https://github.com/ros-industrial/universal_robot/blob/kinetic-devel/ur_kinematics/src/ur_kinematics/test_analytical_ik.py

        """
        num_jt = len(q_guess)
        valid_sols = []
        for sol in sols:
            test_sol = np.ones(num_jt)*9999.
            for i, jt_name in enumerate(self.feasible_ranges.keys()):
                for add_ang in [-2.*np.pi, 0, 2.*np.pi]:
                    test_ang = sol[i] + add_ang                    
                    if test_ang <= self.feasible_ranges[jt_name]['upper'] and \
                    test_ang >= self.feasible_ranges[jt_name]['lower'] and \
                    abs(test_ang - q_guess[i]) < abs(test_sol[i] - q_guess[i]):
                        test_sol[i] = test_ang
            if np.all(test_sol != 9999.):
                valid_sols.append(test_sol)
        if len(valid_sols) == 0:
            return None
        # Instead of returning a single best solution, return all valid sols in order of best to worst
        valid_sols = np.array(valid_sols)
        best_sol_inds = np.argsort(np.sum((self.weights*(valid_sols - np.array(q_guess)))**2,1))
        return valid_sols[best_sol_inds]
        #best_sol_ind = np.argmin(np.sum((self.weights*(valid_sols - np.array(q_guess)))**2,1))
        #return valid_sols[best_sol_ind]

def control_ik(j_eef, dpose, num_envs, damping=0.05):
    """Solve damped least squares, from `franka_cube_ik_osc.py` in Isaac Gym.

    Returns: Change in DOF positions, [num_envs,num_dofs], to add to current positions.
    """
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6).to(j_eef_T.device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def compute_error(desired, current):
    """Compute error between desired and current pose.

    Args:
        desired: Desired pose, (num_envs,7) - [x,y,z,qx,qy,qz,qw]
        current: Current pose, (num_envs,7) - [x,y,z,qx,qy,qz,qw]

    Returns: Position and orientation error, (num_envs,3) and (num_envs,3).
    """
    pos_err = desired[:, 0:3] - current[:, 0:3]
    orn_err = orientation_error(desired[:, 3:7], current[:, 3:7])
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)  # (num_envs,6,1) 
    return dpose


if __name__=="__main__":
    import time
    robot = RobotKinematics('ur5') 
    for i in range(100):
        start = time.time()
        pos, rot = robot.get_fk([0]*6)
        print(f'fk: {time.time()-start}')
        start = time.time()
        joint_angles = robot.get_ik(pos, rot, [0]*6)
        print(f'ik: {time.time()-start}')
        #print(joint_angles)