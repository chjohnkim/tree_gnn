import torch

def get_quat_from_vec(vec, branch_vec, gripper_axis):
    '''
    Returns quaternion that rotates z axis to vec and x axis to a vector orthogonal to vec and branch_vec
    '''
    if gripper_axis == 'x':
        # Compute orthonormal basis
        x_axis = vec/torch.norm(vec, dim=-1, keepdim=True)
        branch_vec = branch_vec/torch.norm(branch_vec, dim=1, keepdim=True)
        z_axis = torch.cross(x_axis, branch_vec, dim=-1)
        z_axis = -z_axis/torch.norm(z_axis, dim=-1, keepdim=True) # NOTE: Switched the sign of this for motion planning version to work
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis/torch.norm(y_axis, dim=-1, keepdim=True)
    elif gripper_axis == 'z':
        z_axis = vec/torch.norm(vec, dim=-1, keepdim=True)
        branch_vec = branch_vec/torch.norm(branch_vec, dim=1, keepdim=True)
        y_axis = torch.cross(z_axis, branch_vec, dim=-1)
        y_axis = y_axis/torch.norm(y_axis, dim=-1, keepdim=True)
        x_axis = torch.cross(y_axis, z_axis, dim=-1)
        x_axis = x_axis/torch.norm(x_axis, dim=-1, keepdim=True)
    # Given three orthonormal basis, the quaternion is given by
    # q = [w, x, y, z] = [sqrt(1+trace(R))/2, (R21-R12)/(4w), (R02-R20)/(4w), (R10-R01)/(4w)]
    # where R is the rotation matrix
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    trace = x_axis[:,0] + y_axis[:,1] + z_axis[:,2]
    w = torch.sqrt(1+trace)/2
    x = (y_axis[:,2] - z_axis[:,1])/(4*w)
    y = (z_axis[:,0] - x_axis[:,2])/(4*w)
    z = (x_axis[:,1] - y_axis[:,0])/(4*w)
    quat = torch.stack((x,y,z, w), dim=-1)
    return quat

def quaternion_to_euler(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to euler angles in radians.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).
    Returns:
        Euler angles as tensor of shape (..., 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    x = torch.atan2(2 * (i * r + j * k), 1 - 2 * (i * i + j * j))
    y = torch.asin(2 * (j * r - k * i))
    z = torch.atan2(2 * (k * r + i * j), 1 - 2 * (j * j + k * k))
    return torch.stack((x, y, z), -1)
    
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def get_transform_from_pose(pos, quat):
    """ 
    Convert pose to transformation matrix
    Args:
        pos: position of shape (..., 3)
        quat: quaternion of shape (..., 4) with real part last
    Returns:
        transform: transformation matrix of shape (..., 4, 4)
    """
    r_matrix = quaternion_to_matrix(quat)
    transform = torch.cat((r_matrix, pos.unsqueeze(-1)), dim=-1)
    # Add last row to the last 2 dimensions
    transform = torch.cat((transform, torch.zeros_like(transform[...,:1,:])), dim=-2)
    transform[...,3,3] = 1
    return transform

def transform_inverse(transform):
    """
    Compute inverse of transformation matrix
    Args:
        transform: transformation matrix of shape (..., 4, 4)
    Returns:
        transform_inv: inverse of transformation matrix of shape (..., 4, 4)
    """        
    # Inverse of transformation matrix is given by
    # [R^T, -R^T p]
    # [0, 1]
    # where R is the rotation matrix and p is the position vector
    r_matrix = transform[...,:3,:3]
    pos = transform[...,:3,3:]
    r_matrix_inv = r_matrix.transpose(-1,-2)
    pos_inv = torch.matmul(-r_matrix_inv, pos)
    transform_inv = torch.cat((r_matrix_inv, pos_inv), dim=-1)
    # Add last row to the last 2 dimensions
    transform_inv = torch.cat((transform_inv, torch.zeros_like(transform_inv[...,:1,:])), dim=-2)
    transform_inv[...,3,3] = 1
    return transform_inv

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

def matrix_from_axis_angle(axis, angle):
    """
    Convert axis-angle representation to rotation matrix
    Args:
        axis: axis of rotation of shape (..., 3)
        angle: angle of rotation of shape (...)
    Returns:
        r_matrix: rotation matrix of shape (..., 3, 3)
    """
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    axis = axis/torch.norm(axis, dim=-1, keepdim=True)
    x, y, z = axis[...,0], axis[...,1], axis[...,2]
    c = torch.cos(angle)
    s = torch.sin(angle)
    r_matrix = torch.stack((
        c + x**2*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s,
        y*x*(1-c) + z*s, c + y**2*(1-c), y*z*(1-c) - x*s,
        z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z**2*(1-c)
    ), dim=-1).view(axis.shape[:-1]+(3,3))
    return r_matrix

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [axis_angle * sin_half_angles_over_angles, torch.cos(half_angles)], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))