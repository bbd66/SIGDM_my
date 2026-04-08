import json
import pdb
import numpy as np
from omegaconf import DictConfig
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
[sys.path.append(i) for i in ['.', '..', '../../ubisoft-laforge-ZeroEGGS-main/ZEGGS']]

from anim import bvh, quat, txform
from utils_zeggs import write_bvh
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


from scipy.spatial.transform import Rotation
from beat_data_proc.MyBVH import load_bvh_data


def euler2mat(angles, euler_orders):
    assert angles.ndim == 3 and angles.shape[2] == 3, f"wrong shape: {angles.shape}"
    assert angles.shape[1] == len(euler_orders)

    nJoints = len(euler_orders)
    nFrames = len(angles)
    rot_mats = np.zeros((nFrames, nJoints, 3, 3), dtype=np.float32)

    for j in range(nJoints):
        # {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations
        R = Rotation.from_euler(euler_orders[j].upper(), angles[:, j, :], degrees=True)  # upper for intrinsic rotation
        rot_mats[:, j, :, :] = R.as_matrix()
    return rot_mats


bone_names = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Neck1",
    "Head",
    "HeadEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandRing",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandPinky",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightHandIndex",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumb4",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandRing",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandPinky",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftHandIndex",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumb4",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightForeFoot",
    "RightToeBase",
    "RightToeBaseEnd",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftForeFoot",
    "LeftToeBase",
    "LeftToeBaseEnd",
    ]


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def preprocess_animation(animation_file, fps=120):
    anim_data = bvh.load(animation_file)       #  'rotations' (8116, 75, 3), 'positions', 'offsets' (75, 3), 'parents', 'names' (75,), 'order' 'zyx', 'frametime' 0.016667
    info = load_bvh_data(animation_file)
    euler_orders=info["euler_orders"][: 75]
    nframes = len(anim_data["rotations"])
    if fps != 120 :
        rate = 120 // fps
        anim_data["rotations"] = anim_data["rotations"][0:nframes:rate]
        anim_data["positions"] = anim_data["positions"][0:nframes:rate]
        dt = 1 / fps
        nframes = anim_data["positions"].shape[0]
    else:
        dt = anim_data["frametime"]
    njoints = len(anim_data["parents"])

    beat_xyz_euler_cp=np.array(anim_data["rotations"])
    pred_pose_rotmat=euler2mat(beat_xyz_euler_cp, euler_orders).astype(np.float32)#34,47,3,3
    print("pred_pose_rotmat shape:",pred_pose_rotmat.shape)

    pred_pose_rotmat=pred_pose_rotmat.reshape(-1,75,3,3)
    pred_pose_rotmat=pred_pose_rotmat.reshape(-1,3,3)
    pred_pose_rotmat=torch.tensor(pred_pose_rotmat)
    
    if pred_pose_rotmat.shape[1:] == (3,3):
        hom_mat = torch.tensor([0, 0, 1]).float()
        rot_mat = pred_pose_rotmat.reshape(-1, 3, 3)
        batch_size, device = rot_mat.shape[0], rot_mat.device
        hom_mat = hom_mat.view(1, 3, 1)
        hom_mat = hom_mat.repeat(batch_size, 1, 1).contiguous()
        hom_mat = hom_mat.to(device)
        pred_pose_rotmat = torch.cat([rot_mat, hom_mat], dim=-1)
    pred_pose_quat=rotation_matrix_to_quaternion(pred_pose_rotmat).reshape(-1,75,4).numpy()
    
    lrot=quat.unroll(pred_pose_quat.copy())
    lpos = anim_data["positions"]
    grot, gpos = quat.fk(lrot, lpos, anim_data["parents"])
    # Find root (Projected hips on the ground)
    root_pos = gpos[:, anim_data["names"].index("Spine2")] * np.array([1, 0, 1])
    # Root direction
    root_fwd = quat.mul_vec(grot[:, anim_data["names"].index("Hips")], np.array([[0, 0, 1]]))
    root_fwd[:, 1] = 0
    root_fwd = root_fwd / np.sqrt(np.sum(root_fwd * root_fwd, axis=-1))[..., np.newaxis]
    # Root rotation
    root_rot = quat.normalize(
        quat.between(np.array([[0, 0, 1]]).repeat(len(root_fwd), axis=0), root_fwd)
    )
    # Find look at direction
    gaze_lookat = quat.mul_vec(grot[:, anim_data["names"].index("Head")], np.array([0, 0, 1]))
    gaze_lookat[:, 1] = 0
    gaze_lookat = gaze_lookat / np.sqrt(np.sum(np.square(gaze_lookat), axis=-1))[..., np.newaxis]
    # Find gaze position
    gaze_distance = 100  # Assume other actor is one meter away
    gaze_pos_all = root_pos + gaze_distance * gaze_lookat
    gaze_pos = np.median(gaze_pos_all, axis=0)
    gaze_pos = gaze_pos[np.newaxis].repeat(nframes, axis=0)

    # Compute local gaze dir
    gaze_dir = gaze_pos - root_pos
    # gaze_dir = gaze_dir / np.sqrt(np.sum(np.square(gaze_dir), axis=-1))[..., np.newaxis]
    gaze_dir = quat.mul_vec(quat.inv(root_rot), gaze_dir)

    # Make relative to root
    lrot[:, 0] = quat.mul(quat.inv(root_rot), lrot[:, 0])
    lpos[:, 0] = quat.mul_vec(quat.inv(root_rot), lpos[:, 0] - root_pos)

    # Local velocities
    lvel = np.zeros_like(lpos)
    lvel[1:] = (lpos[1:] - lpos[:-1]) / dt
    lvel[0] = lvel[1] - (lvel[3] - lvel[2])

    lvrt = np.zeros_like(lpos)
    lvrt[1:] = quat.to_helical(quat.abs(quat.mul(lrot[1:], quat.inv(lrot[:-1])))) / dt
    lvrt[0] = lvrt[1] - (lvrt[3] - lvrt[2])

    # Root velocities
    root_vrt = np.zeros_like(root_pos)
    root_vrt[1:] = quat.to_helical(quat.abs(quat.mul(root_rot[1:], quat.inv(root_rot[:-1])))) / dt
    root_vrt[0] = root_vrt[1] - (root_vrt[3] - root_vrt[2])
    root_vrt[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vrt[1:])
    root_vrt[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vrt[0])

    root_vel = np.zeros_like(root_pos)
    root_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt
    root_vel[0] = root_vel[1] - (root_vel[3] - root_vel[2])
    root_vel[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vel[1:])
    root_vel[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vel[0])

    # Compute character space
    crot, cpos, cvrt, cvel = quat.fk_vel(lrot, lpos, lvrt, lvel, anim_data["parents"])

    # Compute 2-axis transforms
    ltxy = np.zeros(dtype=np.float32, shape=[len(lrot), njoints, 2, 3])
    ltxy[..., 0, :] = quat.mul_vec(lrot, np.array([1.0, 0.0, 0.0]))
    ltxy[..., 1, :] = quat.mul_vec(lrot, np.array([0.0, 1.0, 0.0]))

    ctxy = np.zeros(dtype=np.float32, shape=[len(crot), njoints, 2, 3])
    ctxy[..., 0, :] = quat.mul_vec(crot, np.array([1.0, 0.0, 0.0]))
    ctxy[..., 1, :] = quat.mul_vec(crot, np.array([0.0, 1.0, 0.0]))


    lpos = lpos.reshape(nframes, -1)
    ltxy = ltxy.reshape(nframes, -1)
    lvel = lvel.reshape(nframes, -1)
    lvrt = lvrt.reshape(nframes, -1)

    all_poses = np.concatenate((root_pos, root_rot, root_vel, root_vrt, lpos, ltxy, lvel, lvrt, gaze_dir), axis=1)

    return all_poses, anim_data["parents"], dt, anim_data["order"], njoints


def pose2bvh(poses, outpath, length, smoothing=False, smooth_foot=False):
    parents = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 4, 9, 10, 11, 12, 13, 14, 15,
    12, 17, 18, 19, 20, 17, 22, 23, 24, 25, 12, 27, 28, 29, 30, 27, 32,
    33, 34, 4, 36, 37, 38, 39, 40, 41, 42, 39, 44, 45, 46, 47, 44, 49,
    50, 51, 52, 39, 54, 55, 56, 57, 54, 59, 60, 61, 0, 63, 64, 65, 66,
    67, 0, 69, 70, 71, 72, 73], dtype=np.int32)
    order = 'zyx'
    dt = 0.05
    njoints = 75

    # smoothing
    print("smoothing:",smoothing)
    if smoothing:
        n_poses = poses.shape[0]
        out_poses = np.zeros((n_poses, poses.shape[1]))
        for i in range(out_poses.shape[1]):
            # if (13 + (njoints - 14) * 9) <= i < (13 + njoints * 9): out_poses[:, i] = savgol_filter(poses[:, i], 41, 2)  # NOTE: smoothing on rotation matrices is not optimal
            # else:
            out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal
    else:
        out_poses = poses

    

    # Extract predictions
    P_root_pos = out_poses[:, 0:3]
    P_root_rot = out_poses[:, 3:7]
    P_root_vel = out_poses[:, 7:10]
    P_root_vrt = out_poses[:, 10:13]
    P_lpos = out_poses[:, 13 + njoints * 0: 13 + njoints * 3].reshape([length, njoints, 3])
    P_ltxy = out_poses[:, 13 + njoints * 3: 13 + njoints * 9].reshape([length, njoints, 2, 3])
    P_lvel = out_poses[:, 13 + njoints * 9: 13 + njoints * 12].reshape([length, njoints, 3])
    P_lvrt = out_poses[:, 13 + njoints * 12: 13 + njoints * 15].reshape([length, njoints, 3])

    P_ltxy = torch.as_tensor(P_ltxy, dtype=torch.float32)
    P_lrot = quat.from_xform(txform.xform_orthogonalize_from_xy(P_ltxy).cpu().numpy())        #


    if smoothing:
        #pdb.set_trace()
        next_poses_LeftToeBase = P_lrot[:, -7]      # (length, 4)       7/14, 5/12
        next_poses_RightToeBase = P_lrot[:, -14]
        next_poses_LeftToeBase = np.zeros_like(next_poses_LeftToeBase)
        next_poses_RightToeBase = np.zeros_like(next_poses_RightToeBase)
        P_lrot[:, -7] = next_poses_LeftToeBase
        P_lrot[:, -14] = next_poses_RightToeBase
    # 20fps -> 60fps
    dt = 1 / 15

    write_bvh(outpath,
              P_root_pos,
              P_root_rot,
              P_lpos,
              P_lrot,
              parents, bone_names, order, dt
              )

if __name__ == '__main__':
    animation_file = r"/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/BEAT/trimmed_english_1_emo/val/1_wayne_0_110_110.bvh"
    item=animation_file
    all_poses, parents,dt,order,njoints = preprocess_animation(item, fps=20)
    pose2bvh(poses=all_poses, outpath='111.bvh', length=all_poses.shape[0], smoothing=True, smooth_foot=True)

