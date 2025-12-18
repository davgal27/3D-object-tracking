import torch
import torchgeometry as tgm
import numpy as np
import cv2

def rotation_to_matrix(rot_angles):
    return tgm.angle_axis_to_rotation_matrix(rot_angles)[:, :3, :3]

def translation_to_skew_symetric(t):
    count = 1
    zero = torch.Tensor(count)
    t = torch.stack(
        [zero, -t[:, 2], t[:, 1],
         t[:, 2], zero, t[:, 0],
        -t[:, 1], t[:, 0], zero], dim=1)
    return t.reshape((count, 3, 3))

def stereo_loss(corners1, corners2, focal, rot_angles, t):
    #focal = focal.reshape(-1, 1)
    corners1 = corners1 / focal[0]
    corners2 = corners2 / focal[1]
    corners1 = tgm.convert_points_to_homogeneous(corners1)
    corners2 = tgm.convert_points_to_homogeneous(corners2)

    t = translation_to_skew_symetric(t)
    R = rotation_to_matrix(rot_angles)
    F = torch.bmm(R, t)
    err = torch.matmul(corners1.reshape(-1, 1, 3), F)
    err = err.matmul(corners2.reshape(-1, 3, 1))
    err = err ** 2
    return torch.sum(err)

pi = 3.14

def get_projection(K, pose):
    pose_m = tgm.rtvec_to_pose(pose)
    P = torch.bmm(K, pose_m[:, :3, :])
    return P

def project_points(P, points_3d):
    proj = torch.matmul(P, points_3d)
    proj = torch.t(proj[0])
    proj = proj[..., 0:-1] / proj[..., -1:]

def get_reprojection_error(K, pose, corners, points_3d):
    P = get_projection(K, pose)
    err = None
    for i in range(2):
        proj = project_points(P[i:i+1], points_3d)
        if err is None:
            err = (proj - corners[i]) ** 2
        else:
            err += (proj - corners[i]) ** 2
    err = torch.sum(err)
    return err

def test():
    distance = 3
    corners_3D = np.random.uniform(-1, 1, size=(100, 3)) + np.asarray([[0, 0, distance]])
    fov = 65
    img_size = [1920, 1080]
    f = img_size[0] / 2 * np.arctan(fov * pi / 180 / 2)
    K = np.array(
        [[f, 0, img_size[0] / 2],
         [0, f, img_size[1] / 2],
         [0, 0, 1],
         ])
    cam2_position =
    


def draw_points(img, points_3d, P):
    proj = project_points(P, points_3d)
    for x, y in proj:
        cv2.cir
    
 


def stereo_camera_Rt(corners1, corners2, fov=65, img_size=[1920, 1080]):
    c = np.asarray([img_size[0] / 2, img_size[1] / 2]).reshape(1, 1, 2) 
    #corners2 = (corners1 - c) / 2 + c
    f = img_size[0] / 2 * np.arctan(fov * pi / 180 / 2)

    K = torch.tensor(np.array(
        [[f, 0, img_size[0] / 2],
         [0, f, img_size[1] / 2],
         [0, 0, 1],
         ])).float().reshape(1, 3, 3)

    K = torch.cat([K, K], dim=0)

    pose = np.zeros([2, 6], np.float32)
    pose[0, 1] = 0.01
    pose[1, 1] = 0.01
    pose[1, 3] = -0.25
    pose = torch.tensor(pose, requires_grad=True)

    corners1 = corners1[:, 0, :]
    corners2 = corners2[:, 0, :]
    corners = np.stack([corners1, corners2], axis=0)
    corners = torch.tensor(corners).float()
    z = 10
    x = (corners1[:, 0] - img_size[0] / 2) / f * z
    y = (corners1[:, 1] - img_size[1] / 2) / f * z
    z = np.ones_like(x) * z
    corners_3d = torch.tensor(np.stack([x, y, z, np.ones_like(z)], axis=1).T.astype(np.float32), requires_grad=True)


    optimizer = torch.optim.Adam([pose], lr=0.0005)
    print(pose.detach().numpy()[0], pose.detach().numpy()[1])
    for i in range(2000):
        optimizer.zero_grad()
        loss = get_reprojection_error(K, pose, corners, corners_3d)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(loss.detach().numpy(), pose.detach().numpy()[0], pose.detach().numpy()[1])


    optimizer = torch.optim.Adam([corners_3d, pose], lr=0.0005)
    print(pose.detach().numpy()[0], pose.detach().numpy()[1])
    for i in range(20000):
        optimizer.zero_grad()
        loss = get_reprojection_error(K, pose, corners, corners_3d)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(loss.detach().numpy(), pose.detach().numpy()[0], pose.detach().numpy()[1])





    print(corners_3d.detach().numpy().T)
    print(loss.detach().numpy(), pose.detach().numpy()[0], pose.detach().numpy()[1])


