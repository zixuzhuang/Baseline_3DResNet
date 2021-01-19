import numpy as np
import torch


def rigid_grid(img):
    # rotate and tranlate batch
    # rotation_sigma = 2*np.pi*#/360*6/4
    rotation = 2 * np.pi * 0.005
    translation = 0.05
    affines = []
    r_s = np.random.uniform(-rotation, rotation, img.shape[0])
    t_s = np.random.uniform(-translation, translation, img.shape[0])
    for r, t in zip(r_s, t_s):
        # convert origin to center
        # T1 = np.array([[1,0,1],[0,1,1],[0,0,1]])
        # rotation
        R = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        # translation
        T = np.array([[1, 0, t], [0, 1, t], [0, 0, 1]])
        # convert origin back to corner
        # T2 = np.array([[1,0,-1],[0,1,-1],[0,0,1]])
        # M = T2@T@R@T1
        M = T @ R  # the center is already (0,0), no need to T1, T2
        affines.append(M[:-1])
    M = np.stack(affines, 0)
    M = torch.tensor(M, dtype=img.dtype, device=img.device)
    grid = torch.nn.functional.affine_grid(M, size=img.shape, align_corners=False)
    return grid


def bspline_grid(img):
    # rotate and tranlate batch
    scale = 50
    grid = (
        (torch.rand(img.shape[0], 2, 9, 9, device=img.device, dtype=img.dtype) - 0.5)
        * 2
        / scale
    )
    grid = torch.nn.functional.interpolate(
        grid, size=img.shape[2:], align_corners=False, mode="bicubic"
    )
    grid = grid.permute(0, 2, 3, 1).contiguous()
    return grid


def augment(img, rigid=True, bspline=True):
    assert rigid == True
    grid = rigid_grid(img.abs())
    if bspline:
        grid = grid + bspline_grid(img.abs())
    sample = lambda x: torch.nn.functional.grid_sample(
        x, grid, padding_mode="reflection", align_corners=False, mode="bilinear"
    )
    if torch.is_complex(img):
        img = sample(img.real) + sample(img.imag) * 1j
    else:
        img = sample(img)
    return img, grid
