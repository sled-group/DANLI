"""
Depth to 3D adopted from kornia: https://github.com/kornia/kornia/blob/master/kornia/geometry/depth.py 
"""
from typing import Optional

import torch
import torch.nn.functional as F


def depth_to_3d(
    depth: torch.Tensor, camera_matrix: torch.Tensor, normalize_points: bool = False
) -> torch.Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.
    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
          represented as the Euclidean ray length from the camera position.
    Return:
        tensor with a 3d point per pixel of the same resolution as the input :math:`(B, 3, H, W)`.
    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_3d(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depht type is not a torch.Tensor. Got {type(depth)}.")

    if not (len(depth.shape) == 4 and depth.shape[-3] == 1):
        raise ValueError(
            f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}"
        )

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(
            f"Input camera_matrix type is not a torch.Tensor. "
            f"Got {type(camera_matrix)}."
        )

    if not (len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3)):
        raise ValueError(
            f"Input camera_matrix must have a shape (B, 3, 3). "
            f"Got: {camera_matrix.shape}."
        )

    # create base coordinates grid
    _, _, height, width = depth.shape
    points_2d: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates=False
    )  # 1xHxWx2
    points_2d = points_2d.to(depth.device).to(depth.dtype)

    # depth should come in Bx1xHxW
    points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d: torch.Tensor = unproject_points(
        points_2d, points_depth, camera_matrix_tmp, normalize=normalize_points
    )  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.
    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.
    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.
    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.
    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])
        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def unproject_points(
    point_2d: torch.Tensor,
    depth: torch.Tensor,
    camera_matrix: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    r"""Unproject a 2d point in 3d.
    Transform coordinates in the pixel frame to the camera frame.
    Args:
        point2d: tensor containing the 2d to be projected to
            world coordinates. The shape of the tensor can be :math:`(*, 2)`.
        depth: tensor containing the depth value of each 2d
            points. The tensor shape must be equal to point2d :math:`(*, 1)`.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.
        normalize: whether to normalize the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position.
    Returns:
        tensor of (x, y, z) world coordinates with shape :math:`(*, 3)`.
    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 2)
        >>> depth = torch.ones(1, 1)
        >>> K = torch.eye(3)[None]
        >>> unproject_points(x, depth, K)
        tensor([[0.4963, 0.7682, 1.0000]])
    """
    if not isinstance(point_2d, torch.Tensor):
        raise TypeError(
            f"Input point_2d type is not a torch.Tensor. Got {type(point_2d)}"
        )

    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depth type is not a torch.Tensor. Got {type(depth)}")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(
            f"Input camera_matrix type is not a torch.Tensor. Got {type(camera_matrix)}"
        )

    if not (point_2d.device == depth.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")

    if not point_2d.shape[-1] == 2:
        raise ValueError(
            "Input points_2d must be in the shape of (*, 2)."
            " Got {}".format(point_2d.shape)
        )

    if not depth.shape[-1] == 1:
        raise ValueError(
            "Input depth must be in the shape of (*, 1)." " Got {}".format(depth.shape)
        )

    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError("Input camera_matrix must be in the shape of (*, 3, 3).")

    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    u_coord: torch.Tensor = point_2d[..., 0]
    v_coord: torch.Tensor = point_2d[..., 1]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0, 0]
    fy: torch.Tensor = camera_matrix[..., 1, 1]
    cx: torch.Tensor = camera_matrix[..., 0, 2]
    cy: torch.Tensor = camera_matrix[..., 1, 2]

    # projective
    x_coord: torch.Tensor = (u_coord - cx) / fx
    y_coord: torch.Tensor = (v_coord - cy) / fy

    xyz: torch.Tensor = torch.stack([x_coord, y_coord], dim=-1)
    xyz = convert_points_to_homogeneous(xyz)

    if normalize:
        xyz = F.normalize(xyz, dim=-1, p=2.0)

    return xyz * depth


def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.
    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.
    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.
    Examples:
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)
