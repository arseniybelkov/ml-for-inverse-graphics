import torch
from jaxtyping import Float
from torch import Tensor


def _append_dim(x: Tensor) -> Tensor:
    return x.unsqueeze(-1)


def _prepend_dim(x: Tensor) -> Tensor:
    return x.unsqueeze(0)


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""
    return torch.cat([points, torch.tensor([1.0])])


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""
    return torch.cat([points, torch.tensor([0])])


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""
    return torch.matmul(transform, xyz)


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """
    return transform_rigid(xyz, torch.inverse(cam2world))


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """
    return transform_rigid(xyz, cam2world)


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""
    projected = torch.matmul(intrinsics, xyz[..., :-1])
    z = projected[..., -1:]  # -1: to keep dim
    p = projected[..., :-1]
    return p / z
