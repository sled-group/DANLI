"""
Creat mapping between object class index to a unique color unified across scenes.
Adopted from: `segmentation_definitions.py` at https://github.com/valtsblukis/hlsm
"""
import sys
import random
import hashlib
import torch
from typing import Tuple
from definitions.teach_objects import ObjectClass


# Mappings between integers and colors
def _compute_object_intid_to_color_o(object_intid: int) -> Tuple[int, int, int]:
    # Backup and restore random number generator state so as not to mess with
    # the rest of the project (e.g. if a random seed is fixed, we want to keep
    # it fixed. If it isn't we don't want to fix it.)
    randstate = random.getstate()
    random.seed(object_intid)
    color = (random.randint(50, 240), random.randint(50, 240), random.randint(50, 240))
    random.setstate(randstate)
    return color


def _compute_object_intid_to_color(object_intid: int) -> Tuple[int, int, int]:
    # Backup and restore random number generator state so as not to mess with
    # the rest of the project (e.g. if a random seed is fixed, we want to keep
    # it fixed. If it isn't we don't want to fix it.)
    int_hash = int.from_bytes(
        hashlib.md5(str(object_intid).encode()).digest(), byteorder=sys.byteorder
    )
    r = 20 + int_hash % 200
    g = 20 + int(int_hash / 200) % 200
    b = 20 + int(int_hash / (200 * 2)) % 200
    color = (r, g, b)
    return color


# Precompute color for each object
OBJECT_INTID_TO_COLOR = {
    i: _compute_object_intid_to_color(i) for i in range(ObjectClass.get_num_objects())
}
COLOR_OTHERS = (255, 255, 255)  # white
OBJECT_INTID_TO_COLOR[ObjectClass.OTHERS.value] = COLOR_OTHERS
OBJECT_COLOR_TO_INTID = {c: i for i, c in OBJECT_INTID_TO_COLOR.items()}


def object_color_to_intid(color: Tuple[int, int, int]) -> int:
    global OBJECT_COLOR_TO_INTID
    return OBJECT_COLOR_TO_INTID[color]


def object_intid_to_color(intid: int) -> Tuple[int, int, int]:
    global OBJECT_INTID_TO_COLOR
    return OBJECT_INTID_TO_COLOR[intid]


def object_string_to_color(object_str: str) -> Tuple[int, int, int]:
    return object_intid_to_color(ObjectClass.name_to_id((object_str)))


def object_color_to_string(color: Tuple[int, int, int]) -> str:
    return ObjectClass.id_to_name(object_color_to_intid(color))


# Image-related definitions
def get_class_color_vector() -> torch.Tensor:
    colors = [object_intid_to_color(i) for i in range(ObjectClass.get_num_objects())]
    return torch.tensor(colors)


# TODO: do we need this?
# Convert tensor to RGB by averaging colors of all objects at each position.
# def intid_tensor_to_rgb(data: torch.Tensor) -> torch.Tensor:
#     num_obj = get_num_objects()
#     vlen = data.shape[1]
#     assert (
#         vlen == get_num_objects()
#     ), f"Object 1-hot tensor length ({vlen}) mismatch object number {num_obj}"

#     data = data.float()

#     # All dimensions after batch and channel dimension are assumed to be spatial.
#     num_spatial_dims = len(data.shape) - 2

#     rgb_tensor = torch.zeros_like(data[:, :3])
#     count_tensor = torch.zeros_like(data[:, :1])
#     for c in range(num_obj):
#         channel_slice = data[:, c : c + 1]
#         channel_count_slice = channel_slice > 0.01
#         rgb_color = object_intid_to_color(c)
#         rgb_color = torch.tensor(rgb_color, device=data.device).unsqueeze(0)
#         # Add correct number of spatial dimensions
#         for _ in range(num_spatial_dims):
#             rgb_color = rgb_color.unsqueeze(2)

#         rgb_slice = channel_slice * rgb_color
#         count_tensor += channel_count_slice
#         rgb_tensor += rgb_slice

#     rgb_avg_tensor = rgb_tensor / (count_tensor + 1e-10)
#     return rgb_avg_tensor / 255
