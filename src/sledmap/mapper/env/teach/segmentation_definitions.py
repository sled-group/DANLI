from typing import Tuple

import sys
import torch
import random
import hashlib
from definitions.teach_objects import ObjectClass, AFFORDANCE_TO_OBJECTS, THING_NAMES
from definitions.teach_objects import ithor_oid_to_object_class, normalize_semantic_object_class
from definitions.teach_object_semantic_class import SEMANTIC_CLS_TO_OBJECTS



# Mappings between integers and colors
def _compute_object_intid_to_color_o(object_intid: int) -> Tuple[int, int, int]:
    # Backup and restore random number generator state so as not to mess with the rest of the project
    # (e.g. if a random seed is fixed, we want to keep it fixed. If it isn't we don't want to fix it.)
    randstate = random.getstate()
    random.seed(object_intid)
    color = tuple(random.randint(50, 240) for _ in range(3))
    random.setstate(randstate)
    return color

def _compute_object_intid_to_color(object_intid: int) -> Tuple[int, int, int]:
    # Backup and restore random number generator state so as not to mess with the rest of the project
    # (e.g. if a random seed is fixed, we want to keep it fixed. If it isn't we don't want to fix it.)
    int_hash = int.from_bytes(hashlib.md5(str(object_intid).encode()).digest(), byteorder=sys.byteorder)
    r = 20 + int_hash % 200
    g = 20 + int(int_hash / 200) % 200
    b = 20 + int(int_hash / (200 * 2)) % 200
    color = (r, g, b)
    return color

# Precompute color for each object
UNK_OBJ_INT = 141
UNK_OBJ_STR = "OTHERS"
#COLOR_OTHERS = (255, 0, 0)
COLOR_OTHERS = (100, 100, 100)
OBJECT_INTID_TO_COLOR = {i: _compute_object_intid_to_color(i) for i in range(len(ObjectClass))}
OBJECT_INTID_TO_COLOR[UNK_OBJ_INT] = COLOR_OTHERS
OBJECT_COLOR_TO_INTID = {c: i for i, c in OBJECT_INTID_TO_COLOR.items()}


# -------------------------------------------------------------
# Public API:
# -------------------------------------------------------------
# Simple mappings

def get_all_interactive_objects():
    return THING_NAMES

def get_receptacle_ids():
    return [object_string_to_intid(s) for s in AFFORDANCE_TO_OBJECTS['receptacle']]

def get_pickable_ids():
    return [object_string_to_intid(s) for s in AFFORDANCE_TO_OBJECTS['pickupable']]

def get_togglable_ids():
    return [object_string_to_intid(s) for s in AFFORDANCE_TO_OBJECTS['toggleable']]

def get_openable_ids():
    return [object_string_to_intid(s) for s in AFFORDANCE_TO_OBJECTS['openable']]

def get_ground_ids():
    return [object_string_to_intid(s) for s in SEMANTIC_CLS_TO_OBJECTS['Ground']]

def get_num_objects():
    return ObjectClass.get_num_objects()

def object_color_to_intid(color: Tuple[int, int, int]) -> int:
    global OBJECT_COLOR_TO_INTID
    return OBJECT_COLOR_TO_INTID[color]

def object_intid_to_color(intid: int) -> Tuple[int, int, int]:
    global OBJECT_INTID_TO_COLOR
    return OBJECT_INTID_TO_COLOR[intid]

def object_string_to_intid(object_str) -> int:
    object_str = normalize_semantic_object_class(object_str)
    return ObjectClass.name_to_id(object_str)

def object_intid_to_string(intid: int) -> str:
    return ObjectClass.id_to_name(intid)

def object_string_to_color(object_str : str) -> Tuple[int, int, int]:
    return object_intid_to_color(object_string_to_intid((object_str)))

def object_color_to_string(color: Tuple[int, int, int]) -> str:
    return object_intid_to_string(object_color_to_intid(color))

# Image-related definitions

def get_class_color_vector():
    colors = [object_intid_to_color(i) for i in range(get_num_objects())]
    return torch.tensor(colors)

# Convert tensor to RGB by averaging colors of all objects at each position.
def intid_tensor_to_rgb(data : torch.tensor) -> torch.tensor:
    num_obj = get_num_objects()
    assert data.shape[1] == get_num_objects(), (
        f"Object one-hot tensor got the wrong number of objects ({data.shape[1]}), expected {num_obj}")

    data = data.float()

    # All dimensions after batch and channel dimension are assumed to be spatial.
    num_spatial_dims = len(data.shape) - 2

    rgb_tensor = torch.zeros_like(data[:, :3])
    count_tensor = torch.zeros_like(data[:, :1])
    for c in range(num_obj):
        channel_slice = data[:, c:c+1]
        channel_count_slice = channel_slice > 0.01
        rgb_color = object_intid_to_color(c)

        rgb_color = torch.tensor(rgb_color, device=data.device).unsqueeze(0)
        # Add correct number of spatial dimensions
        for _ in range(num_spatial_dims):
            rgb_color = rgb_color.unsqueeze(2)

        rgb_slice = channel_slice * rgb_color
        count_tensor += channel_count_slice
        rgb_tensor += rgb_slice

    rgb_avg_tensor = rgb_tensor / (count_tensor + 1e-10)
    return rgb_avg_tensor / 255
