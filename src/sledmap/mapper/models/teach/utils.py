from typing import List, Tuple, Union
import torch
import numpy as np

from sledmap.mapper.models.alfred.voxel_grid import DefaultGridParameters as DGP
from sledmap.mapper.models.alfred.voxel_grid import ROUNDING_OFFSET as ROF

GRID_ORIGIN = np.array(DGP.GRID_ORIGIN)

def euclidean_distance(centroid_1, centroid_2):
    return np.linalg.norm(centroid_1 - centroid_2)

def voxel_count(voxel_mask: torch.Tensor) -> int:
    return voxel_mask.sum().item()


def get_centroid_3d_from_voxel_mask(voxel_mask: torch.Tensor) -> np.ndarray:
    """
    :param voxel_mask: tensor of shape (61, 61, 10)
    :return: ndarray of the 3D centroid
    """
    centroid_index_coord = voxel_mask.nonzero().float().mean(dim=0).cpu().numpy()
    centroid_meter_coord = voxel_idx_coord_to_meter_coord(centroid_index_coord)
    return centroid_meter_coord


def voxel_idx_coord_to_meter_coord(voxel_idx_coord: np.ndarray) -> np.ndarray:
    """
    Convert a voxel grid coordinate to meter coordinate.
    @param voxel_idx_coord: a tensor of shape (3,)
    @return: a tensor of shape (3,)
    """
    voxel_meter_coord = voxel_idx_coord*DGP.GRID_RES + GRID_ORIGIN + DGP.GRID_RES * (0.5 - ROF)
    return voxel_meter_coord


def euclidean_distance(centroid_1, centroid_2):
    return np.linalg.norm(centroid_1 - centroid_2)


def compare_and_match_centroids(new_centroid_list: List[np.ndarray],
                                existing_centroid_list: List[np.ndarray],
                                distance_func=euclidean_distance) -> List[Tuple]:
    """

    @param new_centroid_list: a list of 3d centroids
    @param existing_centroid_list: a list of 3d centroids
    @param distance_func: the distance metric used to match centroids
    @return: a list of  matching pairs of idxs in the lists passed in.
            'None' means no match, for example:
            List<(new_idx, exist_idx),(new_idx, exist_idx)>
            List<(new_idx, exist_idx),(None, exist_idx)>
            List<(new_idx, exist_idx),(new_idx, None)>
    """

    idx_pair_list = []

    matched_new_centroid_idxs = []
    matched_existing_centroid_idxs = []

    # {(new_idx, exist_idx): euclidean_distance}
    distance_dict = {}

    # calculate distance between all pairs of new and existing centroids
    for new_idx, new_centroid in enumerate(new_centroid_list):
        for exist_idx, exist_centroid in enumerate(existing_centroid_list):
            distance_dict[(new_idx, exist_idx)] = distance_func(new_centroid, exist_centroid)

    # sort distance in ascending order
    sorted_distance_list = sorted(distance_dict.items(), key=lambda item: item[1])

    for (new_idx, exist_idx), distance in sorted_distance_list:
        if new_idx not in matched_new_centroid_idxs and exist_idx not in matched_existing_centroid_idxs:
            # this is a valid match! add the pair to idx_pair_list
            idx_pair_list.append((new_idx, exist_idx))
            matched_new_centroid_idxs += [new_idx]
            matched_existing_centroid_idxs += [exist_idx]
        else:
            # one of new or exist has been used before, skip
            continue

    unmatched_new_idxs = set(range(len(new_centroid_list))) - set(matched_new_centroid_idxs)
    idx_pair_list += [(idx, None) for idx in unmatched_new_idxs]

    unmatched_exist_idxs = set(range(len(existing_centroid_list))) - set(matched_existing_centroid_idxs)
    idx_pair_list += [(None, idx) for idx in unmatched_exist_idxs]

    return idx_pair_list


def find_nearest_visible_instance_with_type(
    target_centroid: np.ndarray, object_type: Union[str, List[str]], visible_instances):
    """
    Find the nearest instance with certain types to the target instance in the 
    current observation.

    :param visible_instances: list of ObjectInstanceDetection2D
    :return: None or ObjectInstanceDetection2D
    """
    if isinstance(object_type, str):
        object_type = [object_type]
    nearest_instance = None
    nearest_distance = None
    for instance in visible_instances:
        if instance.object_type in object_type:

            if hasattr(instance, 'centroid_2d'):
                centroid = instance.centroid_2d 
            elif hasattr(instance, 'centroid_3d'):
                centroid = instance.centroid_3d
            else:
                raise NotImplementedError('Instance should have a 2D or 3D centroid')
            distance = euclidean_distance(target_centroid, centroid)
            if nearest_distance is None or distance < nearest_distance:
                nearest_instance = instance
                nearest_distance = distance
    return nearest_instance
