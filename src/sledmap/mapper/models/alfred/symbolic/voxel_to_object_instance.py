from collections import defaultdict
from typing import List, DefaultDict, Dict

import cc3d
import torch.nn as nn
import numpy as np

from mapper.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
from sledmap.mapper.env.alfred.segmentation_definitions import object_intid_to_string
from sledmap.mapper.models.alfred.symbolic.object_instance import ObjectInstance
from sledmap.mapper.models.alfred.voxel_grid import VoxelGrid


def euclidean_distance(centroid_1, centroid_2):
    return np.linalg.norm(centroid_1 - centroid_2)

class VoxelToObjectInstance(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_list_of_hashtable_object_instance = []

    def forward(self, state_repr: AlfredSpatialStateRepr) -> List[DefaultDict]:
        """
        apply connected component algorithm on voxel grid and return ObjectInstance
        @param state_repr: state representation
        @return: list of list of ObjectInstance
        """
        voxel = state_repr.data
        b, c, w, l, h = voxel.data.shape # Tensor: (1,124,61,61,10)
        device = voxel.data.device

        voxel_in_numpy = voxel.data.cpu().numpy().astype(np.uint16)

        list_of_hashtable_object_instance = []

        for batch_idx in range(b):
            if batch_idx == len(self.prev_list_of_hashtable_object_instance):
                self.prev_list_of_hashtable_object_instance.append(defaultdict(list))
            hashtable_object_instance = self.prev_list_of_hashtable_object_instance[batch_idx]

            # hashtable_object_instance = self.prev_list_of_hashtable_object_instance[batch_idx]\
            #     if batch_idx < len(self.prev_list_of_hashtable_object_instance)\
            #     else defaultdict(list)

            for obj_class_idx in range(c):
                object_type = object_intid_to_string(obj_class_idx)
                labels_in = voxel_in_numpy[batch_idx, obj_class_idx, :, :, :]

                labels_out = cc3d.connected_components(labels_in, connectivity=26, out_dtype=np.uint16)  # 26-connected

                stats = cc3d.statistics(labels_out)
                num_instances_detected = len(stats['bounding_boxes'])
                # if there is no instance of object_type yet,
                # create the object instances and assign id's
                if len(hashtable_object_instance[object_type]) == 0:
                    for instance_idx in range(1, num_instances_detected):  # skip index 0 cuz it's always the background

                        object_instance = ObjectInstance.create_from_cc3d_stats(
                            instance_id=f'{object_type}_{instance_idx}',
                            object_type=object_type,
                            centroid_3d=stats['centroids'][instance_idx],
                            bounding_box_slices_3d=stats['bounding_boxes'][instance_idx],
                            # e.g.: (slice(17, 26, None), slice(27, 39, None), slice(0, 7, None))
                            voxel_count=int(stats['voxel_counts'][instance_idx]))
                        hashtable_object_instance[object_type].append(object_instance)

                # if there already exist one or more object instances,
                # match the new observation with existing ones and update their centroids
                else:
                    new_list_object_instance = self.compare_and_match_object_instances(object_type, stats, hashtable_object_instance[object_type])

                    # replace the list under object_type with the new list
                    hashtable_object_instance[object_type] = new_list_object_instance

            self.prev_list_of_hashtable_object_instance[batch_idx] = hashtable_object_instance

        return self.prev_list_of_hashtable_object_instance

    @staticmethod
    def compare_and_match_object_instances(object_type, new_stats: Dict, existing_object_instances: List[ObjectInstance]) -> List[ObjectInstance]:

        list_to_return = []

        num_instances_detected = len(new_stats['bounding_boxes'])

        # n^2 + n*log(n) + n^2 best effor matching: match new centroids in a best-effort manner to existing centroids
        # if we run out of existing centroids to match against, then create new centroids

        # {(new_idx, exist_idx): euclidean_distance}
        distance_dict = {}

        matched_new_centroid_idxs = []
        used_existing_centroid_idxs = []

        # skip matching against existing object instances that are "gone from eyesight but still exist"
        for exist_idx, eoi in enumerate(existing_object_instances):
            if eoi.picked_up or eoi.sliced or eoi.hidden_in_container:
                used_existing_centroid_idxs += [exist_idx]

        # calculate distance between all pairs of new and existing instance
        for new_detection_idx in range(1, num_instances_detected):  # skip index 0 cuz it's always the background
            new_centroid = new_stats['centroids'][new_detection_idx]
            for exist_idx, eoi in enumerate(existing_object_instances):
                exist_centroid = eoi.centroid_3d
                distance_dict[(new_detection_idx, exist_idx)] = euclidean_distance(new_centroid, exist_centroid)

        # sort distance in ascending order
        sorted_distance_list = sorted(distance_dict.items(), key=lambda item: item[1])

        for (new_idx, exist_idx), distance in sorted_distance_list:
            if new_idx not in matched_new_centroid_idxs and exist_idx not in used_existing_centroid_idxs:
                # this is a valid match! update info for the existing instance
                existing_object_instances[exist_idx].centroid_3d = new_stats['centroids'][new_idx]
                existing_object_instances[exist_idx].bounding_box_slices_3d = new_stats['bounding_boxes'][new_idx]
                existing_object_instances[exist_idx].voxel_count = new_stats['voxel_counts'][new_idx]

                list_to_return.append(existing_object_instances[exist_idx])
                matched_new_centroid_idxs += [new_idx]
                used_existing_centroid_idxs += [exist_idx]
            elif new_idx not in matched_new_centroid_idxs and exist_idx in used_existing_centroid_idxs:
                continue
            elif new_idx in matched_new_centroid_idxs:
                continue

        # at this point min(len(new_id), len(exist_id)) pairs have been matched.
        # And it is provable that these matches are the closest possible pairs.
        # So now, if there is still any new_id that is unused, then we must create new ObjectInstance for them.
        for new_idx in (set(range(1, num_instances_detected)) - set(matched_new_centroid_idxs)):
            new_instance_id = len(list_to_return) + 1  # instance_id starts from 1 not 0
            object_instance = ObjectInstance.create_from_cc3d_stats(
                instance_id=f'{object_type}_{new_instance_id}',
                object_type=object_type,
                centroid_3d=new_stats['centroids'][new_idx],
                bounding_box_slices_3d=new_stats['bounding_boxes'][new_idx],
                voxel_count=int(new_stats['voxel_counts'][new_idx]))
            list_to_return.append(object_instance)

        return list_to_return




