from typing import Tuple, List, Optional

import numpy as np


class ObjectInstance:

    def __init__(self,
                 instance_id: str,
                 object_type: str,
                 centroid_3d: np.ndarray,
                 bounding_box_slices_3d,
                 voxel_count: int,
                 state=None):

        self.instance_id = instance_id
        self.object_type = object_type
        self.centroid_3d = centroid_3d
        self.bounding_box_slices_3d = bounding_box_slices_3d
        self.voxel_count = voxel_count
        self.state = state

        # these flags are used to decide whether the instance should be updated in the compare_and_match_object_instances() process
        # these states indicate the object instance is "gone from eyesight but still exist"
        self.picked_up = False
        self.sliced = False
        self.hidden_in_container = False

    def set_picked_up(self, picked_up):
        self.picked_up = picked_up

    def set_sliced(self, sliced):
        self.sliced = sliced

    def set_hidden_in_container(self, hidden_in_container):
        self.hidden_in_container = hidden_in_container

    @classmethod
    def create_from_cc3d_stats(cls,
                               instance_id: str,
                               object_type: str,
                               centroid_3d: np.ndarray,
                               bounding_box_slices_3d: Tuple[slice, slice, slice], # e.g.: (slice(17, 26, None), slice(27, 39, None), slice(0, 7, None))
                               voxel_count: int):

        return cls(instance_id, object_type, centroid_3d, bounding_box_slices_3d, voxel_count)


class ObjectInstanceDetection2D:
    def __init__(self,
                 object_type: str,
                 bbox_2d: List[int],
                 conf_score: float = 1.0,
                 instance_mask: Optional[np.ndarray] = None):
        """
        Object instance detected from the object detector
        
        :param object_type: string of object class name
        :param bbox_2d: bbox coordincate of [xmin, ymin, xmax, ymax]
        :param instance_mask: bool tensor of shape (h, w)
        :param conf_score: confidence score of the instance detection
        """
        self.object_type = object_type
        self.bbox_2d = bbox_2d
        self.instance_mask = instance_mask
        self.conf_score = conf_score
        self.tmp_unique_id = self.get_tmp_unique_id()

    def get_tmp_unique_id(self):
        """
        Generate a unique id to identify the instance in current 2D detections.
        """
        return self.object_type + "_" +"_".join([str(i) for i in self.bbox_2d])
    
    def get_interaction_point(self):
        """
        Get the interaction point of this object instance.
        """
        centroid = (
            (self.bbox_2d[0] + self.bbox_2d[2]) // 2, 
            (self.bbox_2d[1] + self.bbox_2d[3]) // 2
        )
        if not self.instance_mask:
            return centroid
        
        # TODO: check how to use the coordinate, img[x,y] or img[y,x]? 

        is_valid = ObjectInstanceDetection2D.check_interaction_point_validity(
            centroid, self.instance_mask, 
        )
        if is_valid:
            return centroid
        
        proposals = [
            (
                (self.bbox_2d[0] + self.bbox_2d[2]) // 4, 
                (self.bbox_2d[1] + self.bbox_2d[3]) // 4
            ), 
            (
                (self.bbox_2d[0] + self.bbox_2d[2]) * 3 // 4, 
                (self.bbox_2d[1] + self.bbox_2d[3]) // 4
            ), 
            (
                (self.bbox_2d[0] + self.bbox_2d[2]) // 4, 
                (self.bbox_2d[1] + self.bbox_2d[3]) * 3 // 4
            ), 
            (
                (self.bbox_2d[0] + self.bbox_2d[2]) * 3 // 4, 
                (self.bbox_2d[1] + self.bbox_2d[3]) * 3 // 4
            ), 
        ]
        for p in proposals:
            if ObjectInstanceDetection2D.check_interaction_point_validity(
                p, self.instance_mask, 
            ):
                return p

        # give up
        return centroid
    
    @classmethod
    def check_interaction_point_validity(cls, point: Tuple[int, int], instance_mask: np.ndarray):
        """
        Check if the interaction point of the object instance is valid.
        """
        H, W = instance_mask.shape

        x, y = point
        if x == 0 or x == W-1 or y == 0 or y == H-1:
            return False
        
        if (
            instance_mask[y, x] 
            and instance_mask[y, x+1]
            and instance_mask[y, x-1]
            and instance_mask[y+1, x]
            and instance_mask[y+1, x+1]
            and instance_mask[y+1, x-1]
            and instance_mask[y-1, x]
            and instance_mask[y-1, x-1]
            and instance_mask[y-1, x+1]
        ):
            return True
        return False
