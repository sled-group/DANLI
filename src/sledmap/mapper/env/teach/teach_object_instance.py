from typing import Tuple, List, Optional, Dict, Union
import json
import random
import numpy as np
import torch
import sledmap.mapper.models.teach.utils as utils
from definitions.teach_object_state import Unary, Relation, PRINT_STATE
from definitions.teach_objects import (
    get_object_affordance,
    get_object_receptacle_compatibility,
)


class TeachObjectInstance:
    def __init__(self, object_type: str):
        self.affordances = get_object_affordance(object_type)


class ObjectInstance(TeachObjectInstance):
    def __init__(
        self,
        instance_id: str,
        object_type: str,
        voxel_mask: torch.Tensor,  # shape: (61, 61, 10)
        state: Optional[Dict[str, Union[Unary, Relation]]] = None,
    ):
        super().__init__(object_type)
        self.instance_id = instance_id
        self.object_type = object_type
        self.voxel_mask = voxel_mask
        self.state = {} if state is None else state

        self.voxel_count = utils.voxel_count(voxel_mask)
        self.centroid_3d = utils.get_centroid_3d_from_voxel_mask(voxel_mask)
        self.nearest_vx = None

        self.is_interacted = False

    def to(self, device):
        self.voxel_mask = self.voxel_mask.to(device)
        return self

    def update_voxel_mask(self, new_voxel_mask):
        self.voxel_mask = new_voxel_mask
        self.voxel_count = utils.voxel_count(new_voxel_mask)
        self.centroid_3d = utils.get_centroid_3d_from_voxel_mask(new_voxel_mask)
    
    def update_distance(self, agent_pose_m, mode='centroid'):
        if mode == 'centroid':
            new_distance = utils.euclidean_distance(self.centroid_3d, agent_pose_m)
            self.state["distance"].set_value(new_distance, verbose=False)
        elif mode == 'nearest_voxel':
            near_vx_m = utils.voxel_idx_coord_to_meter_coord(self.nearest_vx)
            new_distance = utils.euclidean_distance(near_vx_m, agent_pose_m)
            self.state["distance"].set_value(new_distance, verbose=False)
        else:
            raise ValueError("Unknown mode: %s" % mode)
    
    def update_nearest_voxel_coord(self, agent_pose_m, voxel_size, voxel_origin):
        points = self.voxel_mask.nonzero()
        points_m = points*voxel_size + voxel_origin + voxel_size * (0.5 - utils.ROF)
        distances = (points_m - agent_pose_m).norm(dim=1)

        NEAREST_VX_PERCENTAGE = 0.2
        sorted_idx = distances.argsort()
        num_voxels = max(1, int(round(NEAREST_VX_PERCENTAGE * len(sorted_idx))))

        selected_points_idx = sorted_idx[:num_voxels]
        self.nearest_vx = points[selected_points_idx].float().mean(dim=0).cpu().numpy()
        # self.nearest_vx = points[distances.argmin()].cpu().numpy()
    

    def mark_interacted(self):
        self.is_interacted = True
    

    def update_state_from_detection(self, instance_2d, instance_mapping_2to3):
        TRUST_DISTANCE = 2.5 # only update when the distance is smaller than TRUST_DISTANCE
        if self.state.distance() >= TRUST_DISTANCE:
            return
        for predicate in instance_2d.state:
            # for object relations, convert 2d instance ids to 3d instance ids
            if predicate in ['receptacleObjectIds', 'parentReceptacles']:
                iids_3d = {}
                for iid_2d, conf in instance_2d.state[predicate].value_with_conf.items():
                    iid_3d = instance_mapping_2to3.get(iid_2d)
                    if iid_3d:
                        iids_3d[iid_3d] = conf
                # print('2d ids to 3d ids:', instance_2d.state[predicate].value_with_conf.keys(), iid_3d)
                # if iids_3d:
                #     print('Detected instance[%r] %s:'%(self, predicate), iids_3d)
                instance_2d.state[predicate].update(iids_3d)

            self.state[predicate].update_from_estimation(instance_2d.state[predicate])
            if predicate == "simbotIsFilledWithWater":
                self.state['isFilledWithLiquid'].set_value(self.state.simbotIsFilledWithWater())
            elif predicate == "simbotIsFilledWithCoffee":
                self.state['isFilledWithLiquid'].set_value(self.state.simbotIsFilledWithCoffee())

    def __str__(self):
        _str = "%s @(%.2f, %.2f, %.2f) Size: %d" % (
            self.instance_id,
            self.centroid_3d[0],
            self.centroid_3d[1],
            self.centroid_3d[2],
            self.voxel_count,
        )
        return _str

    def __repr__(self):
        _repr = self.__str__()

        distance = self.state.get("distance", None)
        if distance is not None:
            distance = distance.get_value()
        if distance is not None:
            _repr += " (D: %.2f" % distance

        for pred in PRINT_STATE:
            state = self.state.get(pred, None)
            if state is not None:
                state_value = state.get_value()
                if pred in "isDirty" and state_value is True:
                    continue
                if state_value or (pred in "isDirty" and not state_value):
                    _repr += " %s" % pred.replace("simbot", "").replace("isDirty", "isClean")
                    if "dirtyable" not in self.affordances:
                        _repr = _repr.replace(" isClean", "")
                    if pred == "isPlacedOn":
                        _repr += ":%s" % state_value
        _repr += ")"
        return _repr
    
    


class ObjectInstanceDetection2D(TeachObjectInstance):

    # How many pixels around the interaction point need to be in the
    # the instance's mask
    INTERACTION_POINT_SIZE = 8
    SAMPLING_TIME = 10

    def __init__(
        self,
        object_type: str,
        bbox_2d: List[int],
        conf_score: float = 1.0,
        instance_mask: Optional[torch.Tensor] = None,
        state: Dict = {},
    ):
        """
        Object instance detected from the object detector

        :param object_type: string of object class name
        :param bbox_2d: bbox coordincate of [xmin, ymin, xmax, ymax]
        :param instance_mask: bool tensor of shape (h, w)
        :param conf_score: confidence score of the instance detection
        """
        super().__init__(object_type)
        self.object_type = object_type
        self.bbox_2d = bbox_2d
        self.conf_score = conf_score
        self.instance_mask = instance_mask
        self.state = state

        self.tmp_unique_id = self.generate_tmp_unique_id()
        self.centroid_2d = np.array(
            [(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2]
        )
        self.area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])

        self.mask_3d_in_voxel = None
        self.centroid_3d = None
        self.instance_id_3d = None

        self.is_held = False

    def __repr__(self):
        return self.tmp_unique_id

    def to(self, device):
        self.instance_mask = self.instance_mask.to(device)
        return self

    def generate_tmp_unique_id(self):
        """
        Generate a unique id to identify the instance in current 2D detections.
        """
        return self.object_type + "_" + "_".join([str(i) for i in self.bbox_2d])

    def correct_object_type(self, gt_object_type):
        """
        Correct the object type of the instance.
        """
        print("correct_object_type: %s -> %s"%(self.object_type, gt_object_type))
        self.object_type = gt_object_type
        self.tmp_unique_id = self.generate_tmp_unique_id()
        self.state = {}
        

    def assign_3d_voxel_mask(self, mask_3d_in_voxel, centroid_3d):
        """
        Update the 3d voxel mask for the detected object.

        :param mask_3d_in_voxel: tensor of size (61, 61, 10)
        """
        self.mask_3d_in_voxel = mask_3d_in_voxel
        self.centroid_3d = centroid_3d

    def check_is_held(self, inventory_id_list: List[str]) -> bool:
        inventory_type_list = [i.split("_")[0] for i in inventory_id_list]
        W, H = self.instance_mask.shape
        x, y = self.centroid_2d

        def in_hand_position(x, y):
            return x > W * 0.43 and x < W * 0.57 and y > H * 0.5

        def can_be_placed_in_inventory():
            can_place_in_inventory = False
            for inventory_type in inventory_type_list:
                if inventory_type in get_object_receptacle_compatibility(self.object_type):
                    can_place_in_inventory = True
                    break
            return can_place_in_inventory

        # If the the object is in the lower middle of the frame (hand position) ...
        if "pickupable" in self.affordances and in_hand_position(x, y):
            print(self.object_type, "might be in the inventory")
            # ... and either its type is in the inventory
            if self.object_type in inventory_type_list:
                print(self.object_type, "is in inventory_type_list! ")
                self.is_held = True
            # or it can be placed in one of the inventoris
            if can_be_placed_in_inventory():
                print(self.object_type, "can be placed in inventory!")
                self.is_held = True

        # otherwise it is not held
        return self.is_held

    def get_interaction_point(self):
        """
        Get a valid interaction point of the instance.
        """

        centroid_on_grid = (
            int(np.round(self.centroid_2d[0])),
            int(np.round(self.centroid_2d[1])),
        )
        if self.instance_mask is None:
            return centroid_on_grid

        # if centroid is valid, use centroid
        is_valid = ObjectInstanceDetection2D.check_interaction_point_validity(
            point=centroid_on_grid,
            instance_mask=self.instance_mask,
            neighbor_width=ObjectInstanceDetection2D.INTERACTION_POINT_SIZE,
            mode='strict',
        )
        if is_valid:
            return centroid_on_grid

        ip_size = ObjectInstanceDetection2D.INTERACTION_POINT_SIZE
        sampling_time_per_size = ObjectInstanceDetection2D.SAMPLING_TIME

        mask_points = self.instance_mask.nonzero()

        is_valid = False
        
        for i in range(sampling_time_per_size * 5):
            # ensure tolerence up to 32 to find a valid point
            if (i + 1) % sampling_time_per_size == 0:
                ip_size = ip_size // 2
            sampled_point = random.choice(mask_points).cpu().numpy()
            # have to convert array index to image coordinate x, y ! 
            sampled_point[0], sampled_point[1] = sampled_point[1], sampled_point[0]
            # print(i, ip_size, is_valid, sampled_point, self.instance_mask.sum())
            is_valid = ObjectInstanceDetection2D.check_interaction_point_validity(
                point=sampled_point, 
                instance_mask=self.instance_mask,
                neighbor_width=ip_size,
                mode="strict",
            )
            if is_valid:
                break
        if not is_valid:
            return centroid_on_grid
        return sampled_point

    @classmethod
    def check_interaction_point_validity(
        cls,
        point: Union[torch.Tensor, np.ndarray, List[int]],
        instance_mask: torch.Tensor,
        neighbor_width: int = 10,
        mode: str = "tolerance",
    ) -> bool:
        """
        Check if the interaction point of the object instance is valid.
        """
        H, W = instance_mask.shape
        y_idx, x_idx = point  # have to transpose to obtain grid coordinate
        x_idx, y_idx = int(np.round(x_idx)), int(np.round(y_idx))

        offset = neighbor_width

        mask_slice = instance_mask[
            x_idx - offset : x_idx + offset + 1, y_idx - offset : y_idx + offset + 1
        ]

        if mode == "tolerance":
            return mask_slice.sum() > 0
        elif mode == "strict":
            return mask_slice.sum() == mask_slice.nelement()
        else:
            raise ValueError("Invalid mode: %s" % mode)

    @classmethod
    def parse_receptacles(cls, instances: List["ObjectInstanceDetection2D"]):
        def is_A_on_B(A: "ObjectInstanceDetection2D", B: "ObjectInstanceDetection2D"):
            Ax_min, Ay_min, Ax_max, Ay_max = A.bbox_2d
            A_cx, A_cy = A.centroid_2d
            Bx_min, By_min, Bx_max, By_max = B.bbox_2d

            if A_cx < Bx_min or A_cx > Bx_max:
                return False
            if A_cy > By_max or Ay_max < By_min:
                return False
            return True

        for above in instances:
            if above.is_held:
                continue
            above_type = above.object_type
            candidate_receps = get_object_receptacle_compatibility(above_type)
            for below in instances:
                # possible
                if below.object_type not in candidate_receps:
                    continue
                if is_A_on_B(above, below):
                    # print("%r is on %r" % (above, below))
                    if "parentReceptacles" not in above.state:
                        above.state["parentReceptacles"] = Relation(
                            "parentReceptacles", {below.tmp_unique_id: 0.8}
                        )
                    else:
                        above.state["parentReceptacles"].add(below.tmp_unique_id, 0.8, verbose=False)
                    if "receptacleObjectIds" not in below.state:
                        below.state["receptacleObjectIds"] = Relation(
                            "receptacleObjectIds", {above.tmp_unique_id: 0.8}
                        )
                    else:
                        below.state["receptacleObjectIds"].add(above.tmp_unique_id, 0.8, verbose=False)
