from typing import Union, Tuple, Optional, Dict, List
import numpy as np
import random

from sledmap.mapper.utils.base_cls import Action
from definitions.teach_objects import ithor_oid_to_object_class, objects_are_similar
from definitions.teach_actions import BotPrimitiveActions, NavigationActions, InteractionActions
from sledmap.mapper.env.teach.segmentation_definitions import object_string_to_intid
from sledmap.mapper.env.teach.teach_object_instance import ObjectInstance, ObjectInstanceDetection2D
from sledmap.mapper.models.teach.utils import euclidean_distance

from sledmap.mapper.env.teach.teach_env_params import FRAME_SIZE

class TeachAction(Action):
    def __init__(
        self, 
        action_type: str, 
        object_type: Optional[str] = None, 
        instance_id: Optional[str] = None, 
        instance_tmp_id_in_2d_detection: Optional[str] = None, 
        interaction_point: Optional[np.ndarray] = None
    ):
        super().__init__()
        assert TeachAction.is_valid(action_type), f"Invalid TeachAction type: {action_type}"
        self.action_type = action_type
        # self.action_id = BotPrimitiveActions.name_to_id(action_type) # TODO: add "None" to action ids
        
        self.object_type = object_type
        self.object_type_id = None if object_type is None else object_string_to_intid(object_type)

        self.instance_id = instance_id
        self.instance_tmp_id_in_2d_detection = instance_tmp_id_in_2d_detection
        self.interaction_point = interaction_point

        self.fail = False
    
    def __repr__(self):
        _repr = "Action: " + self.action_type
        if self.interaction_point is not None:
            x, y = self.interaction_point
            _repr += f" @({x:.2f}, {y:.2f})"
        if self.instance_id:
            _repr += f" 3D: {self.instance_id}"
        if self.instance_tmp_id_in_2d_detection:
            _repr += f" 2D: {self.instance_tmp_id_in_2d_detection}"
        return _repr
    
    def to_dict(self):
        if self.interaction_point is  None:
            return {"action_type": self.action_type}
        x, y = self.interaction_point
        return {
            "action_type": self.action_type,
            "object_type": self.object_type,
            "instance_id": self.instance_id,
            "instance_tmp_id_in_2d_detection": self.instance_tmp_id_in_2d_detection,
            "interaction_point": f"{int(x)},{int(y)}"
        }

    @classmethod
    def stop_action(cls):
        return cls("Stop")
    
    @classmethod
    def fail_action(cls):
        return cls("Failed")
    
    @classmethod
    def stop_navi_action(cls):
        return cls("EndNavi")

    @classmethod
    def is_valid(cls, action_type):
        return action_type == "None" or BotPrimitiveActions.has_action(action_type)
    
    def mark_as_failed(self):
        self.fail = True

    def to_teach_api(self) -> Tuple[str, Union[Tuple[float], None]]:
        return self.action_type, self.interaction_point
    
    def is_navigation(self):
        return NavigationActions.has_action(self.action_type)
    
    def is_interaction(self):
        return InteractionActions.has_action(self.action_type)

    def is_stop(self):
        return self.action_type == "Stop"
    
    def is_failed(self):
        return self.action_type == "Failed"
    
    def is_stop_navi(self):
        return self.action_type == "EndNavi"
    
    def add_interaction_point(self, interaction_point: np.ndarray):
        self.interaction_point = interaction_point
    
    @classmethod
    def create_empty_action(cls):
        return TeachAction("None")
    
    @classmethod
    def create_from_teach_api(
        cls, 
        teach_api_action: Dict[str, Union[str, float, None]], 
        symbolic_world_repr: "TeachSymbolicWorld", 
    ) -> "TeachAction":
        
        action_type = teach_api_action["action_name"]

        oid = teach_api_action["oid"]
        object_type = ithor_oid_to_object_class(oid) if oid else None
        
        x, y = teach_api_action["x"], teach_api_action["y"]
        x = x * FRAME_SIZE if x is not None else None
        y = y * FRAME_SIZE if y is not None else None

        if x is None or y is None or oid is None:
            return TeachAction(action_type, object_type)

        object_detections = symbolic_world_repr.latest_object_detections
        if not object_detections:
            print("Did not detect any objects, thus cannot find the instance that "
                  "was interacted with. May cause a problem in symbolic state update.")
            return TeachAction(action_type, object_type)

        i_point = np.array([x, y])
        # print(action_type, object_type, x, y)

        # first get the nearest 2D detections to the interaction point
        distances = []
        for idx, instance in enumerate(object_detections):
            distances.append((idx, euclidean_distance(i_point, instance.centroid_2d)))
        distances_sorted = sorted(distances, key=lambda x: x[1])
        
        #
        # then we select the instance whose mask contains the interaction point
        instance_candidates = []
        for instance_idx, _ in distances_sorted: 
            instance = object_detections[instance_idx]
            if instance.object_type == object_type:
                instance_candidates.append(instance)
        # for instance_idx, _ in distances_sorted: 
        #     instance = object_detections[instance_idx]
        #     if objects_are_similar(instance.object_type, object_type):
        #         instance_candidates.append(instance)
        # print(instance_candidates)
        is_valid = False
        instance = None
        for instance in instance_candidates:
            is_valid = ObjectInstanceDetection2D.check_interaction_point_validity(
                point=i_point,
                instance_mask=instance.instance_mask,
                neighbor_width=32, # =.=
                mode='tolerance'
            )
            if is_valid:
                print("Find valid point:", instance)
                break
        
        instance_id_mapping_2D_to_3D = symbolic_world_repr.instance_mapping_2to3
        if not is_valid:
            print("Correct instance type using ground truth information: ")
            # Sometimes our perception model does not predict the object type correctly
            # We use the ground truth object type to correct such errors
            instance = object_detections[distances_sorted[0][0]]
            wrong_instance_id_2d = instance.tmp_unique_id
            
            wrong_instance_id_3d = instance_id_mapping_2D_to_3D.get(wrong_instance_id_2d, None)
            if wrong_instance_id_3d is None:
                # sometimes we get a empty mask from 2D detection which will not register
                # to the world (due to empty volume). 
                pass 
            else:
                symbolic_world_repr.delete_instance(wrong_instance_id_3d)

            #correct instance type
            instance.correct_object_type(object_type)
            
            instance_id_2d = instance.tmp_unique_id
            instance_id_3d = symbolic_world_repr.register_new_instance(instance)
        else:
            instance_id_2d = instance.tmp_unique_id
            instance_id_3d = instance_id_mapping_2D_to_3D.get(instance_id_2d, None)
            # Note: in very rare cases, the 2D detection may not be registered to the world
            # this may due to that the location of the object is out of the voxel boundary 
            # e.g. pick something too high to the agent (i.e. above 2.5 meters for our setting)
            # we have a correction mechanism to for picking up but not other interactions
            

        # try:
        # assert objects_are_similar(object_type, instance.object_type)
        # except:
        #     from pprint import pprint
        #     pprint([(object_detections[i].tmp_unique_id, dis) for i, dis in distances_sorted])
        #     print(instance)
        #     print(object_type, instance.object_type)
            
        #     import torch
        #     torch.set_printoptions(profile="full")
        #     import ipdb; ipdb.set_trace()
        #     #  instance = object_detections[distances_sorted[0][0]]

        return TeachAction(
            action_type = action_type, 
            object_type = object_type, 
            instance_id = instance_id_3d, 
            instance_tmp_id_in_2d_detection = instance_id_2d, 
            interaction_point = i_point
        )

    @classmethod
    def create_action_with_instance(cls, action_type: str, instance_id_3d: str, detection: ObjectInstanceDetection2D) -> "TeachAction":
        return TeachAction(
            action_type = action_type, 
            object_type = detection.object_type, 
            instance_id = instance_id_3d, 
            instance_tmp_id_in_2d_detection = detection.tmp_unique_id, 
            interaction_point = detection.get_interaction_point()
        )


    def __eq__(self, other: "TeachAction"):
        return (
            self.action_type == other.action_type 
            and self.object_type == other.object_type 
        )
    

    