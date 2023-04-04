import copy
from typing import List, Dict, Union
from pprint import pprint

import random
import torch
import numpy as np

import definitions.teach_object_state as teach_obj_state

from definitions.teach_objects import ObjectClass, THING_NAMES
from definitions.teach_object_state import create_default_object_state
from definitions.teach_object_semantic_class import SemanticClass

from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.env.teach.teach_object_instance import ObjectInstance, ObjectInstanceDetection2D
from sledmap.mapper.models.alfred.voxel_grid import VoxelGrid
import sledmap.mapper.models.teach.utils as utils

WaterBottomReceptacles = SemanticClass.get_all_objs_in_semcls("WaterBottomReceptacles")
WaterContainers = SemanticClass.get_all_objs_in_semcls("WaterContainers")
StoveTopCookers = SemanticClass.get_all_objs_in_semcls("StoveTopCookers")
WaterTaps = SemanticClass.get_all_objs_in_semcls("WaterTaps")
OpenableRecepcales = SemanticClass.get_all_objs_in_semcls("OpenableReceptacles")
BigReceptacles = SemanticClass.get_all_objs_in_semcls("Tables") | {"CounterTop", "Bed", "Dresser", "Shelf", "Sofa"}
Landmarks = SemanticClass.get_all_objs_in_semcls("StandingReceptacles")

class TeachSymbolicWorld:
    def __init__(self, logger=None):
        self.close_threshold = 1.35 # consider objects within 1.4 meters to the agent as close 
        self._print = print if logger is None else logger.info
        # self._print = lambda x:x

        self.reset()
    
    def reset(self):
        self.instance_id_register = {c: set() for c in ObjectClass.get_all_names()}
        self.all_object_instances = {}
        self.instance_mapping_2to3 = {}
        self.inventory_instance_ids = []
        self.latest_object_detections = []
        self.agent_pose_m = None
        self.opened_stove_knob_ids = []
        self.instances_with_state_change = []
        self.instance_match_log_info = {}

    def update(
            self, 
            voxel_observability: VoxelGrid, 
            object_detections: List[ObjectInstanceDetection2D],
            last_succeed_action: TeachAction,
            agent_pose_m: torch.Tensor,
            verbose=True,
        ) -> "TeachSymbolicWorld":

            self.latest_object_detections = object_detections
            if not object_detections:
                return self
            
            # Record the current agent pose
            self.agent_pose_m_cuda = agent_pose_m.to(voxel_observability.data.device)
            self.agent_pose_m = agent_pose_m.numpy()
            self.vx_origin = voxel_observability.origin
            self.vx_size = voxel_observability.voxel_size

            # Link instances detected in the current observation to instances 
            # in the global symbolic world state
            self.match_detections_to_instances(
                voxel_observability,  
                object_detections,
                last_succeed_action,
                verbose=verbose,
            )

            instances_before_state_change = copy.deepcopy(self.all_object_instances)
            # Update the state of each instance
            self.update_object_physical_states(
                object_detections, 
                last_succeed_action,
                verbose=verbose,
            )

            self.instances_with_state_change = []
            state_change_log_str = "Instance State Changes: \n"
            for iid, instance in self.all_object_instances.items():
                instance_before = instances_before_state_change[iid]
                temp_str = ""
                for attr in instance.state:
                    if attr == 'distance':
                        continue
                    if instance_before.state[attr] != instance.state[attr]:
                        temp_str += f" * {attr}: {instance_before.state[attr]()} -> {instance.state[attr]()}\n"
                if temp_str:
                    state_change_log_str += "%r:\n%s"%(instance, temp_str)
                    self.instances_with_state_change.append(instance)
            if verbose:
                self._print(state_change_log_str)
                self._print("After Updating Inventory IDs: %r"%self.inventory_instance_ids)
        
            return self


    def get_all_instances(self):
        return self.all_object_instances.values()
    
    def get_all_instance_ids(self):
        return self.all_object_instances.keys()
    
    def get_instance(self, instance_id) -> ObjectInstance:
        return self.all_object_instances.get(instance_id, None)
    
    def get_instances_of_type(self, object_type: str) -> List[ObjectInstance]:
        return [i for i in self.get_all_instances() if i.object_type == object_type]
    
    def get_interactable_instances(self):
        return [i for i in self.get_all_instances() if i.state.interactable()]
    
    def get_visible_instances(self):
        return [i for i in self.get_all_instances() if i.state.visible()]
    
    def get_2D_detection_of_instance(self, instance_or_id: Union[str, ObjectInstance]) -> ObjectInstanceDetection2D:
        if isinstance(instance_or_id, ObjectInstance):
            instance_id = instance_or_id.instance_id
        else:
            instance_id = instance_or_id
        
        for detection in self.latest_object_detections:
            if instance_id == self.instance_mapping_2to3.get(detection.tmp_unique_id):
                return detection
        return None
    
    def get_hint_list(self, search_types: set, spatial_relations=None):
        hint_list = []
        if spatial_relations:
            for obj_type in search_types:
                landmarks = spatial_relations.get(obj_type, [])
                for landmark_type in landmarks:
                    if landmark_type not in Landmarks:
                        continue
                    # add up to 5 instance per spatial relation as hints
                    lm_instances = self.get_instances_of_type(landmark_type)[:5]
                    hint_list.extend([i.instance_id for i in lm_instances])
        
        if not hint_list:
            for instance in self.get_all_instances():
                if instance.object_type not in OpenableRecepcales:
                    continue
                if not instance.state.isOpen():
                    hint_list.append(instance)
            sorted_hint_list = sorted(hint_list, key=lambda x: x.voxel_count, reverse=True)
            hint_list = [instance.instance_id for instance in sorted_hint_list][:3]
        
        return hint_list
    
    def match_detections_to_instances(
            self, 
            voxel_observability: VoxelGrid, 
            object_detections: List[ObjectInstanceDetection2D],
            last_succeed_action: TeachAction,
            verbose: bool = False
        ):
        """
        @param state_repr: state representation
        @param observation: observation
        @return: list of list of ObjectInstance
        """
        
        batch_size= voxel_observability.data.shape[0]
        assert batch_size == 1, "Only support batch size of 1"
        obs_mask = voxel_observability.data[0].bool() #(1, 61, 61, 10)

        self.instance_mapping_2to3 = {}
        self.instance_match_log_info = {
            "new": "New instances:\n", 
            "match": "Matched instances:\n", 
            "delete": "Deleted instances:\n"
        }
        self.instance_volume_increment = {}
        self.new_instance_ids = []

        # register the objects from the first observation
        if not self.all_object_instances:
            for new_instance in object_detections:
                if new_instance.mask_3d_in_voxel.sum() == 0:
                    continue
                self.register_new_instance(new_instance, verbose)
                instance_id = self.instance_mapping_2to3.get(new_instance.tmp_unique_id, "???")
                new_instance.instance_id_3d = instance_id
            if verbose and self.instance_match_log_info['new'] != "New instances:\n":
                self._print(self.instance_match_log_info['new'][:-1])
            return self.instance_mapping_2to3
        
        # --------------------------- Begin match and update ---------------------------

        # Special handling for objects in hand: since inventories always have poor depth 
        # estimations, exlude them from the automatic merging process below, but manully
        # merge them here.
        detected_held_instances = [i for i in object_detections if i.is_held]
        detected_held_instance_types = [i.object_type for i in detected_held_instances]
        if self.inventory_instance_ids and last_succeed_action.action_type != "Place":
            exist_instances_in_hand_per_type = {}
            for exist_picked_instance_id in self.inventory_instance_ids:
                instance = self.get_instance(exist_picked_instance_id)
                if instance is not None:
                    ins_type = instance.object_type
                    if ins_type not in exist_instances_in_hand_per_type:
                        exist_instances_in_hand_per_type[ins_type] = []
                    exist_instances_in_hand_per_type[ins_type].append(instance)

            self._print("########### Match inventories ")
            self._print("exist_instances_in_hand_per_type")
            self._print(exist_instances_in_hand_per_type)
            self._print("detected_held_instances")
            self._print(detected_held_instances)
            self.match_picked_or_placed_instances(
                exist_instances_in_hand_per_type, 
                detected_held_instances,
                obs_mask,
                verbose=verbose
            )
        
        
        # Get currently visible registered instances of each object class. 
        all_exist_instances = self.get_all_instances()
        exist_ins_3d_masks = [i.voxel_mask for i in all_exist_instances]
        exist_ins_3d_masks = torch.stack(exist_ins_3d_masks, dim=0).bool() #(n, 61, 61, 10)
        
        # For each of the registered instance, if any of its voxel is visible in the current 
        # observability map, we consider it as visible. 
        exist_ins_3d_masks_vis = (exist_ins_3d_masks & obs_mask).sum(dim = (1, 2, 3)) > 0

        visible_exist_instances_per_class = {c:[] for c in ObjectClass.get_all_names()}
        for idx, instance in enumerate(all_exist_instances):
            if not exist_ins_3d_masks_vis[idx] or instance.instance_id in self.inventory_instance_ids:
                continue
            visible_exist_instances_per_class[instance.object_type].append(instance)
        
        # candidate_exist_instances_picked = {}
        # if last_succeed_action.action_type == "Pickup":
        #     for o_type in detected_held_instance_types:
        #         candidate_exist_instances_picked[o_type] = \
        #             visible_exist_instances_per_class[o_type]

        # Get the number of 2D detections of each object class. 
        new_instances_per_class = {}
        for detection in object_detections:
            if detection.object_type not in new_instances_per_class:
                new_instances_per_class[detection.object_type] = []
            if detection.is_held:
                continue
            new_instances_per_class[detection.object_type].append(detection)
        
        # Compare the number of visible registered instances with the number of 2D detections
        # for each object class, and match 2D detections to existing instances.
        new_instances_spawned = []
        candidate_exist_instances_picked = {}
        for obj_cls, new_instances in new_instances_per_class.items():

            exist_instances_vis = visible_exist_instances_per_class[obj_cls]
            new_instance_centroids = [i.centroid_3d for i in new_instances]
            exist_instance_centroids = [i.centroid_3d for i in exist_instances_vis]
            
            idx_pair_list = utils.compare_and_match_centroids(
                new_instance_centroids, exist_instance_centroids
            )
            # if obj_cls == 'Mug':
            #     pprint(new_instances)
            #     pprint(exist_instances_vis)
            #     pprint(new_instance_centroids)
            #     pprint(exist_instance_centroids)
            #     pprint(idx_pair_list)
            
            for new_idx, exist_idx in idx_pair_list:

                # If the number of 2D detected instances < the number of visible existing 
                # instances, delete the visible voxels from the existing instance.  
                # If there is no voxels left, delete this instance.
                if new_idx is None:
                    exist_instance = exist_instances_vis[exist_idx]
                    exist_instance_type = exist_instance.object_type
                    # if verbose:
                    #     self._print('Update not detected instance: ' + str(exist_instance))
                    
                    # Special handling for instances just picked up:
                    # Match the "disappered" existing instance to the instances considered 
                    # as picked in the 2D detections.
                    if (
                        last_succeed_action.action_type == "Pickup"
                        and exist_instance_type in detected_held_instance_types
                    ):
                        self._print("%r may be just picked up: skip updating"%exist_instance)
                        if exist_instance_type not in candidate_exist_instances_picked:
                            candidate_exist_instances_picked[exist_instance_type] = []
                        candidate_exist_instances_picked[exist_instance_type].append(exist_instance)
                        continue

                    # Special handling for isInsideClosed instances
                    instance_id = exist_instance.instance_id
                    isInsideClosed = exist_instance.state.isInsideClosed()
                    if last_succeed_action.action_type == "Close" and not isInsideClosed:
                        # Note that the state isInsideClosed is not be updated yet, so we
                        # need to handle the case that the latest action is `close`.
                        closed_instance = self.get_instance(last_succeed_action.instance_id)
                        if closed_instance is not None:
                            contains = closed_instance.state.receptacleObjectIds()
                            if instance_id in contains:
                                isInsideClosed = True
                    
                    if isInsideClosed:
                        if verbose:
                            self._print('Skip the instance that hides inside a closed container') 
                        continue
                    
                    # Special handling for instances just got sliced: no need

                    # Normal update: keep only the unobservable part of the voxel mask
                    exist_mask_old = exist_instance.voxel_mask.clone()
                    exist_mask_new = exist_mask_old & (~obs_mask.squeeze())
                    instance_info_before = str(exist_instance)
                    exist_instance.update_voxel_mask(exist_mask_new)
                    if exist_instance.voxel_count == 0:
                        if exist_instance.is_interacted or exist_instance.instance_id == 'ButterKnife_1':
                            # we do not want to delete interacted instances
                            # so just use the old voxel mask to avoid empty volume
                            exist_instance.update_voxel_mask(exist_mask_old)
                            self._print('For interacted instances using the old mask: %r'%(exist_instance))
                        else:
                            self.delete_instance(exist_instance, verbose)
                    else:
                        self.instance_match_log_info['match'] += (
                            ' r ' + instance_info_before + ' -> ' + str(exist_instance) + '\n'
                        )
                    continue
                
                # If the number of 2D detected instances > the number of visible existing 
                # instances, register the new instance. 
                if exist_idx is None:
                    new_instance = new_instances[new_idx]
                    
                    # Special handling for instances just placed:
                    # Such instances suddently appeared in the 2D detections, but they should
                    # be matched with existing instances in the inventory. 
                    if last_succeed_action.action_type == "Place":
                        inventory_types = [i.split("_")[0] for i in self.inventory_instance_ids]
                        new_instance_type = new_instance.object_type
                        if new_instance_type in inventory_types:
                            self._print("Add new instance to new_instances_spawned")
                            new_instances_spawned.append(new_instance)
                            # exist_instances_of_type = self.get_instances_of_type(new_instance_type)
                            # exist_instances_just_got_placed[new_instance_type] = exist_instances_of_type
                            continue
                        self._print(
                            "Warning: find new instance %s not in inventory after Placing [%r]?"%(
                                new_instance.tmp_unique_id, self.inventory_instance_ids
                            )
                        )
                    
                    # Special handling for actions that should not discover new instances
                    if last_succeed_action.action_type in ["Place", "Close", "ToggleOn", "ToggleOff", "Pour"]:
                        self._print(
                            "Warning: find new instance %s after %s?"%(
                                new_instance.tmp_unique_id, last_succeed_action
                            )
                        )
                        # continue
                    
                    # Normal update: register the new instance
                    if new_instance.mask_3d_in_voxel.sum() > 0:
                        self.register_new_instance(new_instances[new_idx], verbose)
                    continue
                
                # If a new instance is matched with an existing instance, update the
                # existing instance's voxel mask based on the new observation
                new_instance = new_instances[new_idx]
                exist_instance = exist_instances_vis[exist_idx]
                instance_info_before = str(exist_instance)
                exist_instance_volume = exist_instance.voxel_count
                self.instance_mapping_2to3[new_instance.tmp_unique_id] = exist_instance.instance_id
                # if verbose:
                #     self._print('Match: %s %s'%(new_instance.tmp_unique_id, str(exist_instance)))
                #     self._print('    Update: ' + str(exist_instance))
                
                old_mask_invis = exist_instance.voxel_mask & (~obs_mask.squeeze())
                new_mask = new_instance.mask_3d_in_voxel
                new_mask = new_mask | old_mask_invis
                
                exist_instance.update_voxel_mask(new_mask)
                
                self.instance_match_log_info['match'] += (
                    ' m ' + instance_info_before + ' -> ' + str(exist_instance) + '\n'
                )
                self.instance_volume_increment[exist_instance.instance_id] = (
                    exist_instance.voxel_count - exist_instance_volume
                )
                # if verbose:
                #     self._print('         -> ' + str(exist_instance))
                #     self._print('    old_mask_invis:' + str(old_mask_invis.sum()))

        if last_succeed_action.action_type == "Pickup":
            self._print("########### Update instances after Pickup")
            self._print("candidate_exist_instances_picked")
            self._print(candidate_exist_instances_picked)
            self._print("detected_held_instances")
            self._print(detected_held_instances)
            self.match_picked_or_placed_instances(
                candidate_exist_instances_picked, 
                detected_held_instances,
                obs_mask,
                verbose=verbose
            )

            if last_succeed_action.instance_id is None and detected_held_instances:
                self._print("A successful Pickup without an instance id? Try to register a new id for it")
                # a detected may not successfully registered into the map due to it is outside the voxel boundary.
                # in that case, we have to correct the registeration here.
                picked_type = last_succeed_action.object_type
                for det in detected_held_instances:
                    if det.object_type == picked_type and det.tmp_unique_id not in self.instance_mapping_2to3:
                        interact_instance_id = self.register_new_instance(det, verbose)
                        last_succeed_action.instance_id = interact_instance_id

        elif last_succeed_action.action_type == "Place":
            candidate_exist_instances_placed = {}
            for inventory_id in self.inventory_instance_ids:
                instance = self.get_instance(inventory_id)
                if instance is not None:
                    o_type = instance.object_type
                    if o_type not in candidate_exist_instances_placed:
                        candidate_exist_instances_placed[o_type] = []
                    candidate_exist_instances_placed[o_type].append(instance)

            self._print("########### Update instances after Place")
            self._print("candidate_exist_instances_placed")
            self._print(candidate_exist_instances_placed)
            self._print("new_instances_spawned")
            self._print(new_instances_spawned)
            self.match_picked_or_placed_instances(
                candidate_exist_instances_placed, 
                new_instances_spawned,
                obs_mask,
                verbose=verbose
            ) 

        delete_ids = []
        for i in self.get_all_instances():
            if i.voxel_mask.sum() == 0: 
                self._print("Instance should %s has no voxels???"%i.instance_id)
                self._print("Should not happen!!!")
                delete_ids.append(i.instance_id)
        for iid in delete_ids:
            self.delete_instance(self.get_instance(iid))
        
        if verbose:
            if self.instance_match_log_info['new'] != "New instances:\n":
                self._print(self.instance_match_log_info['new'][:-1])
            if self.instance_match_log_info['delete'] != "Deleted instances:\n":
                self._print(self.instance_match_log_info['delete'][:-1])
            # if self.instance_match_log_info['match'] != "Matched instances:\n":
            #     self._print(self.instance_match_log_info['match'][:-1])
        
        for detection in object_detections:
            instance_id = self.instance_mapping_2to3.get(detection.tmp_unique_id, "???")
            detection.instance_id_3d = instance_id

    
    def match_picked_or_placed_instances(
        self, 
        exist_instances_per_type: Dict[str, ObjectInstance], 
        new_instances: List[ObjectInstanceDetection2D], 
        obs_mask: torch.Tensor,
        verbose=False
    ):
        # match between 2D detections believed as held and existing instances
        for obj_type, exist_instances in exist_instances_per_type.items():
            new_instances_of_type = [i for i in new_instances if i.object_type == obj_type]
            new_instance_centroids = [i.centroid_3d for i in new_instances_of_type]
            exist_instance_centroids = [i.centroid_3d for i in exist_instances]
            idx_pair_list = utils.compare_and_match_centroids(
                new_instance_centroids, exist_instance_centroids
            )
            # pprint(new_instances_of_type)
            # pprint(new_instance_centroids)

            # pprint(exist_instances)
            # pprint(exist_instance_centroids)
                
            # print(obj_type, idx_pair_list)
            for new_idx, exist_idx in idx_pair_list:
                if new_idx is None: 
                    # not matched to any inventory
                    self._print(
                        'Warning: exist instance %s not matched to any picked/placed/inventory'%(
                            exist_instances[exist_idx]
                        )
                    )
                    continue
                
                if exist_idx is None:
                    new_instance = new_instances_of_type[new_idx]
                    # if new_instance.mask_3d_in_voxel.sum() > 0:
                    self._print("Warning: failed to match_picked_or_placed_instances: "
                                "no exist instance. Add it anyway.")
                    self.register_new_instance(new_instances[new_idx], verbose)
                    
                    continue
                
                new_instance = new_instances_of_type[new_idx]
                exist_instance = exist_instances[exist_idx]
                self._print(
                    'Match inventory: %s -> %s'%(
                        new_instance.tmp_unique_id, str(exist_instance)
                    )
                )
                exist_instance.update_voxel_mask(new_instance.mask_3d_in_voxel)
                assert exist_instance.voxel_count > 0
                self.instance_mapping_2to3[new_instance.tmp_unique_id] = exist_instance.instance_id
        # print("### End match_picked_or_placed_instances")
    
    def update_object_physical_states(
        self, 
        object_detections: List[ObjectInstanceDetection2D],
        last_succeed_action: Union[TeachAction, None],
        verbose: bool = True
    ):
        object_detection_id_to_instance = {
            i.tmp_unique_id: i for i in object_detections
        }
        instance_mapping_3to2 = {
            v: k for k, v in self.instance_mapping_2to3.items()
        }
        
        
        # First update object distance, visibility and interactability 
        for instance in self.get_all_instances():
            # if verbose:
            #     # self._print("%s %s %s"%(str(instance), "visible", "interactable"))
            if instance.voxel_count > 10:
                instance.update_nearest_voxel_coord(self.agent_pose_m_cuda, self.vx_size, self.vx_origin)
                instance.update_distance(self.agent_pose_m, mode='nearest_voxel')
            else:
                instance.update_distance(self.agent_pose_m, mode='centroid')

            if instance.instance_id in instance_mapping_3to2:
                instance.state["visible"].set_value(True)
                if instance.state.distance() < self.close_threshold:
                    instance.state["interactable"].set_value(True)
                else:
                    instance.state["interactable"].set_value(False)
            else:
                instance.state["visible"].set_value(False)
                instance.state["interactable"].set_value(False)
            
            # for big receptacles such as countertop, if any of its child is interactable
            # then the receptacle is interactable
            if instance.object_type in BigReceptacles:
                for child_id in instance.state.receptacleObjectIds():
                    child_instance = self.get_instance(child_id)
                    if child_instance is not None and child_instance.state.interactable():
                        instance.state["interactable"].set_value(True)
                        break

        # Then update the state based on state estimations in 2D detections        
        for instance_tmp_id_2d, instance_id in self.instance_mapping_2to3.items():
            instance_2d = object_detection_id_to_instance[instance_tmp_id_2d]
        
            instance = self.get_instance(instance_id)
            if instance is not None and instance_2d.state:
                instance.update_state_from_detection(instance_2d, self.instance_mapping_2to3)
        
        # link facuet to sink
        faucets = self.get_instances_of_type("Faucet")
        for obj_type in SemanticClass.get_all_objs_in_semcls('WaterBottomReceptacles'):
            for instance in self.get_instances_of_type(obj_type):
                loc = instance.centroid_3d
                min_dis, cloest_faucet = np.inf, None
                for faucet in faucets:
                    dis = np.linalg.norm(faucet.centroid_3d - loc)
                    if dis < min_dis:
                        min_dis = dis
                        cloest_faucet = faucet
                if cloest_faucet is not None:
                    instance.state["receptacleObjectIds"].add(cloest_faucet.instance_id)
                    cloest_faucet.state["parentReceptacles"].add(instance.instance_id)
                    # self._print("Link %s to %s"%(instance, cloest_faucet))

        if last_succeed_action is None or not last_succeed_action.is_interaction():
            return
        # Finally update the state based on the last executed action.
        # Note that the effect of action is more realiable thus we use it to 
        # overwrite the state updating from 2D estimations.
        action_type = last_succeed_action.action_type
        interact_instance_id = last_succeed_action.instance_id
        interact_instance = self.get_instance(interact_instance_id)

        if interact_instance is None:
            # if verbose:
            self._print("A successful interaction without an instance to interact with? "
                "This should not happen!")
            if action_type == "Pickup":
                # This may due to picked up a incorrectly recognized instance, which is deleted
                # after instance match.
                if len(self.new_instance_ids) == 1:
                    # if there is only one new instance, we can safely assume it is the one picked up
                    interact_instance_id = self.new_instance_ids[0]
                    interact_instance = self.get_instance(interact_instance_id)
                else:
                    # Try to correct the picked instance by selecting the instance that has the largest volume change.
                    interact_instance_id = max(self.instance_volume_increment, key=self.instance_volume_increment.get)
                    interact_instance = self.get_instance(interact_instance_id)
                # if verbose:
                self._print("Correct the picked instance to: %r"%interact_instance)
            else:
                return
        
        interact_instance.mark_interacted()
        self._print(str(last_succeed_action))


        if action_type == 'Pickup':
            # Update the state of the directly picked instance
            # Note that directly picked instance is alwalys the first one in the list
            if verbose:
                self._print("Update the state of the directly picked instance")
                self._print("%s %s %s %s"%(str(interact_instance), "isPickedUp", "parentReceptacles", "isPlacedOn"))
            
            self.inventory_instance_ids.append(interact_instance_id)
            interact_instance.state["isPickedUp"].set_value(True)
            interact_instance.state["simbotIsPickedUp"].set_value(True)
            interact_instance.state["parentReceptacles"].update({})
            interact_instance.state["isPlacedOn"].set_value(None)
            
            
            # Update the state of instances that on/in the picked instance
            for iid in interact_instance.state.receptacleObjectIds():
                instance = self.get_instance(iid)
                if instance is not None:
                    self.inventory_instance_ids.append(iid)
                    
                    if verbose:
                        self._print("Update the state of instances that on/in the picked instance")
                        self._print("%s %s"%(str(instance), "isPickedUp"))
                    
                    instance.state["isPickedUp"].set_value(True)
            
            # remove them from their original parent receptcale's receptacleObjectIds
            for parent_instance in self.get_all_instances():
                childs = parent_instance.state.receptacleObjectIds()
                if childs:
                    for picked_id in self.inventory_instance_ids:
                        if picked_id in childs:
                            parent_instance.state["receptacleObjectIds"].remove(picked_id)

            # Old incorrect version 
            # for picked_iid in self.inventory_instance_ids:
            #     picked_instance = self.get_instance(picked_iid)
            #     for iid in picked_instance.state.parentReceptacles():
            #         old_receptcale = self.get_instance(iid)
            #         if old_receptcale:
            #             old_receptcale.state["receptacleObjectIds"].remove(iid)
            
            # remove their original parent receptcale's from their parentReceptacles
            for iid in interact_instance.state.receptacleObjectIds():
                instance = self.get_instance(iid)
                if instance is not None:
                    for iid_parent_recep in instance.state.parentReceptacles():
                        if iid_parent_recep not in self.inventory_instance_ids:
                            
                            if verbose:
                                self._print("Update the state of instances that on/in the picked instance")
                                self._print("%s %s"%(str(instance), "parentReceptacles"))
                            
                            instance.state["parentReceptacles"].remove(iid_parent_recep)
            # Note: if A on B on C on D, picked C, we ensure D is removed from the 
            #       parentReceptacles of A, B, C, while A, B, C are still stacked

        elif action_type == 'Place':

            if len(self.inventory_instance_ids) == 0:
                if verbose:
                    self._print("Place succeeds while no object is in hand? "
                        "This should not happen!")
                return
            
            # Update the object spatial relationships
            for iid in self.inventory_instance_ids:
                instance = self.get_instance(iid)
                if instance is not None:
                    if verbose:
                        self._print("Update the state of the directly picked instance")
                        self._print("%s %s %s"%(str(instance), "isPickedUp", "parentReceptacles"))

                    instance.state["isPickedUp"].set_value(False)
                    instance.state["parentReceptacles"].add(interact_instance_id)

                    if verbose:
                        self._print("Update the state of the receptacle")
                        self._print("%s %s"%(str(interact_instance), "receptacleObjectIds"))

                    interact_instance.state["receptacleObjectIds"].add(iid)

            # Update contextualized interaction outcomes
            # 1. Place dirty object under running water -> clean
            # 2. Place fillable container under running water -> filled with water
            # 3. Place mug under working coffeemachine -> filled with coffee 
            # 4. Place cookable on top of pot/pan on top of open stove -> cooked
            # 5. Place boilable in pot with water on top of open stove -> boiled
            # 6. Place pot with water and boiled on top of open stove -> boiled
            # 7. Place bread slice into working toaster -> cooked

            placed_id = self.inventory_instance_ids[0]
            placed_instance = self.get_instance(placed_id)
            recep_id = interact_instance_id
            recep_instance = self.get_instance(recep_id)
            if placed_instance is None:
                self._print("Inventory instance not found! Something went wrong")
                return

            if verbose:
                self._print("Update the state of the placed_instance")
                self._print("%s %s"%(str(placed_instance), "isPlacedOn"))

            placed_instance.state["isPlacedOn"].set_value(recep_id)

            # 1. Place dirty object under running water -> clean
            if (
                placed_instance.state.isDirty() 
                and recep_instance.object_type in WaterBottomReceptacles
            ):
                faucet_instance_2d = utils.find_nearest_visible_instance_with_type(
                    target_centroid=last_succeed_action.interaction_point,
                    object_type='Faucet', 
                    visible_instances=object_detections
                )
                if faucet_instance_2d is not None:
                    faucet_id = self.instance_mapping_2to3[faucet_instance_2d.tmp_unique_id]
                    faucet = self.get_instance(faucet_id)
                    if faucet is not None and faucet.state.isToggled():
                        
                        if verbose:
                            self._print("Place dirty object under running water -> clean")
                            self._print("Open faucet: " + str(faucet))
                            self._print("%s %s"%(str(placed_instance), "isDirty"))
                        
                        placed_instance.state["isDirty"].set_value(False)
            
            # 2. Place fillable container under running water -> filled with water
            if (
                placed_instance.object_type in WaterContainers
                and recep_instance.object_type in WaterBottomReceptacles
            ):
                faucet_instance_2d = utils.find_nearest_visible_instance_with_type(
                    target_centroid=last_succeed_action.interaction_point,
                    object_type='Faucet', 
                    visible_instances=object_detections
                )
                if faucet_instance_2d is not None:
                    faucet_id = self.instance_mapping_2to3[faucet_instance_2d.tmp_unique_id]
                    faucet = self.get_instance(faucet_id)
                    if faucet is not None and faucet.state.isToggled():
                        
                        if verbose:
                            self._print("Place fillable container under running water -> filled with water")
                            self._print("Open faucet: " + str(faucet))
                            self._print("%s %s %s"%(str(placed_instance), "isFilledWithLiquid", "simbotIsFilledWithWater"))

                        placed_instance.state["isFilledWithLiquid"].set_value(True)
                        placed_instance.state["simbotIsFilledWithWater"].set_value(True)
            
            # 3. Place mug under working coffeemachine -> filled with coffee 
            if (
                placed_instance.object_type == 'Mug'
                and recep_instance.object_type == 'CoffeeMachine'
                and recep_instance.state.isToggled()
            ):
                if verbose:
                    self._print("Place mug under working coffeemachine -> filled with coffee ")
                    self._print("%s %s %s"%(str(placed_instance), "isFilledWithLiquid", "simbotIsFilledWithCoffee"))
                
                placed_instance.state["isFilledWithLiquid"].set_value(True)
                placed_instance.state["simbotIsFilledWithCoffee"].set_value(True)
            
            # 4. Place cookable on top of pot/pan on top of open stove -> cooked
            if (
                "stovecookable" in placed_instance.affordances
                and recep_instance.object_type in StoveTopCookers
            ):
                stove_instance_2d = utils.find_nearest_visible_instance_with_type(
                    target_centroid=last_succeed_action.interaction_point,
                    object_type='StoveBurner', 
                    visible_instances=object_detections
                )
                if stove_instance_2d is not None:
                    stove_id = self.instance_mapping_2to3[stove_instance_2d.tmp_unique_id]
                    stove = self.get_instance(stove_id)
                    if stove is not None and stove.state.isToggled():
                        
                        if verbose:
                            self._print("Place cookable on top of pot/pan on top of open stove -> cooked")
                            self._print("Open stove: " + str(stove))
                            self._print("%s %s "%(str(placed_instance), "isCooked"))
                        
                        placed_instance.state["isCooked"].set_value(True)
            
            # 5. Place boilable in pot with water on top of open stove -> boiled
            if (
                "boilable" in placed_instance.affordances
                and recep_instance.object_type == 'Pot'
                and recep_instance.state.simbotIsFilledWithWater()
            ):
                stove_instance_2d = utils.find_nearest_visible_instance_with_type(
                    target_centroid=last_succeed_action.interaction_point,
                    object_type='StoveBurner', 
                    visible_instances=object_detections
                )
                if stove_instance_2d is not None:
                    stove_id = self.instance_mapping_2to3[stove_instance_2d.tmp_unique_id]
                    stove = self.get_instance(stove_id)
                    if stove is not None and stove.state.isToggled():

                        if verbose:
                            self._print("Place boilable in pot with water on top of open stove -> boiled")
                            self._print("Open stove: " + str(stove))
                            self._print("%s %s "%(str(placed_instance), "simbotIsBoiled"))

                        placed_instance.state["simbotIsBoiled"].set_value(True)

            # 6. Place pot with water and boiled on top of open stove -> boiled
            if (
                placed_instance.object_type == 'Pot'
                and placed_instance.state.simbotIsFilledWithWater()
            ):
                stove_instance_2d = utils.find_nearest_visible_instance_with_type(
                    target_centroid=last_succeed_action.interaction_point,
                    object_type='StoveBurner', 
                    visible_instances=object_detections
                )
                if stove_instance_2d is not None:
                    stove_id = self.instance_mapping_2to3[stove_instance_2d.tmp_unique_id]
                    stove = self.get_instance(stove_id)
                    if stove is not None and stove.state.isToggled():
                        for iid in placed_instance.state.receptacleObjectIds():
                            instance = self.get_instance(iid)
                            if instance is not None and "boilable" in instance.affordances:
                            
                                if verbose:
                                    self._print("Place pot with water and boiled on top of open stove -> boiled")
                                    self._print("Open stove: " + str(stove))
                                    self._print("%s %s "%(str(instance), "simbotIsBoiled"))

                                instance.state["simbotIsBoiled"].set_value(True)
            
            # 7. Place bread slice into working toaster -> cooked
            if (
                "toastable" in placed_instance.affordances
                and recep_instance.object_type == 'Toaster'
            ):
                if recep_instance.state.isToggled():
                    
                    if verbose:
                        self._print("Place bread slice into working toaster -> cooked")
                        self._print("%s %s "%(str(placed_instance), "isCooked"))
                    
                    placed_instance.state["isCooked"].set_value(True)

            # Reset the inventory list
            self.inventory_instance_ids = []
        
        elif action_type == 'Open':

            if verbose:
                self._print("%s %s"%(str(interact_instance), "isOpen"))

            interact_instance.state["isOpen"].set_value(True)

            for iid in interact_instance.state.receptacleObjectIds():
                instance = self.get_instance(iid)
                if instance is not None and instance.state.visible():

                    if verbose:
                        self._print("%s %s"%(str(instance), "isInsideClosed"))

                    instance.state["isInsideClosed"].set_value(False)
        
        elif action_type == 'Close':

            if verbose:
                self._print("%s %s"%(str(interact_instance), "isOpen"))

            interact_instance.state["isOpen"].set_value(False)
            for iid in interact_instance.state.receptacleObjectIds():
                instance = self.get_instance(iid)
                if instance is not None and not instance.state.visible():

                    if verbose:
                        self._print("%s %s"%(str(instance), "isInsideClosed"))

                    instance.state["isInsideClosed"].set_value(True)
        
        elif action_type == 'ToggleOn':
            if verbose:
                self._print("%s %s"%(str(interact_instance), "isToggled"))
            interact_instance.state["isToggled"].set_value(True)
            # Note that the toggle status of stoves can only be detected from perceptions

            # contextualized interaction outcomes
            # 1. Toggle on a faucet will (1) clean every dirty instance in the sink
            #    and (2) fill every container in the sink with water
            # 2. Toggle on a coffee machine with mug on it will fill the mug with coffee
            # 3. Toggle on a stove will cook... Boil...
            # 4. Toggle on a microwave will cook... boil...
            # 5. Toggle on a toaster will cook the breadslice inside it

            # 1. Toggle on a faucet will (1) clean every dirty instance in the sink
            if interact_instance.object_type in WaterTaps:
                instance_ids_in_sink = []
                # sink/sinkbasin is messed up in ai2thor
                sink_instance_2d = utils.find_nearest_visible_instance_with_type(
                    target_centroid=last_succeed_action.interaction_point,
                    object_type=["Sink", "Bathtub"], 
                    visible_instances=object_detections
                )
                if sink_instance_2d is not None:
                    sink_id = self.instance_mapping_2to3[sink_instance_2d.tmp_unique_id]
                    sink = self.get_instance(sink_id)
                    if sink is not None:
                        instance_ids_in_sink.extend(
                            sink.state.receptacleObjectIds()
                        )
                
                basin_instance_2d = utils.find_nearest_visible_instance_with_type(
                    target_centroid=last_succeed_action.interaction_point,
                    object_type=["SinkBasin", "BathtubBasin"], 
                    visible_instances=object_detections
                )
                if basin_instance_2d is not None:
                    basin_id = self.instance_mapping_2to3[basin_instance_2d.tmp_unique_id]
                    basin = self.get_instance(basin_id)
                    if basin is not None:
                        instance_ids_in_sink.extend(
                            basin.state.receptacleObjectIds()
                        )

                for iid in instance_ids_in_sink:
                    instance = self.get_instance(iid)
                    if instance is not None:
                        if "dirtyable" in instance.affordances:
                            
                            if verbose:
                                self._print("Toggle on a faucet will clean every dirty instance in the sink")
                                self._print("%s %s"%(str(instance), "isDirty"))
                            
                            instance.state["isDirty"].set_value(False)
                        if "canFillWithLiquid" in instance.affordances:

                            if verbose:
                                self._print("Toggle on a faucet will fill every container in the sink with water")
                                self._print("%s %s"%(str(instance), "simbotIsFilledWithWater"))

                            instance.state["simbotIsFilledWithWater"].set_value(True)
            
            # 2. Toggle on a coffee machine with mug on it will fill the mug with coffee
            if interact_instance.object_type == "CoffeeMachine":
                for i in interact_instance.state.receptacleObjectIds():
                    instance = self.get_instance(i)
                    if instance is not None and instance.object_type == "Mug":

                        if verbose:
                            self._print("Toggle on a coffee machine with mug on it will fill the mug with coffee")
                            self._print("%s %s %s"%(str(instance), "isFilledWithLiquid", "simbotIsFilledWithCoffee"))

                        instance.state["isFilledWithLiquid"].set_value(True)
                        instance.state["simbotIsFilledWithCoffee"].set_value(True)
            
            # 3. Toggle on a stove will cook... Boil...
            # Note: since we do not know the stove-knob mapping, we can only update the state
            # based on the perception, or by toggling all the stove knobs
            if interact_instance.object_type == "StoveKnob":
                print("self.opened_stove_knob_ids", self.opened_stove_knob_ids)
                self.opened_stove_knob_ids.append(interact_instance.instance_id)
                all_burners = self.get_instances_of_type("StoveBurner")
                open_burners = [i for i in all_burners if i.state.isToggled()]
                burners_to_toggle = []
                if len(self.opened_stove_knob_ids) >= 4:
                    burners_to_toggle = all_burners
                    if verbose:
                        self._print("Marked all the stoves as on since all the knobs are toggled")
                
                # elif len(self.opened_stove_knob_ids) > len(open_burners):
                #     off_burners = [i for i in all_burners if not i.state.isToggled()]
                #     number = len(self.opened_stove_knob_ids) - len(open_burners)
                #     burners_to_toggle = random.sample(off_burners, number)
                #     if verbose:
                #         self._print("The number of toggled knobs is larger than open burners,"
                #                     " so we will toggle on %d burners"%number)
                    
                for burner in burners_to_toggle:
                    if verbose:
                        self._print("%s %s"%(str(burner), "isToggled"))
                    burner.state["isToggled"].set_value(True)

                
                for open_burner in open_burners:
                    stove_holds = open_burner.state.receptacleObjectIds()

                    for iid in stove_holds:
                        instance = self.get_instance(iid)
                        if instance is not None and "stovecookable" in instance.affordances:
                            
                            if verbose:
                                self._print("Toggle on a stove will cook...")
                                self._print("%s %s "%(str(instance), "isCooked"))
                            
                            instance.state["isCooked"].set_value(True)
                        if instance is not None and "boilable" in instance.affordances:
                            pot_ids = [i for i in instance.state.parentReceptacles() if "Pot" in i]
                            if pot_ids:
                                 pot = self.get_instance(pot_ids[0])
                                 if pot is not None and pot.state.simbotIsFilledWithWater():

                                     if verbose:
                                        self._print("Toggle on a stove will boil...")
                                        self._print("%s %s "%(str(instance), "simbotIsBoiled"))

                                     instance.state["simbotIsBoiled"].set_value(True)

            # 4. Toggle on a microwave will cook... boil...
            if interact_instance.object_type == "Microwave":
                for i in interact_instance.state.receptacleObjectIds():
                    instance = self.get_instance(i)
                    if instance is not None and "microwavable" in instance.affordances:
                        
                        if verbose:
                            self._print("Toggle on a microwave will cook... ")
                            self._print("%s %s "%(str(instance), "isCooked"))
                        
                        instance.state["isCooked"].set_value(True)
                    
                    if instance is not None and "boilable" in instance.affordances:
                        bowl_ids = [i for i in instance.state.parentReceptacles() if "Bowl" in i]
                        if bowl_ids:
                            bowl = self.get_instance(bowl_ids[0])
                            if bowl is not None and bowl.state.simbotIsFilledWithWater():

                                if verbose:
                                    self._print("Toggle on a microwave will boil... ")
                                    self._print("%s %s "%(str(instance), "simbotIsBoiled"))

                                instance.state["simbotIsBoiled"].set_value(True)
                    
            # 5. Toggle on a toaster will cook the breadslice in it
            if interact_instance.object_type == "Toaster":
                for iid in interact_instance.state.receptacleObjectIds():
                    instance = self.get_instance(iid)
                    if instance is not None and "toastable" in instance.affordances:

                        if verbose:
                            self._print("Toggle on a toaster will cook the breadslice in it")
                            self._print("%s %s "%(str(instance), "isCooked"))

                        instance.state["isCooked"].set_value(True)
        

        elif action_type == 'ToggleOff':

            if verbose:
                self._print("%s %s"%(str(interact_instance), "isToggled"))

            interact_instance.state["isToggled"].set_value(False)

            if (
                interact_instance.object_type == "StoveKnob"
                and interact_instance.instance_id in self.opened_stove_knob_ids
             ):
                self.opened_stove_knob_ids.remove(interact_instance.instance_id)

        elif action_type == 'Slice':

            if verbose:
                self._print("%s %s"%(str(interact_instance), "isSliced"))

            interact_instance.state["isSliced"].set_value(True)

            # Set the sliced instance id as the "sliceParent" of new spawned XXXsliced instances  
            interact_instance_type = interact_instance.object_type
            for new_iid in self.new_instance_ids:
                if interact_instance_type in new_iid and 'Sliced' in new_iid:
                    sliced_instance = self.get_instance(new_iid)
                    if sliced_instance is not None:
                        sliced_instance.state["sliceParent"].set_value(interact_instance.instance_id)

            # TODO: if something is sliced during cooking, toggle the isCooked state
            #       of spawned slice instances
        
        elif action_type == 'Pour':
            liquid = None
            if self.inventory_instance_ids:
                holding = self.get_instance(self.inventory_instance_ids[0])
                print("holding", holding)
                if holding is not None:
                    liquid = "coffee" if holding.state.simbotIsFilledWithCoffee() else liquid
                    liquid = "water" if holding.state.simbotIsFilledWithWater() else liquid

                    if verbose:
                        self._print("%s %s %s %s"%("Holding:", str(holding), "isFilledWithLiquid", liquid))

                    print("%s %s %s %s"%("Holding:", str(holding), "isFilledWithLiquid", liquid))

                    holding.state["isFilledWithLiquid"].set_value(False)
                    if liquid == "coffee":
                        holding.state["simbotIsFilledWithCoffee"].set_value(False)
                    elif liquid == "water":
                        holding.state["simbotIsFilledWithWater"].set_value(False)
                        print("%r"%holding)
            else:
                self._print("Successfully pouring without holding an item? something went wrong")
                return

            if 'canFillWithLiquid' in interact_instance.affordances:
                if liquid == "coffee":

                    if verbose:
                        self._print("%s %s %s %s"%("Pour to:", str(interact_instance), "isFilledWithLiquid", liquid))

                    interact_instance.state["isFilledWithLiquid"].set_value(True)
                    interact_instance.state["simbotIsFilledWithCoffee"].set_value(True)
                elif liquid == "water":

                    if verbose:
                        self._print("%s %s %s %s"%("Pour to:", str(interact_instance), "isFilledWithLiquid", liquid))
                    
                    interact_instance.state["isFilledWithLiquid"].set_value(True)
                    interact_instance.state["simbotIsFilledWithWater"].set_value(True)
        
        return


    def register_new_instance(self, new_instance, verbose=False) -> str:
        object_type = new_instance.object_type
        for intid in range(0, 999):
            if intid not in self.instance_id_register[object_type]:
                self.instance_id_register[object_type].add(intid)
                break
        
        instance_id_global = f"{object_type}_{intid}"
        new_instance_to_add = ObjectInstance(
            instance_id=instance_id_global,
            object_type=object_type,
            voxel_mask=new_instance.mask_3d_in_voxel,
            state=create_default_object_state())

        self.all_object_instances[instance_id_global] = new_instance_to_add
        self.instance_mapping_2to3[new_instance.tmp_unique_id] = instance_id_global
        
        self.instance_match_log_info["new"] += " + %s\n"%str(new_instance_to_add)
        self.new_instance_ids.append(instance_id_global)

        return instance_id_global
            

    def delete_instance(self, instance_to_delete, verbose=False):
        self.instance_match_log_info["delete"] += " - %r\n"%instance_to_delete
        
        if isinstance(instance_to_delete, str):
            instance_to_delete = self.get_instance(instance_to_delete)

        instance_id = instance_to_delete.instance_id
        intid = int(instance_id.split("_")[-1])
        object_type = instance_to_delete.object_type
        self.instance_id_register[object_type].remove(intid)
        del self.all_object_instances[instance_id]

    def get_symbolic_state_dict(self):
        return self.symbolic_to_dict(self.get_all_instances())

    @staticmethod
    def symbolic_to_dict(all_instances):
        state_dict = {}
        exist_types = set()
        for instance in all_instances:
            iid = instance.instance_id
            state_dict[iid] = {
                'objectId': iid,
                'objectType': instance.object_type,
                'centroid': instance.centroid_3d.tolist()
            }
            for state_name in instance.state:
                state_dict[iid][state_name] = instance.state[state_name]()
            exist_types.add(instance.object_type)

        return state_dict