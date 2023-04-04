from typing import Dict, List, Tuple, Union, Optional
import math
import random

from sledmap.mapper.models.teach.skills.navigation import NavigationSkill
from sledmap.mapper.env.teach.teach_action import TeachAction
from definitions.teach_objects import get_object_receptacle_compatibility

from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState


# get new or get closest that meet constraint
# new: (1) observed but simbotIsPickedUp is False or (2) Find not observed one
# existing: just need to meet the constraint

# when to get new?


class SubgoalSkillGet(NavigationSkill):
    def __init__(
        self,
        logger=None,
    ):
        super().__init__(logger)

        # below should appear in reset
        self.STATE = "INIT"  # INIT, SOLVING, SUCCESS, FAILED
        self.log_func("SubgoalSkillGet is initialized!")
        self.failure_nums = 0
        self.num_loop = 0
        self.try_pick_num = 0

    def reset(self):
        super().reset()

        self.STATE = "INIT"
        self.log_func("SubgoalSkillGet is reset!")
        self.failure_nums = 0
        self.num_loop = 0
        self.try_pick_num = 0

    def check_progress(
        self, subgoal: TeachSubgoal, state_repr: NeuralSymbolicAgentState
    ) -> Tuple[Union[str, None], bool]:
        """
        If there is any target instance meet the constraint,
            - If there is one in hand, return the instance and the True indicating it is in hand
            - Else return the closest instance and False
        If there is no target object meet the constraint, return None and False

        :param subgoal: _description_
        :param state_repr: _description_
        :return: instance id and whether it is in hand
        """
        constraint = subgoal.subject_constraint
        exclude_id_list = subgoal.exclude_instance_ids


        candidates = []
        for instance_id in state_repr.get_all_instance_ids():
            if instance_id in exclude_id_list:
                continue
            instance = state_repr.get_instance_by_id(instance_id)
            if TeachSubgoal.check_instance_condition(instance, constraint):
                candidates.append(instance)

        if not candidates:
            return None, False

        for candidate in candidates:
            if candidate.instance_id in state_repr.inventory_instance_ids or candidate.state.isPickedUp():
                return candidate, True

        instance_and_distance = [(i, i.state.distance()) for i in candidates]
        instance_and_distance.sort(key=lambda x: x[1])
        return instance_and_distance[0][0], False


    def step(
        self,
        subgoal: TeachSubgoal,
        state_repr: NeuralSymbolicAgentState,
    ) -> TeachAction:

        target_instance = None
        
        self.num_steps += 1
        for i in range(10):
            # exceed step limits: transit to failed
            if self.num_steps >= self.MAXIMUM_NUM_STEPS:
                self.STATE = "FAILED"
                self.log_func("Failed due to reach step limit")


            if self.STATE == "SOLVING":
                target_instance, is_picked = self.check_progress(subgoal, state_repr)
                self.log_func(f"Progress check: {target_instance} {is_picked}")
                
                # already a target object in hand: transit to success
                if target_instance is not None and is_picked:
                    self.STATE = "SUCCESS"
                    continue
                
                # there is a target instance, but not close enough or picked yet
                if target_instance is not None:
                    interactable = target_instance.state["interactable"].get_value()
                    if not interactable:
                        action = self.navigate_to(target_instance, state_repr)
                        if not action.is_stop():
                            # still navigating
                            return action
                    
                    # deal with inventory tracking errors
                    if self.try_pick_num <= 1 and not state_repr.last_action_failed:
                        target_instance_id = target_instance.instance_id
                        detection = state_repr.get_2D_detection_of_instance(
                            target_instance_id
                        )
                        self.log_func("Trying to pickup: %r"%detection)
                        if detection is not None:
                            return TeachAction.create_action_with_instance(
                                action_type="Pickup",
                                instance_id_3d=target_instance_id,
                                detection=detection,
                            )
                    self.try_pick_num += 1
                    
                    # until here the agent believes that the target has been reached
                    

                    # there is an object in hand, need to place it first
                    inventory_ids = state_repr.inventory_instance_ids
                    if inventory_ids:

                        if self.failure_nums > 6:
                            self.log_func("Failed due to too many failures")
                            self.STATE = "FAILED"
                            continue
                        
                        inventory_type = inventory_ids[0].split("_")[0]
                        valid_receps = get_object_receptacle_compatibility(inventory_type)

                        detections = state_repr.observation.object_detections
                        sorted_by_area = sorted(range(len(detections)), key=lambda k: detections[k].area, reverse=True)
                        
                        last_action = state_repr.last_action
                        for idx in sorted_by_area:
                            detection = detections[idx]
                            if detection.object_type in valid_receps:
                                instance_id = state_repr.get_3D_instance_id_of_detection(detection)
                                if (
                                    last_action.action_type == "Place"
                                    and state_repr.last_action_failed 
                                    and last_action.instance_id == instance_id
                                ):
                                    continue
                                self.failure_nums += 1
                                self.log_func("Hand not empty! Trying to first place the inventory.")
                                return TeachAction.create_action_with_instance(
                                    action_type="Place",
                                    instance_id_3d=instance_id,
                                    detection=detection,
                                )
                        
                        self.log_func("Failed to make hand for picking")
                        self.STATE = "FAILED"
                        continue

                    
                    if state_repr.last_action.action_type == "Pickup" and state_repr.last_action_failed:
                        self.failure_nums += 1
                        if self.failure_nums > 5:
                            self.log_func("Failed due to too many failures")
                            self.STATE = "FAILED"
                            continue 
                        action = random.choice(["Forward", "Pan Right", "Pan Left"])
                        return TeachAction(action_type=action)
                    
                    # no object in hand, ready to pick up
                    target_instance_id = target_instance.instance_id
                    detection = state_repr.get_2D_detection_of_instance(
                        target_instance_id
                    )
                    if detection is not None:
                        self.log_func("Target instance is in view. Trying to pickup: %r"%detection)
                        return TeachAction.create_action_with_instance(
                            action_type="Pickup",
                            instance_id_3d=target_instance_id,
                            detection=detection,
                        )
                    else:
                        for detection in state_repr.observation.object_detections:
                            if detection.object_type == target_instance.object_type:
                                target_instance_id = state_repr.get_3D_instance_id_of_detection(detection)
                                self.log_func("Target instance not in view. Trying to pick up"
                                              " another instance with the same type: %r"%detection)
                                return TeachAction.create_action_with_instance(
                                    action_type="Pickup",
                                    instance_id_3d=target_instance_id,
                                    detection=detection,
                                )

                        self.STATE = "FAILED"
                        self.log_func("Failed due to no valid object to pick up")
                        continue
                
                # did not see any target instance, have to search
                else:
                    action = self.search_target(subgoal.get_goal_object_types(), state_repr)
                    if action.is_stop():
                        self.STATE = "SOLVING"
                        continue
                    elif action.is_failed():
                        self.STATE = "FAILED"
                        continue
                    return action
                    
            elif self.STATE == "INIT":
                self.STATE = "SOLVING"
                continue

            elif self.STATE == "FAILED":
                self.log_func("Subgoal failed after %d steps" % self.num_steps)
                # relax constraint
                self.reset()
                return TeachAction.fail_action()

            elif self.STATE == "SUCCESS":
                self.log_func("Subgoal succeed after %d steps" % self.num_steps)
                subgoal.assign_goal_instance_id(target_instance.instance_id)
                self.reset()
                return TeachAction.stop_action()

        self.log_func("Escape from bouncing")
        return TeachAction.fail_action()