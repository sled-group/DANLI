from typing import Dict, List, Tuple, Union, Optional
import math
import copy

from definitions.teach_objects import get_object_receptacle_compatibility

from sledmap.mapper.models.teach.skills.navigation import NavigationSkill
from sledmap.mapper.models.teach.skills.subgoal_skill_get import SubgoalSkillGet
from sledmap.mapper.env.teach.teach_action import TeachAction

from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState
from definitions.teach_object_semantic_class import SemanticClass

OpenableRecepcales = SemanticClass.get_all_objs_in_semcls("OpenableReceptacles")


class SubgoalSkillPlace(NavigationSkill):
    def __init__(
        self,
        logger=None,
    ):
        super().__init__(logger)
        self.get_skill = SubgoalSkillGet(logger=logger)

        # below should appear in reset
        self.place_attemps_num = 0
        self.get_subj_subgoal = None
        self.target_receptacle = None
        self.instance_ids_to_remove = []
        self.remove_instance = None
        self.STATE = "INIT"  # INIT, GET, PLACE, CLEAR_RECEPTACLE, REMOVE_INSTANCE, SUCCESS, FAILED
        self.log_func("SubgoalSkillPlace is initialized!")

    def reset(self):
        super().reset()
        self.get_skill.reset()

        self.place_attemps_num = 0
        self.get_subj_subgoal = None
        self.target_receptacle = None
        self.instance_ids_to_remove = []
        self.remove_instance = None
        self.STATE = "INIT"
        self.log_func("SubgoalSkillPlace is reset!")

    
    def check_place_progress(
        self, subgoal: TeachSubgoal, state_repr: NeuralSymbolicAgentState
    ) -> Tuple[Union[str, None], bool]:
        """
        Return the valid receptacle instance cloeset to the agent. 
        If no valid receptacle is found, return None.

        :param subgoal: _description_
        :param state_repr: _description_
        :return: instance id or None
        """

        constraint = subgoal.object_constraint
        exclude_id_list = subgoal.exclude_instance_ids_obj

        candidates = []
        for instance_id, instance in state_repr.get_all_instances():
            if instance_id in exclude_id_list:
                continue
            if TeachSubgoal.check_instance_condition(instance, constraint):
                candidates.append(instance)

        if not candidates:
            return None

        instance_and_distance = [
            (i, i.state["distance"].get_value()) for i in candidates
        ]
        instance_and_distance.sort(key=lambda x: x[1])
        return instance_and_distance[0][0]
    

    def step(
        self,
        subgoal: TeachSubgoal,
        state_repr: NeuralSymbolicAgentState,
    ) -> TeachAction:

        self.num_steps += 1
        for i in range(10):
            if self.num_steps >= self.MAXIMUM_NUM_STEPS:
                self.STATE = "FAILED"
                self.log_func("Failed due to reach step limit")

            if self.STATE == "INIT":
                self.get_subj_subgoal = self.create_get_subgoal(subgoal, state_repr)
                self.STATE = "GET"
                continue
            
            elif self.STATE == "GET":
                action = self.get_skill.step(self.get_subj_subgoal, state_repr)
                self.log_func("[%s]: %r"%(self.STATE, action))
                if action.is_stop():
                    self.get_skill.reset()
                    self.log_func("Successfully get the subject: %s for the Place subgoal!"%self.get_subj_subgoal.goal_instance_id)
                    self.STATE = "PLACE"
                    continue
                elif action.is_failed():
                    self.log_func("Failed to get the subject for the Place subgoal!")
                    self.STATE = "FAILED"
                    continue
                else:
                    return action
            
            elif self.STATE == "PLACE":
                if state_repr.last_action.action_type == "Place" and not state_repr.last_action_failed:
                    subgoal.assign_goal_instance_id(state_repr.last_action.instance_id)
                    self.STATE = "SUCCESS"
                    continue

                target_instance = self.get_closet_valid_instance(
                    subgoal.object_constraint, 
                    state_repr, 
                    exclude_instance_ids=subgoal.exclude_instance_ids_obj
                )

                # have a valid receptacle
                if target_instance is not None:
                    interactable = target_instance.state["interactable"].get_value()
                    target_instance_id = target_instance.instance_id
                    detection = state_repr.get_2D_detection_of_instance(
                        target_instance_id
                    )
                    # ... and it is close to the agent: 
                    if interactable and detection is not None:
                        # try to place
                        if self.place_attemps_num <= 5:
                            self.log_func("Place attempt: {}".format(self.place_attemps_num))
                            self.place_attemps_num += 1
                            if target_instance.object_type in OpenableRecepcales and 'isOpen' in target_instance.state and not target_instance.state['isOpen'].get_value():
                                self.log_func("Try to open: %r"%target_instance)
                                return TeachAction.create_action_with_instance(
                                    action_type="Open",
                                    instance_id_3d=target_instance_id,
                                    detection=detection,
                                )
                            return TeachAction.create_action_with_instance(
                                action_type="Place",
                                instance_id_3d=target_instance_id,
                                detection=detection,
                            )
                        # otherwise, try to clear the receptacle
                        else:
                            self.target_receptacle = target_instance
                            for instance_id in target_instance.state['receptacleObjectIds'].get_values():
                                self.instance_ids_to_remove.append(instance_id)
                            
                            if not self.instance_ids_to_remove:
                                self.log_func("Nothing to clear. Subgoal failed!") 
                                self.STATE = "FAILED"
                            else:
                                self.log_func("Failed to place the object. Try to empty the receptacle.") 
                                self.STATE = "CLEAR_RECEPTACLE"
                            continue
                            
                    else:
                        # try to move closer
                        action = self.navigate_to(target_instance, state_repr)
                        if action.is_stop():
                            # still navigating
                            self.log_func("The instance may not exist (fake ID). Exclude it and retry.")
                            subgoal.exclude_instance_ids_obj.append(target_instance_id)
                            if detection is not None:
                                if self.place_attemps_num <= 5:
                                    self.log_func("Place attempt: {}".format(self.place_attemps_num))
                                    self.place_attemps_num += 1
                                    if target_instance.object_type in OpenableRecepcales and 'isOpen' in target_instance.state and not target_instance.state['isOpen'].get_value():
                                        self.log_func("Try to open: %r"%target_instance)
                                        return TeachAction.create_action_with_instance(
                                            action_type="Open",
                                            instance_id_3d=target_instance_id,
                                            detection=detection,
                                        )
                                    return TeachAction.create_action_with_instance(
                                        action_type="Place",
                                        instance_id_3d=target_instance_id,
                                        detection=detection,
                                    )

                            self.STATE = "FAILED"
                            continue
                        elif action.is_failed():
                            self.log_func("Failed to move closer to the instance. Subgoal failed!")
                            self.STATE = "FAILED"
                            continue
                        else:
                            return action
                # no valid receptacle in currently observed
                else:
                    action = self.search_target(subgoal.get_goal_object_types(), state_repr)
                    if action.is_stop():
                        self.STATE = "PLACE"
                        continue
                    elif action.is_failed():
                        self.STATE = "FAILED"
                        continue
                    return action

            elif self.STATE == "CLEAR_RECEPTACLE":
                
                if len(self.instance_ids_to_remove) == 0:
                    self.log_func("Successfully cleared the receptacle")
                    self.STATE = "GET"
                    continue
                else:
                    remove_instance_id = self.instance_ids_to_remove.pop(0)
                    remove_instance = state_repr.get_instance_by_id(target_instance_id)
                    if remove_instance:
                        self.log_func("Try to remove the instance: %s"%remove_instance_id)
                        self.remove_instance = remove_instance
                        self.STATE = "REMOVE_INSTANCE"
                    else:
                        self.STATE = "CLEAR_RECEPTACLE"
                        continue
            
            elif self.STATE == "REMOVE_INSTANCE":
                if not self.remove_instance.state["isPickedUp"].get_value():
                    detection = state_repr.get_2D_detection_of_instance(
                        self.remove_instance.instance_id
                    )
                    if detection is not None:
                        return TeachAction.create_action_with_instance(
                            action_type="Pickup",
                            instance_id_3d=self.remove_instance.instance_id,
                            detection=detection,
                        )
                else:
                    valid_receps = get_object_receptacle_compatibility(remove_instance.object_type)
                    detections = state_repr.observation.object_detections
                    sorted_by_area = sorted(range(len(detections)), key=detections.area, reverse=True)
                    
                    for idx in sorted_by_area:
                        detection = detections[idx]
                        if detection.object_type in valid_receps:
                            instance_id = state_repr.get_3D_instance_id_of_detection(detection)
                            return TeachAction.create_action_with_instance(
                                action_type="Place",
                                instance_id_3d=instance_id,
                                detection=detection,
                            )
                    # Have to navigate and place ... give up
                    self.STATE = "FAILED"
                    continue
                
                # back to continue CLEAR_RECEPTACLE
                self.STATE = "CLEAR_RECEPTACLE"
                continue
            
            elif self.STATE == "FAILED":
                self.log_func("Subgoal failed after %d steps" % self.num_steps)
                # relax constraint
                self.reset()
                return TeachAction.fail_action()

            elif self.STATE == "SUCCESS":
                self.log_func("Subgoal succeed after %d steps" % self.num_steps)
                self.reset()
                return TeachAction.stop_action()

        self.log_func("Escape from bouncing")
        return TeachAction.fail_action()