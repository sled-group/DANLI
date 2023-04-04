from typing import Dict, List, Tuple, Union, Optional
import math, copy
import random

from definitions.teach_object_semantic_class import SemanticClass, check_obj_X_is_Y

from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState

from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.models.teach.skills.navigation import NavigationSkill
from sledmap.mapper.models.teach.skills.subgoal_skill_get import SubgoalSkillGet
from sledmap.mapper.models.teach.skills.subgoal_skill_place import SubgoalSkillPlace
from sledmap.mapper.models.teach.skills.subgoal_skill_clean import SubgoalSkillClean

import sledmap.mapper.models.teach.utils as utils


class SubgoalSkillCook(NavigationSkill):
    def __init__(
        self,
        logger=None,
    ):
        super().__init__(logger)
        self.place_skill = SubgoalSkillPlace(logger=logger)

        # below should appear in reset
        self.STATE = "INIT"  # INIT, PLACE_TO_SINK, TOGGLE_WATER, SUCCESS, FAILED
        self.log_func("SubgoalSkillCook is initialized!")
        self.failure_num = 0
        self.place_to_cooker_sg = None
        self.place_to_toaster_sg = None
        self.toggle_times = 0
        self.target_type = None

    def reset(self):
        super().reset()
        self.place_skill.reset()

        self.STATE = "INIT"
        self.log_func("SubgoalSkillCook is reset!")
        self.failure_num = 0
        self.place_to_cooker_sg = None
        self.place_to_toaster_sg = None
        self.toggle_times = 0
        self.target_type = None

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
                self.target_type = (
                    subgoal.subject_constraint["semanticClass"]
                    if "semanticClass" in subgoal.subject_constraint
                    else subgoal.subject_constraint["objectType"]
                )
                
                if 'BreadSliced' in self.target_type:
                    self.STATE = "COOK_BREAD"
                elif "Potato" in self.target_type:
                    self.STATE = "COOK_POTATO"
                else:
                    self.STATE = 'FAILED' # not implemented
                continue
            
            elif self.STATE == "COOK_BREAD":
                self.place_to_toaster_sg = TeachSubgoal(
                    predicate="parentReceptacles",
                    subject_constraint={"objectType": self.target_type},
                    object_constraint={"objectType": "Toaster"},
                )
                action = self.place_skill.step(self.place_to_toaster_sg, state_repr)
                if action.is_stop():
                    self.STATE = "TOGGLE"
                    continue
                elif action.is_failed():
                    self.STATE = "FAILED"
                    continue
                else:
                    return action
            
            elif self.STATE == "COOK_POTATO":
                all_instances = state_repr.symbolic_world_repr.get_all_instances()
                all_instances = sorted(all_instances, key=lambda x: x.state['distance'].get_value())
                for instance in all_instances:
                    if instance.object_type in ['Pan', 'Pot', 'Microwave']:
                        self.place_to_cooker_sg = TeachSubgoal(
                            predicate="parentReceptacles",
                            subject_constraint={"objectType": self.target_type},
                            object_constraint={"objectType": instance.object_type},
                        )
                
                if self.place_to_cooker_sg is not None:
                    action = self.place_skill.step(self.place_to_cooker_sg, state_repr)
                    if action.is_stop():
                        self.STATE = "TOGGLE"
                        continue
                    elif action.is_failed():
                        self.STATE = "FAILED"
                        continue
                    else:
                        return action
                else:
                    self.STATE = "FAILED" 
                    continue
            
            elif self.STATE == "TOGGLE":
                interactables = state_repr.symbolic_world_repr.get_interactable_instances()
                for instance in interactables:
                    if instance.object_type == 'Microwave' and instance.state['isOpen'].get_value():
                        self.failure_num += 1 
                        if self.failure_num > 5:
                            self.log_func("Cannot close the microwave!")
                            self.STATE = "FAILED"
                            continue
                        else:
                            target_instance_id = instance.instance_id
                            detection = state_repr.get_2D_detection_of_instance(target_instance_id)
                            return TeachAction.create_action_with_instance(
                                        action_type="Close",
                                        instance_id_3d=instance.instance_id,
                                        detection=detection,
                                    )

                    if instance.object_type in ['Toaster', 'Microwave', 'StoveKnob']:
                        if not instance.state['isToggled'].get_value():
                            target_instance_id = instance.instance_id
                            detection = state_repr.get_2D_detection_of_instance(target_instance_id)
                            self.toggle_times += 1
                            if self.toggle_times > 10:
                                self.STATE = "FAILED"
                                continue
                            else:
                                return TeachAction.create_action_with_instance(
                                    action_type="ToggleOn",
                                    instance_id_3d=target_instance_id,
                                    detection=detection,
                                )
                self.STATE = 'FAILED'
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