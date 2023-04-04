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


class SubgoalSkillCoffee(NavigationSkill):
    def __init__(
        self,
        logger=None,
    ):
        super().__init__(logger)
        self.get_skill = SubgoalSkillGet(logger=logger)
        self.clean_skill = SubgoalSkillClean(logger=logger)
        self.place_skill = SubgoalSkillPlace(logger=logger)
        self.get_mug_sg = TeachSubgoal(
            predicate="isPickUp",
            subject_constraint={"objectType": "Mug"}
        )
        self.clean_mug_sg = TeachSubgoal(
            predicate="isClean",
            subject_constraint={"objectType": "Mug"}
        )
        self.place_mug_sg = TeachSubgoal(
            predicate="parentReceptacles",
            subject_constraint={"objectType": "Mug"},
            object_constraint={"objectType": "CoffeeMachine"},
        )
        

        # below should appear in reset
        self.STATE = "INIT"  # INIT, PLACE_TO_SINK, TOGGLE, SUCCESS, FAILED
        self.log_func("SubgoalSkillCoffee is initialized!")
        self.failure_num = 0

    def reset(self):
        super().reset()
        self.get_skill.reset()
        self.clean_skill.reset()
        self.place_skill.reset()

        self.STATE = "INIT"
        self.log_func("SubgoalSkillCoffee is reset!")
        self.failure_num = 0
    


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

                for instance in state_repr.symbolic_world_repr.get_interactable_instances():
                    recep_ids = instance.state["parentReceptacles"].get_values()
                    for recep in recep_ids:
                        if "CoffeeMachine" in recep: 
                            self.STATE = 'TOGGLE'

                if self.STATE == "TOGGLE":
                    continue
                self.STATE = 'GET_MUG'
                
            elif self.STATE == "GET_MUG":
                action = self.get_skill.step(self.get_mug_sg, state_repr)
                if action.is_stop():
                    for instance_id in state_repr.inventory_instance_ids:
                        instance = state_repr.get_instance_by_id(instance_id)
                        if 'isDirty' in instance.state and instance.state['isDirty'].get_value() == False:
                            self.STATE = 'PLACE'
                            break
                
                    if self.STATE == "PLACE":
                        continue
                    else:
                        self.STATE = "CLEAN_MUG"
                        continue
                
                elif action.is_failed():
                    self.STATE = "FAILED"
                    continue
                else:
                    return action
            
            elif self.STATE == "CLEAN_MUG":
                action = self.clean_skill.step(self.clean_mug_sg, state_repr)
                if action.is_stop():
                    self.STATE = "PLACE"
                    continue
                elif action.is_failed():
                    self.STATE = "FAILED"
                    continue
                else:
                    return action
            
            elif self.STATE == "PLACE":
                action = self.place_skill.step(self.place_mug_sg, state_repr)
                if action.is_stop():
                    self.STATE = "TOGGLE"
                    continue
                elif action.is_failed():
                    self.STATE = "FAILED"
                    continue
                else:
                    return action
            
            elif self.STATE == "TOGGLE":

                if state_repr.last_action.action_type == "ToggleOn":
                    if not state_repr.last_action_failed:
                        subgoal.assign_goal_instance_id(self.target_instance.instance_id)
                        self.STATE = "SUCCUSS"
                        for detection in state_repr.observation.object_detections:
                            if detection.object_type == "CoffeeMachine":
                                target_instance_id = state_repr.get_3D_instance_id_of_detection(detection)
                                return TeachAction.create_action_with_instance(
                                    action_type="ToggleOff",
                                    instance_id_3d=target_instance_id,
                                    detection=detection,
                                )
                        continue
                    else:
                        self.failure_num += 1 
                        if self.failure_num > 5:
                            self.log_func("Too many slice failures!")
                            self.STATE = "FAILED"
                            continue
                        else:
                            self.log_func("Failed to toggle the target. Retry!")
                            action_name = random.choice(["Pan Left", "Pan Right"])
                            return TeachAction(action_name) # break the loop
                
                for detection in state_repr.observation.object_detections:
                    if detection.object_type == "CoffeeMachine":
                        target_instance_id = state_repr.get_3D_instance_id_of_detection(detection)
                        self.target_instance = state_repr.get_instance_by_id(target_instance_id)
                        return TeachAction.create_action_with_instance(
                            action_type="ToggleOn",
                            instance_id_3d=target_instance_id,
                            detection=detection,
                        )
                
                self.log_func("Failed to detect the target. Subgoal failed!")
                self.STATE = "FAILED"
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