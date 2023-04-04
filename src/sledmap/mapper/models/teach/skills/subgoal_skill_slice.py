from typing import Dict, List, Tuple, Union, Optional
import math
import random
import copy

from definitions.teach_object_semantic_class import SemanticClass

from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState

from sledmap.mapper.models.teach.skills.navigation import NavigationSkill
from sledmap.mapper.models.teach.skills.subgoal_skill_get import SubgoalSkillGet
from sledmap.mapper.env.teach.teach_action import TeachAction


KNIVES = SemanticClass.get_all_objs_in_semcls("Knives")

class SubgoalSkillSlice(NavigationSkill):
    def __init__(
        self,
        logger=None,
    ):
        super().__init__(logger)
        self.get_skill = SubgoalSkillGet(logger=logger)
        self.get_knife_sg = TeachSubgoal(
                                predicate="isPickedUp", 
                                subject_constraint={'semanticClass': 'Knives'}
                            )

        # below should appear in reset
        self.STATE = "INIT"  # INIT, GET_KNIFE, GOTO_TARGET, SLICE, SUCCESS, FAILED
        self.log_func("SubgoalSkillSlice is initialized!")
        self.failure_num = 0
        self.knife_in_hand = False
        self.try_num = 0

    def reset(self):
        super().reset()
        self.get_skill.reset()

        self.STATE = "INIT"
        self.failure_num = 0
        self.log_func("SubgoalSkillSlice is reset!")
        self.knife_in_hand = False
        self.try_num = 0


    def step(
        self,
        subgoal: TeachSubgoal,
        state_repr: NeuralSymbolicAgentState,
    ) -> TeachAction:

        self.num_steps += 1

        interactables = state_repr.symbolic_world_repr.get_interactable_instances()
        if self.try_num < 3 and self.knife_in_hand and not state_repr.last_action_failed:
            interactables = state_repr.symbolic_world_repr.get_interactable_instances()
            egg_or_vase = [i for i in interactables if i.object_type in ['Egg', 'Vase']]
            if egg_or_vase:
                target_instance = egg_or_vase[0]
                target_instance_id = target_instance.instance_id
                detection = state_repr.get_2D_detection_of_instance(
                    target_instance_id
                )
                if detection is not None:
                    self.try_num += 1
                    return TeachAction.create_action_with_instance(
                        action_type="Slice",
                        instance_id_3d=target_instance_id,
                        detection=detection,
                    )


        for i in range(10):
            if self.num_steps >= self.MAXIMUM_NUM_STEPS:
                self.STATE = "FAILED"
                self.log_func("Failed due to reach step limit")

            if self.STATE == "INIT":
                inventory_ids = state_repr.inventory_instance_ids
                if inventory_ids and inventory_ids[0].split('_')[0] in KNIVES:
                    self.STATE = "GOTO_TARGET"
                    continue
                else:
                    self.STATE = "GET_KNIFE"
                    continue
            
            elif self.STATE == "GET_KNIFE":
                action = self.get_skill.step(self.get_knife_sg, state_repr)
                if action.is_stop():
                    self.log_func("Successfully get the knife!")
                    self.STATE = "GOTO_TARGET"
                    self.knife_in_hand = True
                    continue
                elif action.is_failed():
                    self.log_func("Did not find any knife in the scene. Failed!")
                    self.STATE = "FAILED"
                    continue
                else:
                    return action


            elif self.STATE == "GOTO_TARGET":
                target_constraint = copy.deepcopy(subgoal.subject_constraint)
                target_instance = self.get_closet_valid_instance(
                    subgoal.subject_constraint, 
                    state_repr, 
                    exclude_instance_ids=subgoal.exclude_instance_ids
                )
                if target_instance is not None:

                    if target_instance.state['interactable'].get_value():
                        self.log_func("Successfully reached the target!")
                        self.target_instance = target_instance
                        self.STATE = "SLICE"
                        continue
                    
                    action = self.navigate_to(target_instance, state_repr)
                    if action.is_stop():
                        self.log_func("Successfully reached the target!")
                        self.target_instance = target_instance
                        self.STATE = "SLICE"
                        continue
                    elif action.is_failed():
                        self.log_func("Failed to reach the target. Subgoal failed!")
                        self.STATE = "FAILED"  # TODO: retry another
                        continue
                    else:
                        return action
                
                else:
                    action = self.search_target(subgoal.get_goal_object_types(), state_repr)
                    if action.is_stop():
                        # can this really happen?
                        self.STATE = "FAILED" # "SLICE"
                        continue
                    elif action.is_failed():
                        self.STATE = "FAILED"
                        continue
                    return action


            elif self.STATE == "SLICE":

                if state_repr.last_action.action_type == "Slice":
                    if not state_repr.last_action_failed:
                        subgoal.assign_goal_instance_id(self.target_instance.instance_id)
                        self.STATE = "SUCCESS"
                        continue
                    else:
                        self.failure_num += 1
                        
                        if self.failure_num > 5:
                            self.log_func("Too many slice failures!")
                            self.STATE = "FAILED"
                            continue
                        else:
                            self.log_func("Failed to slice the target. Retry!")
                            action_name = random.choice(["Forward", "Pan Left", "Pan Right"])
                            return TeachAction(action_name) # break the loop
                
                target_instance_id = self.target_instance.instance_id
                detection = state_repr.get_2D_detection_of_instance(
                    target_instance_id
                )
                if detection is not None:
                    return TeachAction.create_action_with_instance(
                        action_type="Slice",
                        instance_id_3d=target_instance_id,
                        detection=detection,
                    )
                else:
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