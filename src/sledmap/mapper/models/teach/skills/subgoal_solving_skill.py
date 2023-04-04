from typing import Union, Tuple, List, Dict, Optional

import copy
from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState

from sledmap.mapper.models.teach.skills.search import SearchSkill
from sledmap.mapper.models.teach.skills.go_for_object_instance import (
    GoForObjectInstanceSkill,
)

from sledmap.mapper.env.teach.teach_action import TeachAction

class SubgoalSovlingSkill:
    MAXIMUM_NUM_STEPS = 50
    

    def __init__(self, logger=None):
        self.logger = logger
        self.log_func = logger.info
        self.num_steps = 0
        self.STATE = "INIT"
        
        self.target_instance = None 
        self.goto_instance_skill = GoForObjectInstanceSkill()
        self.search_skill = SearchSkill()
        self.is_searching = False
        

    def reset(self):
        self.target_instance = None 
        self.goto_instance_skill.reset()
        self.search_skill.reset()
        self.is_searching = False
        
        self.num_steps = 0
        self.STATE = "INIT"

    def set_goal(self,):
        pass

    @classmethod
    def get_closet_valid_instance(
        self, constraint: dict, state_repr: NeuralSymbolicAgentState, exclude_instance_ids: List[str] = []
    ) -> Tuple[Union[str, None], bool]:
        """
        Return the valid instance cloeset to the agent. 
        If no valid instance is found, return None.

        :param subgoal: _description_
        :param state_repr: _description_
        :return: instance id or None
        """

        candidates = []
        for instance_id in state_repr.get_all_instance_ids():
            if instance_id in exclude_instance_ids:
                continue
            instance = state_repr.get_instance_by_id(instance_id)
            if TeachSubgoal.check_instance_condition(instance, constraint):
                candidates.append(instance)

        if not candidates:
            return None

        instance_and_distance = [
            (i, i.state["distance"].get_value()) for i in candidates
        ]
        instance_and_distance.sort(key=lambda x: x[1])
        return instance_and_distance[0][0]

    def navigate_to(self, target_instance, state_repr: NeuralSymbolicAgentState):
        # try to move closer
        # if self.target_instance is None:

        interactable = target_instance.state.interactable()
        if interactable:
            self.log_func("Already close to the target instance: %r" % target_instance)
            self.target_instance = None
            self.goto_instance_skill.reset()
            return TeachAction.stop_action()

        self.log_func(" =====> Set navigation goal: %r" % target_instance)
        self.target_instance = target_instance
        self.goto_instance_skill.set_goal(target_instance)
        
        # self.log_func(" =====> Agent position: %r" % state_repr.spatial_state_repr.get_agent_pos_m())
        # self.log_func(" =====> is navigating to: %r" % target_instance)
        
        action = self.goto_instance_skill.act(state_repr)

        # #####################    DEBUG    #################################
        # action_name = self.oracle_navigator.step(
        #     [target_instance.object_type]
        # )
        # action = TeachAction(action_name)
        # #####################    DEBUG    #################################


        if action.is_stop() or action.is_failed():
            self.log_func("Navigation end!")
            self.target_instance = None
            self.goto_instance_skill.reset()

        return action 


    def search_target(self, subgoal: TeachSubgoal, state_repr: NeuralSymbolicAgentState):
        self.log_func("Searching for %r:"%subgoal)
        # return TeachAction.fail_action()

        instances_of_interest = []
        instance_ids_of_interest = []
        for instance_id in state_repr.get_all_instance_ids():
            instance = state_repr.get_instance_by_id(instance_id)
            if instance.object_type in subgoal.get_goal_object_types():
                instances_of_interest.append(instance)
                instance_ids_of_interest.append(instance_id)

        if not self.is_searching:
            self.log_func("Initialize a search process")
            hints = state_repr.symbolic_world_repr.get_hint_list()
            print('=====================> search hints!')
            print('\n'.join(hints))
            self.search_skill.set_goal(
                instance_ids_of_interest, hints  # TODO: add hint list
            )
            self.is_searching = True

        self.log_func(" =====> is searching for: %r" % subgoal)

        action = self.search_skill.act(
            state_repr,
            instances_of_interest
        )
        self.log_func("Get a search action: %r" % action)
        
        # #####################    DEBUG    #################################
        # target = (
        #     SemanticClass.get_all_objs_in_semcls(
        #         subgoal.subject_constraint["semanticClass"]
        #     )
        #     if "semanticClass" in subgoal.subject_constraint
        #     else [subgoal.subject_constraint["objectType"]]
        # )
        # action_name = self.oracle_navigator.step(target)
        # action = TeachAction(action_name)
        # #####################    DEBUG    #################################
        
        interactable = False
        target_instance = self.search_skill.target_object_instance_for_final_go_for
        if target_instance is not None:
            interactable = target_instance.state["interactable"].get_value()

        if interactable or action.is_stop() or action.is_failed():
            self.log_func("Search ends!")
            self.is_searching = False
            self.search_skill.reset()
            
            if action.is_failed():
                self.log_func("Do not find any instance for: %r" % subgoal)
            if interactable:
                self.log_func("Goal instance [%r] interactable based on distance estimation" % target_instance)
                return TeachAction.stop_action()
        
        return action


    def create_get_subgoal(self, subgoal: TeachSubgoal, state_repr: NeuralSymbolicAgentState):
        """
        Generate a SubgoalSkillGet whose goal is to find a valid subject for the Place subgoal
        """
        subj_constraint = subgoal.subject_constraint


        # exclude instances that already on the object receptacle
        exclude_instance_ids = []
        for instance_id in state_repr.get_all_instance_ids():
            instance = state_repr.get_instance_by_id(instance_id)
            if TeachSubgoal.check_goal_instance(subgoal, instance, state_repr.get_all_instances()):
                exclude_instance_ids.append(instance_id)

        # for instance_id, instance in all_instances.items():
        #     instance_type = instance.object_type
        #     instance_receps = instance.state['parentReceptacles'].get_values()
        #     if instance_type in str(instance.state['isPlacedOn'].get_value()):
        #         exclude_instance_ids.append(instance_id)
        #         continue
        #     if any([check_obj_X_is_Y(i.split("_")[0], obj_type) for i in instance_receps]):
        #         exclude_instance_ids.append(instance_id)

        return TeachSubgoal(
            predicate="isPickedUp",
            subject_constraint=copy.deepcopy(subj_constraint),
            exclude_instance_ids=exclude_instance_ids,
        )