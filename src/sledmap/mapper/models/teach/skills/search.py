from typing import List, Dict
import copy
import random

from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState
from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.env.teach.teach_object_instance import ObjectInstance

from sledmap.mapper.models.teach.skills.explore_map_skill import ExploreMapSkill
from sledmap.mapper.models.teach.skills.go_for_object_instance import GoForObjectInstanceSkill
from definitions.teach_objects import AFFORDANCE_TO_OBJECTS

from sledmap.mapper.utils.base_cls import Skill

_OPENABLE_RECEPTACLES = AFFORDANCE_TO_OBJECTS['openable']

class SearchSkill(Skill):
    """
    The Search skill takes a list of object_type's of interest as input, and generates actions to
     explore around the map to try to find new instance(s) of those types.
     Once a new instance is found, this skill then generates actions to go to the new instance candidate
     and return stop when it reaches the candidate.
     It also optionally takes as input a list of object instances to use their centroids as way points to conduct
     the explore. If the hint list is exhausted and there is still no candidate showing up, the Search skill then
     turn into frontier-based exploration with a max step limit: max_explore_step.
     If all of the above effort has been exhausted and there is still not any candidate,
     the skill enter a "FAILED" state and will return 'Stop' action thereafter.
     If the skill successfully find and navigates to a candidate,
     it will end up in a "FINISH" state and return 'Stop' action thereafter.
     Read more about the self.state below in the code.
    """

    def __init__(self, max_explore_step: int = 999):
        super().__init__()
        self.max_explore_step = max_explore_step
        
        # below need to appear in set_goal
        self.object_instance_hint_list = []
        # these are used to track what are the existing objects when the search just began,
        # so later we can use these to check if we have found a new instance of target type
        self.initial_object_instance_list = []
        self.type_of_interest = set()

        self.state = "INIT"  # for this skill there are these states: INIT, USE_HINT, EXPLORE, FINAL_GO_FOR, FINISH, FAILED.
        self.act_count = 0
        self.finished_hint_count = 0
        self.explore_step_num = 0
        self.current_go_for_object_instance_skill = None # when this is None, it means we will need to use a new hint from the hint list
        self.explore_map_skill = None
        self.final_go_for_object_instance_skill = None
        self.target_object_instance_for_final_go_for = None
        self.just_reset_explore_goal = False

    def set_goal(self, type_of_interest, exist_instance_ids_of_intereted_type, object_instance_hints=None):
        # these are used to track what are the existing objects when the search just began,
        # so later we can use these to check if we have found a new instance of target type
        self.reset()
        self.type_of_interest = type_of_interest
        self.initial_object_instance_ids = copy.deepcopy(exist_instance_ids_of_intereted_type)
        self.object_instance_hint_list = [] if object_instance_hints is None else object_instance_hints

    def reset(self):
        self.initial_object_instance_ids = []
        self.object_instance_hint_list = []
        self.type_of_interest = set()

        self.state = "INIT"  # for this skill there are these states: INIT, USE_HINT, EXPLORE, FINAL_GO_FOR, FINISH, FAILED.
        self.act_count = 0
        self.finished_hint_count = 0
        self.explore_step_num = 0
        self.current_go_for_object_instance_skill = None # when this is None, it means we will need to use a new hint from the hint list
        self.explore_map_skill = None
        self.final_go_for_object_instance_skill = None
        self.target_object_instance_for_final_go_for = None
        self.just_reset_explore_goal = False

    def status_summary(self, agent_pos=None) -> str:
        summary = "Searching for: %r\n"%(self.type_of_interest)
        summary += " - Search State: %s"%(self.state)
        if self.state == 'USE_HINT':
            summary += " [Hints: %r -> work on: %d]"%(self.object_instance_hint_list, self.finished_hint_count)
            if self.current_go_for_object_instance_skill is not None:
                summary += "\n - " + self.current_go_for_object_instance_skill.status_summary(agent_pos)
            else:
                summary += "\n - Just reached a hint and try to open it"
        elif self.state == 'EXPLORE':
            if self.explore_map_skill is not None:
                summary += "\n - " + self.explore_map_skill.status_summary(agent_pos)
            else:
                summary += "\n - No exploration goal. Rotate to observe more."
        elif self.state == 'FINAL_GO_FOR':
            if self.final_go_for_object_instance_skill is not None:
                summary += "\n - " + self.final_go_for_object_instance_skill.status_summary(agent_pos)
            else:
                summary += "\n - No final go for target? Something went wrong."
        return summary

    def act(self, neural_symbolic_state: NeuralSymbolicAgentState) -> TeachAction:
        assert neural_symbolic_state.spatial_state_repr.data.data.shape[0] == 1  # currently only batch_size=1 is suppoorted
        self.act_count += 1

        instances_of_intereted_type = []
        for instance_id in neural_symbolic_state.get_all_instance_ids():
            instance = neural_symbolic_state.get_instance_by_id(instance_id)
            if instance.object_type in self.type_of_interest:
                instances_of_intereted_type.append(instance)

        print('============> SEARCH: ', self.state)
        # TODO: add open action!!!

        # for each branch in this infinite loop, there is either a 'continue' or a 'return' statement, guaranteed.
        while True:  # only 'return' statement can break out of this loop. This is to make sure we always return some action
            if self.state == 'INIT':
                # the very first time this is called, record all existing object instances
                # currently only batch size = 1 is supported
                # must use deepcopy()!
                self.state = 'USE_HINT'
                continue

            if self.state == 'USE_HINT':
                # check if a new instance of target type has appeared in voxel, if so, stop search immediately
                diff_list_object_instance = get_new_instance_of_interest(instances_of_intereted_type,
                                                                         self.initial_object_instance_list)

                print(self.finished_hint_count, self.object_instance_hint_list)

                if len(diff_list_object_instance) > 0:  # have some new instances, then pick one as target to GoFor
                    self.target_object_instance_for_final_go_for = diff_list_object_instance[0]  # TODO: sort by distance to agent and pick the closeset one as target
                    self.final_go_for_object_instance_skill = GoForObjectInstanceSkill(self.target_object_instance_for_final_go_for)
                    self.state = 'FINAL_GO_FOR'
                    continue

                if self.finished_hint_count >= len(self.object_instance_hint_list):
                    # have exhausted all hints
                    self.state = 'EXPLORE'
                    continue
                else:  # there is a valid hint to use
                    # if we just started using a new instance from the hint, initialize the go_for skill.
                    # This happens every first time to use a new hint from the hint list
                    if self.current_go_for_object_instance_skill is None:

                        current_object_instance_id = self.object_instance_hint_list[self.finished_hint_count]
                        current_object_instance = neural_symbolic_state.get_instance_by_id(current_object_instance_id)


                        # print('=====> HINT ID:', self.finished_hint_count)

                        if current_object_instance is None:
                            self.finished_hint_count += 1
                            self.state = 'USE_HINT'
                            continue
                    
                        # print('=====> GO FOR HINT:', current_object_instance)
                        
                        self.current_go_for_object_instance_skill = GoForObjectInstanceSkill(current_object_instance)

                    
                    # go_for has already been initialized at this point, now just act
                    action = self.current_go_for_object_instance_skill.act(neural_symbolic_state)
                    self.state = 'USE_HINT'  # next state should still be USE_HINT

                    if action.is_stop():
                        self.current_go_for_object_instance_skill = None

                        # if the object is an openable one, open it
                        current_object_instance_id = self.object_instance_hint_list[self.finished_hint_count]
                        current_object_instance = neural_symbolic_state.get_instance_by_id(current_object_instance_id)

                        self.finished_hint_count += 1
                        self.state = 'USE_HINT'
                        if current_object_instance is not None and current_object_instance.object_type in _OPENABLE_RECEPTACLES:
                            detection = neural_symbolic_state.get_2D_detection_of_instance(current_object_instance.instance_id)
                            if detection is not None: 
                                return TeachAction.create_action_with_instance(action_type='Open',
                                                                            instance_id_3d=current_object_instance.instance_id,
                                                                            detection=detection)
                        continue
                    else:
                        return action

            elif self.state == 'EXPLORE':
                # check if a new instance of target type has appeared in voxel, if so, stop search immediately
                diff_list_object_instance = get_new_instance_of_interest(instances_of_intereted_type,
                                                                         self.initial_object_instance_list)
                if len(diff_list_object_instance) > 0:  # have some new instances, then pick one as target to GoFor
                    self.target_object_instance_for_final_go_for = diff_list_object_instance[0]
                    self.final_go_for_object_instance_skill = GoForObjectInstanceSkill(self.target_object_instance_for_final_go_for)
                    self.state = 'FINAL_GO_FOR'
                    self.explore_step_num = 0
                    continue

                if self.explore_map_skill is None:
                    # self.explore_map_skill = ExploreMapSkill(max_nav_count=self.max_explore_step,
                    #                                          simple_explore_baseline=False)
                    # randomly select an instance far away from the agent as the explore target
                    all_instances = neural_symbolic_state.get_all_instances()
                    candidates = [i for i in all_instances if i.voxel_count >=5 and not i.state.interactable()]
                    sorted_farest_first = sorted(candidates, key=lambda x: -x.state.distance())
                    if len(sorted_farest_first) <= 3:
                        # return TeachAction(random.choice(["Turn Left", "Turn Right"]))
                        self.explore_step_num += 1 
                        return TeachAction("Turn Left")
                    target_instance = random.choice(sorted_farest_first[:10])
                    self.explore_map_skill = GoForObjectInstanceSkill(target_instance)
                    self.just_reset_explore_goal = True

                action = self.explore_map_skill.act(neural_symbolic_state)

                if not action.is_stop() and self.explore_step_num < 30:
                    self.just_reset_explore_goal = False
                    self.explore_step_num += 1 
                    self.state = 'EXPLORE'  # next state should still be EXPLORE
                    return action
                elif not self.just_reset_explore_goal:
                    self.explore_map_skill = None  # reset a new explore goal
                    self.explore_step_num = 0
                    self.state = 'EXPLORE'
                    continue
                else:
                    self.state = 'FAILED'
                    continue

            elif self.state == 'FINAL_GO_FOR':
                assert self.final_go_for_object_instance_skill is not None

                action = self.final_go_for_object_instance_skill.act(neural_symbolic_state)
                if action.is_stop():
                    self.state = 'FINISH'
                    continue
                else:
                    self.state = 'FINAL_GO_FOR'
                    return action

            elif self.state == 'FINISH':
                self.state = 'FINISH'
                return TeachAction.stop_action()

            elif self.state == 'FAILED':
                self.state = 'FAILED'
                return TeachAction.fail_action()

            else:
                raise Exception('search:act(): Invalid state!')

    def has_failed(self) -> bool:
        pass

    def get_trace(self, device="cpu") -> Dict:
        pass

def get_new_instance_of_interest(new_instances_list: List[ObjectInstance],
                                 old_instance_id_list: List[str]) -> List[ObjectInstance]:
    print("old_instance_id_list", old_instance_id_list)
    diff_list_object_instance = []
    for new_instance in new_instances_list:
        if new_instance.instance_id not in old_instance_id_list:
            diff_list_object_instance.append(new_instance)
    diff_list_object_instance.sort(key=lambda x: x.state['distance'].get_value())
    print("diff_list_object_instance", diff_list_object_instance)
    return diff_list_object_instance


def calculate_diff_of_object_instance_list(new_object_instance_list: Dict,
                                           old_object_instance_list: Dict,
                                           object_types_of_interest: List[str]) -> List[ObjectInstance]:
    diff_list_object_instance = []
    for obj_type in object_types_of_interest:
        old_count = len(old_object_instance_list[obj_type])
        new_count = len(new_object_instance_list[obj_type])
        if new_count > old_count:
            diff_list_object_instance += new_object_instance_list[obj_type][old_count:]

    return diff_list_object_instance
