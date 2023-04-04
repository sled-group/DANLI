from typing import List, Dict
import copy

from sledmap.mapper.env.alfred.alfred_action import AlfredAction
from sledmap.mapper.models.alfred.handcoded_skills.explore_map_skill import ExploreMapSkill
from sledmap.mapper.models.alfred.handcoded_skills.go_for_object_instance import GoForObjectInstanceSkill
from sledmap.mapper.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
from sledmap.mapper.models.alfred.symbolic.object_instance import ObjectInstance
from sledmap.mapper.utils.base_cls import Skill


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

    def __init__(self,
                 object_types_of_interest: List[str],
                 object_instance_hints: List[ObjectInstance] = None,
                 max_explore_step: int = 999):
        super().__init__()

        assert len(object_types_of_interest) >= 1  # must have at least one type to search for

        self.object_types_of_interest = object_types_of_interest
        self.object_instance_hint_list = object_instance_hints if object_instance_hints is not None else []
        self.max_explore_step = max_explore_step

        # these are used to track what are the existing objects when the search just began,
        # so later we can use these to check if we have found a new instance of target type
        self.initial_object_instance_list = None

        self.state = "INIT"  # for this skill there are these states: INIT, USE_HINT, EXPLORE, FINAL_GO_FOR, FINISH, FAILED.

        self.act_count = 0

        self.finished_hint_count = 0
        self.current_go_for_object_instance_skill = None # when this is None, it means we will need to use a new hint from the hint list

        self.explore_map_skill = None

        self.final_go_for_object_instance_skill = None
        self.target_object_instance_for_final_go_for = None

    def act(self, state_repr: AlfredSpatialStateRepr, new_object_instance_list: Dict) -> AlfredAction:
        assert state_repr.data.data.shape[0] == 1  # currently only batch_size=1 is suppoorted
        self.act_count += 1

        # for each branch in this infinite loop, there is either a 'continue' or a 'return' statement, guaranteed.
        while True:  # only 'return' statement can break out of this loop. This is to make sure we always return some action
            if self.state == 'INIT':
                # the very first time this is called, record all existing object instances
                # currently only batch size = 1 is supported
                # must use deepcopy()!
                self.initial_object_instance_list = copy.deepcopy(new_object_instance_list)

                self.state = 'USE_HINT'
                continue

            if self.state == 'USE_HINT':
                # check if a new instance of target type has appeared in voxel, if so, stop search immediately
                diff_list_object_instance = calculate_diff_of_object_instance_list(new_object_instance_list,
                                                                                   self.initial_object_instance_list,
                                                                                   object_types_of_interest=self.object_types_of_interest)
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
                        self.current_go_for_object_instance_skill = GoForObjectInstanceSkill(
                            self.object_instance_hint_list[self.finished_hint_count])

                    # go_for has already been initialized at this point, now just act
                    action = self.current_go_for_object_instance_skill.act(state_repr)
                    self.state = 'USE_HINT'  # next state should still be USE_HINT

                    if action.is_stop():
                        self.current_go_for_object_instance_skill = None
                        self.finished_hint_count += 1
                        self.state = 'USE_HINT'
                        continue
                    else:
                        return action

            elif self.state == 'EXPLORE':
                # check if a new instance of target type has appeared in voxel, if so, stop search immediately
                diff_list_object_instance = calculate_diff_of_object_instance_list(new_object_instance_list,
                                                                                   self.initial_object_instance_list,
                                                                                   object_types_of_interest=self.object_types_of_interest)
                if len(diff_list_object_instance) > 0:  # have some new instances, then pick one as target to GoFor
                    self.target_object_instance_for_final_go_for = diff_list_object_instance[0]  # TODO: sort by distance to agent and pick the closeset one as target
                    self.final_go_for_object_instance_skill = GoForObjectInstanceSkill(self.target_object_instance_for_final_go_for)
                    self.state = 'FINAL_GO_FOR'
                    continue

                if self.explore_map_skill is None:
                    self.explore_map_skill = ExploreMapSkill(max_nav_count=self.max_explore_step,
                                                             simple_explore_baseline=False)

                action = self.explore_map_skill.act(state_repr)

                if action.is_stop():
                    self.state = 'FAILED'
                    continue
                else:
                    self.state = 'EXPLORE'  # next state should still be EXPLORE
                    return action

            elif self.state == 'FINAL_GO_FOR':
                assert self.final_go_for_object_instance_skill is not None

                action = self.final_go_for_object_instance_skill.act(state_repr)
                if action.is_stop():
                    self.state = 'FINISH'
                    continue
                else:
                    self.state = 'FINAL_GO_FOR'
                    return action

            elif self.state == 'FINISH':
                self.state = 'FINISH'
                return AlfredAction.stop_action()

            elif self.state == 'FAILED':
                self.state = 'FAILED'
                return AlfredAction.fail_action()

            else:
                raise Exception('search:act(): Invalid state!')

    def has_failed(self) -> bool:
        pass

    def set_goal(self, goal):
        pass

    def get_trace(self, device="cpu") -> Dict:
        pass


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
