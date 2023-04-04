from sledmap.mapper.utils.base_cls import Task, Agent, Action, Observation, ObservationFunction, StateRepr

from sledmap.mapper.models.alfred.handcoded_skills.explore_map_skill import ExploreMapSkill


class ExploreAgent(Agent):
    def __init__(self,
                 observation_function: ObservationFunction,
                 max_nav_count : int,
                 simple_explore_baseline : bool
                 ):
        super().__init__()
        self.explore_map_skill = ExploreMapSkill(
            max_nav_count=max_nav_count,
            simple_explore_baseline=simple_explore_baseline
        )
        self.observation_function = observation_function

        # State:
        self.state_repr = None
        self.count = 0

    def start_new_rollout(self, task: Task, state_repr: StateRepr = None):
        self.explore_map_skill.start_new_rollout()
        self.state_repr = state_repr
        self.count = 0

    def finalize(self, total_reward):
        pass

    def get_trace(self, device="cpu"):
        trace = {
            "obs_func": self.observation_function.get_trace(device),
            "explore_map_skill": self.explore_map_skill.get_trace(device)
        }
        return trace

    def clear_trace(self):
        self.explore_map_skill.clear_trace()
        self.observation_function.clear_trace()

    def act(self, observation: Observation) -> Action:
        self.state_repr = self.observation_function(observation, self.state_repr, goal=None)

        ll_action: Action = self.explore_map_skill.act(self.state_repr)
        assert not self.explore_map_skill.has_failed()
        return ll_action
