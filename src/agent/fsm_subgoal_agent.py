from agent.subgoal_agent import BaseSubgoalAgent

from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState

from sledmap.mapper.models.teach.skills.subgoal_skill_get import SubgoalSkillGet
from sledmap.mapper.models.teach.skills.subgoal_skill_place import SubgoalSkillPlace
from sledmap.mapper.models.teach.skills.subgoal_skill_slice import SubgoalSkillSlice
from sledmap.mapper.models.teach.skills.subgoal_skill_clean import SubgoalSkillClean
from sledmap.mapper.models.teach.skills.subgoal_skill_empty import SubgoalSkillEmpty
from sledmap.mapper.models.teach.skills.subgoal_skill_coffee import SubgoalSkillCoffee
from sledmap.mapper.models.teach.skills.subgoal_skill_water import SubgoalSkillWater
from sledmap.mapper.models.teach.skills.subgoal_skill_cook import SubgoalSkillCook


class FSMSubgoalAgent(BaseSubgoalAgent):
    def __init__(self, args, device, logger=None) -> None:
        super().__init__(args, device, logger)

        self.subgoal_resolver_functions = {
            "isCooked": SubgoalSkillCook,
            "isClean": SubgoalSkillClean,
            "isPickedUp": SubgoalSkillGet,
            "isFilledWithLiquid": SubgoalSkillWater,
            "isEmptied": SubgoalSkillEmpty,
            "isSliced": SubgoalSkillSlice,
            "simbotIsBoiled": SubgoalSkillCook,
            "simbotIsFilledWithCoffee": SubgoalSkillCoffee,
            "parentReceptacles": SubgoalSkillPlace,
        }
        self.subgoal_skill = None

    def reset(self):
        super().reset()
        self.subgoal_skill = None
    
    def resolve_subgoal(
        self, tracked_state: NeuralSymbolicAgentState, subgoal: TeachSubgoal
    ) -> TeachAction:
        # return TeachAction("Stop")
        if self.subgoal_skill is None:
            resolver_type = self.subgoal_resolver_functions[subgoal.predicate]
            self.subgoal_skill = resolver_type(logger=self.logger)
            # self.subgoal_skill.oracle_navigator = self.oracle_navigator
            self.log_func("Subgoal Skill for addressing %s is created" % subgoal.predicate)


        action = self.subgoal_skill.step(subgoal, tracked_state)
        return action

    def move_to_next_subgoal(self) -> TeachSubgoal:
        self.subgoal_skill = None
        return super().move_to_next_subgoal()