import copy
from agent.subgoal_agent import BaseSubgoalAgent

from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState

from sledmap.mapper.models.teach.skills.plan_and_execute import PlanAndExecuteSkill

class PDDLSubgoalAgent(BaseSubgoalAgent):
    def __init__(self, args, device, logger=None) -> None:
        super().__init__(args, device, logger)
        self.subgoal_skill = PlanAndExecuteSkill(args, logger=self.logger)

    def reset(self):
        super().reset()
        self.subgoal_skill.reset()
    
    def move_to_next_subgoal(self) -> TeachSubgoal:
        self.subgoal_skill.reset()
        return super().move_to_next_subgoal()
    
    def resolve_subgoal(
        self, tracked_state: NeuralSymbolicAgentState, subgoal: TeachSubgoal
    ) -> TeachAction:
        # return TeachAction("Stop")
        action = self.subgoal_skill.step(subgoal, tracked_state)
        status = self.subgoal_skill.current_status
        status['curr_subgoal_idx'] = self.sg_pointer
        self.current_status_records.append(status)
        return action

    def create_empty_record(self) -> dict:
        self.record = {
            "predicted_completed_subgoals": [],
            "predicted_future_subgoals": [],
            "gt_completed_subgoals": [],
            "gt_future_subgoals": [],
        }
        for k in self.record:
            for sg in getattr(self, k):
                sg = list(sg) if sg[1] == "parentReceptacles" else list(sg[:2])
                self.record[k].append(sg)
        self.record["execution_record"] = []
        self.record["agent_id"] = self.args.agent_id
        return self.record
    
    def log_completed_subgoal(self, sg_done: TeachSubgoal):
        super().log_completed_subgoal(sg_done)
        self.record['execution_record'].append(self.subgoal_skill.get_record())

    def get_current_status(self) -> dict:
        return self.current_status_records