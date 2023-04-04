import os, json, copy, random
from datetime import datetime
from typing import List, Tuple, Union, Any

import torch
import numpy as np
from pprint import pprint

from model.utils.helper_util import create_logger

# from model.model.object_detector import MaskRCNNDetector
from model.model.oracle_navigator import OracleNavigator

from agent.fsm_subgoal_agent import FSMSubgoalAgent
from agent.pddl_subgoal_agent import PDDLSubgoalAgent


# Mapper
from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.env.teach.neural_symbolic_state_tracker import (
    NeuralSymbolicAgentStateTracker,
)

random.seed(19980316)
np.random.seed(19980316)
torch.manual_seed(19980316)
torch.cuda.manual_seed(19980316)
torch.cuda.manual_seed_all(19980316)


class NeuralSymbolicAgent:
    """
    Hierarchical agent with subgoal prediction and sub-policies for manipulation and navigation
    """

    def __init__(self, args, pid, logger=None):
        """ """
        self.args = args
        self.agent_id = pid
        self.args.agent_id = pid
        self.gpu_id = pid % args.gpu_number
        self.plot = args.plot
        self.pause_at = args.pause_at
        self.device = (
            f"cuda:{self.gpu_id}" if args.eval_device == "cuda" else args.eval_device
        )
        # self.logger = create_logger("agent%s" % self.agent_id, debug=args.eval_debug)
        current_time = datetime.now().strftime("%H:%M:%S")
        self.logger = (
            logger
            if logger is not None
            else create_logger(
                None,
                log_name="agent%d_%s.txt" % (self.agent_id, current_time),
                log_dir=args.log_dir,
                debug=args.debug,
                output=["file"],
            )
        )
        self.logger.info(args)
        with open(os.path.join(args.log_dir, "args_%d.json" % self.agent_id), "w") as f:
            json.dump(self.args.__dict__, f, indent=4)
        self.logger.info(
            f"Agent No.{self.agent_id} is created on device: {self.device})"
        )

        self._build()

        self.STATE = "INIT"  # INIT, IN_PROGRESS, TERMINATE

    def _build(self):
        self.state_tracker = NeuralSymbolicAgentStateTracker(
            self.args, self.device, self.logger
        )

        self.logger.info("Building agent type: %s" % self.args.agent_type)
        if self.args.agent_type == "pddl":
            self.subgoal_agent = PDDLSubgoalAgent(self.args, self.device, self.logger)
        elif self.args.agent_type == "fsm":
            self.subgoal_agent = FSMSubgoalAgent(self.args, self.device, self.logger)
            if self.args.save_meta_data:
                raise NotImplementedError(
                    "Saving demo meta data is not implemented for FSM agent"
                )
        else:
            raise ValueError("Unknown agent type: %s" % self.args.agent_type)

        # self.oracle_navigator = OracleNavigator(self.logger)
        # self.subgoal_agent.oracle_navigator = self.oracle_navigator

    def reset(self, env=None):
        self.state_tracker.reset(env=env)
        self.subgoal_agent.reset()
        # self.oracle_navigator.reset(env=env)
        self.last_action = TeachAction.create_empty_action()
        self.tracked_state = None
        self.env_step = 0
        self.replay_step_num = 0
        self.predicted_action_history = []
        self.all_meta_data = {"steps": []}
        self.STATE = "INIT"

    def start_new_edh(
        self,
        edh_instance,
        edh_history_images,
        edh_name=None,
        env=None,
        history_events=None,
    ):
        self.reset(env=env)
        self.env = env
        self.logger.info("Start a new edh instance: %s" % edh_name)
        if self.args.save_meta_data:
            meta_folder = edh_name.split("/")[-1][:-5].replace(".", "_")
            self.meta_save_dir = os.path.join(self.args.meta_save_dir, meta_folder)
        else:
            self.meta_save_dir = None

        # record for debug
        dialogs = edh_instance["dialog_history_cleaned"]
        dialogs = "\n".join([t[0] + ": " + t[1] for t in dialogs])
        history_actions = edh_instance["driver_action_history"]
        history_actions_str = []
        for a in history_actions:
            a_str = a["action_name"]
            if a["oid"] is not None:
                a_str += " %s %.2f %.2f" % (a["oid"].split("|")[0], a["x"], a["y"])
            history_actions_str.append(a_str)
        history_actions_str = " ".join(history_actions_str)

        self.logger.info("Dialog history: \n" + dialogs)
        self.logger.info("Action history: \n" + history_actions_str)
        self.logger.info("Agent state is initialized")
        self.dialogs = dialogs

        self.logger.info("Predict subgoals from edh history:")
        future_subgoals, dialog_history = self.subgoal_agent.set_subgoals(edh_instance)
        if not future_subgoals:
            self.logger.info("No subgoal is predicted. End session!")
            self.STATE = "TERMINATE"
            if self.args.agent_type == "pddl":
                self.subgoal_agent.record["external_end_reason"] = "no_subgoal"
                self.subgoal_agent.record["execution_record"].append(
                    self.subgoal_agent.subgoal_skill.get_record()
                )
                self.subgoal_agent.save_execution_log()
            return
        self.all_meta_data["subgoals"] = future_subgoals
        self.all_meta_data["dialogs"] = dialog_history

        self.logger.info(
            "Drive the agent following the ground truth history trajectory:"
        )
        edh_driver_action_history = edh_instance["driver_action_history"]
        self.tracked_state, self.replay_step_num = self.state_tracker.replay_trajectory(
            edh_history_images,
            edh_driver_action_history,
            events=history_events,
            verbose=self.args.debug,
            visualize=self.plot,
            meta_data=self.all_meta_data,
            meta_save_dir=self.meta_save_dir,
        )
        self.last_action = self.state_tracker.last_action

    def step(self, frame) -> TeachAction:
        """Given the current observation, predict the action to take"""
        event = self.env.last_event if self.env is not None else None

        self.tracked_state = self.state_tracker.step(
            frame, self.last_action, event, verbose=self.args.debug
        )
        self.current_frame = frame

        while True:
            if self.STATE == "TERMINATE":
                terminate_action = self.return_action(TeachAction.stop_action())
                if self.args.save_meta_data:
                    with open(
                        os.path.join(self.meta_save_dir, "meta_data.json"), "w"
                    ) as f:
                        json.dump(self.all_meta_data, f, indent=4)
                return terminate_action

            elif self.STATE == "INIT":
                self.logger.info("Task Context: \n" + self.dialogs)
                self.STATE = "IN_PROGRESS"
                continue

            elif self.STATE == "IN_PROGRESS":
                if self.env_step < self.args.max_steps:
                    action = self.subgoal_agent.step(self.tracked_state)

                    if len(self.predicted_action_history) > 12:
                        if all(
                            [
                                i.action_type
                                in ["Turn Left", "Turn Right", "Look Up", "Look Down"]
                                or i.fail
                                for i in self.predicted_action_history[-12:]
                            ]
                        ):
                            self.logger.info(
                                "Stuck in one place and turning endlessly. End session!"
                            )
                            if self.args.agent_type == "pddl":
                                self.subgoal_agent.record[
                                    "external_end_reason"
                                ] = "get_stuck_and_rotate_endlessly"
                                self.subgoal_agent.record["execution_record"].append(
                                    self.subgoal_agent.subgoal_skill.get_record()
                                )
                                self.subgoal_agent.save_execution_log()
                            self.STATE = "TERMINATE"
                            continue
                        action_type = self.predicted_action_history[-1].action_type
                        if action_type != "Forward" and all(
                            [
                                i.action_type == action_type
                                for i in self.predicted_action_history[-10:]
                            ]
                        ):
                            self.logger.info(
                                "Repeat 1 action for 10 times. End session!"
                            )
                            if self.args.agent_type == "pddl":
                                self.subgoal_agent.record[
                                    "external_end_reason"
                                ] = "repeat_action_endlessly"
                                self.subgoal_agent.record["execution_record"].append(
                                    self.subgoal_agent.subgoal_skill.get_record()
                                )
                                self.subgoal_agent.save_execution_log()
                            self.STATE = "TERMINATE"
                            continue
                    if action.is_stop():
                        self.STATE = "TERMINATE"
                        continue

                    return self.return_action(action)
                else:
                    self.logger.info("Reach the maximum steps. End session!")
                    self.STATE = "TERMINATE"
                    if self.args.agent_type == "pddl":
                        self.subgoal_agent.record[
                            "external_end_reason"
                        ] = "reach_overall_step_limit"
                        self.subgoal_agent.save_execution_log()
                    continue

    def return_action(
        self, action: TeachAction
    ) -> Tuple[str, Union[List[float], None]]:

        action_type = action.action_type
        normalized_point = action.interaction_point
        if normalized_point is not None:
            # Note: have to use y,x instead of x,y
            normalized_point = [
                normalized_point[1] / 900,
                normalized_point[0] / 900,
            ]

        if self.args.save_meta_data:
            img_idx = self.replay_step_num + self.env_step

            symbolic_dict = self.tracked_state.get_symbolic_state_dict()
            instance_match_log_info = self.tracked_state.get_instance_match_log_info()
            symbolic_meta = {
                "world_state": symbolic_dict,
                "match_info": instance_match_log_info,
            }
            save_dir = os.path.join(self.meta_save_dir, "symbolic")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, "%d.json" % (img_idx)), "w") as f:
                json.dump(symbolic_meta, f, indent=2)

            all_status = self.subgoal_agent.get_current_status()
            last_action = self.last_action.to_dict()
            if all_status:
                for idx, status in enumerate(all_status):
                    action_to_take = (
                        action.to_dict() if idx == len(all_status) - 1 else None
                    )
                    self.all_meta_data["steps"].append(
                        {
                            "img_idx": img_idx,
                            "step_idx": self.env_step,
                            "stage": "rollout",
                            "status": status,
                            "action_to_take": action_to_take,
                            "last_action": last_action,
                            "last_action_success": not self.last_action.is_failed(),
                        }
                    )
            else:
                action_to_take = action.to_dict()
                self.all_meta_data["steps"].append(
                    {
                        "img_idx": img_idx,
                        "step_idx": self.env_step,
                        "stage": "rollout",
                        "status": None,
                        "action_to_take": action_to_take,
                        "last_action": last_action,
                        "last_action_success": not self.last_action.is_failed(),
                    }
                )

        if self.plot:
            wait_key = None
            if self.pause_at == "each_step":
                wait_key = 0
            elif self.pause_at == "interaction" and action.is_interaction():
                wait_key = 0

            # wait_key = None if self.env_step < 35 else 0
            # wait_key = 0
            voxel_visualizer = self.state_tracker.voxel_grid_visualizer
            self.tracked_state.visualize(
                action=action,
                waitkey=wait_key,
                voxel_grid_visualizer=voxel_visualizer,
                # voxel_grid_visualizer=None,
                interactive=False,  # (self.env_step %5 == 0)
                save_dir=self.meta_save_dir,
                img_idx=self.replay_step_num + self.env_step,
            )

        coord_str = (
            "" if not normalized_point else " (%.2f, %.2f)" % tuple(normalized_point)
        )
        self.logger.info(
            "[Returned Action] %s %s%s | Env step: %d"
            % (action_type, normalized_point, coord_str, self.env_step)
        )

        self.env_step += 1
        self.last_action = action
        self.state_tracker.log_action(action)
        self.predicted_action_history.append(action)

        return action_type, normalized_point
