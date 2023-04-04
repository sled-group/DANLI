import os
import copy
import json
from attrdict import AttrDict
import numpy as np
from typing import Tuple, Union, List, Optional, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "true"

from transformers import BartTokenizer

from model.model.model_plm import PLModelWrapper
from model.data.subgoal_seq2seq import SubgoalDataset
from model.utils.data_util import process_edh_for_subgoal_prediction

from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import (
    NeuralSymbolicAgentState,
)


class BaseSubgoalAgent:
    def __init__(self, args, device, logger=None) -> None:
        self.args = args
        self.device = device
        self.logger = logger
        self.log_func = logger.info if logger is not None else print
        self.execution_log_save_dir = os.path.join(args.log_dir, "execution_log")
        if not os.path.exists(self.execution_log_save_dir):
            os.makedirs(self.execution_log_save_dir)
        self._load_subgoal_predictor()
        self.all_gt_subgoals = {}
        if self.args.gt_subgoals_file:
            with open(self.args.gt_subgoals_file, "r") as f:
                self.all_gt_subgoals = json.load(f)
        # state variables: enrue them to be reset in self.reset
        self.predicted_future_subgoals: List[tuple] = []
        self.completed_subgoals: List[TeachSubgoal] = []  # completed by the agent
        self.sg_pointer = 0
        self.current_subgoal = None
        self.STATE = "INIT"  # INIT, TERMINATE, SG_IN_PROGRESS, SG_COMPLETED, SG_FAILED

    def _load_subgoal_predictor(self):
        root_dir = self.args.subgoal_predictor_path
        ckpt_dir = os.path.join(
            root_dir, self.args.subgoal_predictor_ckpt
        )  # "ckpt-epoch=13-avg_accu=0.8526.ckpt")
        cfg_dir = os.path.join(root_dir, "args.json")
        tokenizer_dir = os.path.join(root_dir, "tokenizer/")
        self.sg_predictor_args = AttrDict(**json.load(open(cfg_dir)))
        self.sg_predictor_args.is_inference = True
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_dir)
        self.sg_predictor_data_handler = SubgoalDataset(
            self.tokenizer, args=self.sg_predictor_args, split=None
        )
        self.sg_predictor = PLModelWrapper.load_from_checkpoint(
            ckpt_dir, args=self.sg_predictor_args, tokenizer=self.tokenizer
        )
        self.sg_predictor.to(device=self.device)
        self.sg_predictor.eval()
        self.sg_predictor.freeze()
        self.log_func(f"Subgoal predictor is loaded from: {ckpt_dir}")

    def reset(self):
        self.predicted_future_subgoals = []
        self.completed_subgoals = []
        self.sg_pointer = 0
        self.current_subgoal = None
        self.STATE = "INIT"

    def set_subgoals(self, edh_instance):
        self.edh_file = (
            edh_instance["instance_id"]
            if "instance_id" in edh_instance
            else edh_instance["game_id"]
        )
        edh, dialog_history = process_edh_for_subgoal_prediction(edh_instance)
        edh = self.sg_predictor_data_handler.data_collect(
            [edh], inference_mode=True, device=self.device
        )
        sg_predictions = self.sg_predictor.predict(edh, max_step=64)

        predicted_completed_subgoals = sg_predictions[0]["subgoals_done"]
        predicted_future_subgoals = sg_predictions[0]["subgoals_future"]
        # remove the last EOS token
        self.predicted_completed_subgoals = predicted_completed_subgoals[:-1]
        self.predicted_future_subgoals = predicted_future_subgoals[:-1]

        # self.predicted_future_subgoals = self.predicted_future_subgoals + [
        #     ('Bread', 'isSliced', 'Pan'),
        #     # ('PotatoSliced', 'isCooked', '111'),
        # ]

        self.log_func("Completed subgoals:")
        self.log_func(self.predicted_completed_subgoals)
        self.log_func("Future subgoals:")
        self.log_func(self.predicted_future_subgoals)
        self.gt_completed_subgoals = []
        self.gt_future_subgoals = []
        if self.all_gt_subgoals and "instance_id" in edh_instance:
            game_id = edh_instance["game_id"]
            edh_id = edh_instance["instance_id"].split(".")[-1]
            if (
                game_id in self.all_gt_subgoals
                and "gt_all_subgoals" in self.all_gt_subgoals[game_id]
            ):
                game_gt_subgoals = self.all_gt_subgoals[game_id]["gt_all_subgoals"]
                if edh_id in self.all_gt_subgoals[game_id]["edh_to_gt_subgoal_idx"]:
                    edh_sg_idx = self.all_gt_subgoals[game_id]["edh_to_gt_subgoal_idx"][
                        edh_id
                    ]
                    self.gt_future_subgoals = [game_gt_subgoals[i] for i in edh_sg_idx]
                    if edh_sg_idx:
                        self.gt_completed_subgoals = game_gt_subgoals[: edh_sg_idx[0]]

            self.log_func(f"GT future subgoals for game {game_id} {edh_id}:")
            self.log_func(self.gt_future_subgoals)

        if self.args.use_gt_subgoals:
            self.predicted_future_subgoals = self.gt_future_subgoals

        self.record = self.create_empty_record()

        return self.predicted_future_subgoals, dialog_history

    def step(self, tracked_state: NeuralSymbolicAgentState) -> TeachAction:
        # return TeachAction(action_type="Stop")

        print("=============================> Subgoal Agent Step")
        self.current_status_records = []
        while True:
            if self.STATE == "TERMINATE":
                self.save_execution_log()
                return TeachAction(action_type="Stop")

            elif self.STATE == "INIT":
                self.log_func(
                    " ======> ALL predicted subgoals: %r"
                    % self.predicted_future_subgoals
                )
                if self.args.last_subgoal_only:
                    self.sg_pointer = len(self.predicted_future_subgoals) - 1
                lifted_sg = self.predicted_future_subgoals[self.sg_pointer]
                sg = TeachSubgoal.create_from_predicted_tuple(lifted_sg)
                if not self.args.last_subgoal_only:
                    self.log_func("Start addressing the first subgoal: %r" % sg)
                else:
                    self.log_func("Directly work on the last subgoal: %r" % sg)
                self.current_subgoal = sg
                self.STATE = "SG_IN_PROGRESS"

            elif self.STATE == "SG_IN_PROGRESS":
                curr_sg = self.get_current_subgoal()
                self.log_func("Addressing subgoal: %r ... " % curr_sg)
                action = self.resolve_subgoal(tracked_state, curr_sg)

                if action.is_stop():
                    # successfully completed the subgoal
                    self.STATE = "SG_COMPLETED"
                elif action.is_failed():
                    self.STATE = "SG_FAILED"
                else:
                    return action

            elif self.STATE == "SG_COMPLETED":
                sg_done = self.get_current_subgoal()
                self.log_completed_subgoal(sg_done)
                self.log_func("SG%d completed: %r " % (self.sg_pointer, sg_done))
                sg_next = self.move_to_next_subgoal()
                if sg_next is None:
                    self.log_func("All subgoals are completed or failed. Terminate!")
                    self.STATE = "TERMINATE"

                elif self.check_completion(tracked_state, sg_next):
                    self.log_func("Next subgoal is already completed: %r " % (sg_next))
                    self.STATE == "SG_COMPLETED"

                else:
                    self.log_func("Begin to work on the next subgoal: %r" % sg_next)
                    self.STATE = "SG_IN_PROGRESS"

            elif self.STATE == "SG_FAILED":
                # if the subgoal is failed, it means we have tried our best.
                sg = self.get_current_subgoal()
                self.log_func(
                    "SG%d: %r failed! Denote as completed and move on. "
                    % (self.sg_pointer, sg)
                )
                self.STATE = "SG_COMPLETED"

                # if self.backtrack_num < self.max_allowed_backtracks:
                #     sg = self.backtrack_prev_navi_subgoal()
                #     if sg is not None:
                #         self.backtrack_num += 1
                #         self.log_func("Backtrack to and retry subgoal: %r" % self.sg)
                #         self.STATE = "SG_IN_PROGRESS"
                #     else:
                #         self.log_func("No more backtrackable subgoals. Terminate!")
                #         self.STATE = "TERMINATE"

                # else:
                #     self.log_func("Reach max allowed backtracks. Terminate!")
                #     self.STATE = "TERMINATE"

    def get_current_subgoal(self) -> TeachSubgoal:
        return self.current_subgoal

    def log_completed_subgoal(self, sg_done: TeachSubgoal):
        self.completed_subgoals.append(sg_done)
        self.sg_pointer += 1

    def move_to_next_subgoal(self) -> TeachSubgoal:
        if self.sg_pointer == len(self.predicted_future_subgoals):
            return None

        lifted_sg = self.predicted_future_subgoals[self.sg_pointer]
        next_sg = TeachSubgoal.create_from_predicted_tuple(
            lifted_sg
        )  # TODO: add exclude instance list here

        self.current_subgoal = next_sg
        return next_sg

    def resolve_subgoal(
        self, tracked_state: NeuralSymbolicAgentState, subgoal: TeachSubgoal
    ) -> TeachAction:
        """
        Resolve the subgoal here
        """
        raise NotImplementedError("Have to implemente the subgoal solver here")

    def backtrack_subgoal(
        self,
    ) -> Union[TeachSubgoal, None]:
        """
        Backtrack to a previous subgoal
        """
        raise NotImplementedError("Have to implemente the subgoal backtracker here")

    def check_completion(
        self, tracked_state: NeuralSymbolicAgentState, subgoal: TeachSubgoal
    ):
        """
        Check if the subgoal is aleardy completed
        """
        # only consider "food making" subgoals
        if subgoal.predicate not in [
            "isCooked",
            "simbotIsFilledWithCoffee",
            "isSliced",
        ]:
            return False

        num_done = 0
        for sg in (
            self.completed_subgoals + self.predicted_future_subgoals[: self.sg_pointer]
        ):
            if sg == subgoal.predicated_subgoal_tuple:
                num_done += 1
        all_instances = tracked_state.get_all_instances()

        num_goal_instances = 0
        for instance in all_instances:
            if TeachSubgoal.check_goal_instance(subgoal, instance, all_instances):
                num_goal_instances += 1

        if num_goal_instances > num_done:
            return True
        return False

    def create_empty_record(self) -> dict:
        return {}

    def save_execution_log(
        self,
    ):
        file_name = os.path.join(
            self.execution_log_save_dir, "log_%s.json" % self.edh_file
        )
        with open(file_name, "w") as f:
            json.dump(self.record, f, indent=2)

    def get_current_status(self) -> dict:
        return self.current_status_records
