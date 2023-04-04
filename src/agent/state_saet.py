import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Dict, Tuple, Optional
from detectron2.structures.instances import Instances as dt2_instances_type

import cv2
from PIL.Image import Image

NONE_ARG = ("None", 3, 1.0)


@dataclass
class ETAgentAction:
    action_type: str  # one of ['navigation', 'interaction']

    # dict of action prediction proposals
    # Available keys: action_output/arg1_output/arg2_output
    # Each tuple consists of (name, index, probability)
    proposals: list

    # the selected action to be executed
    idx_in_proposal: int  # index in the proposals
    name: tuple  # action name in the form (name_string, index, probability)
    arg1: Optional[tuple] = NONE_ARG  # (obj name, index, probability) of the 1st arg
    arg2: Optional[tuple] = NONE_ARG  # (obj name, index, probability) of the 2nd arg

    # point coordinate in agent's egocentric view
    point_proposals: Optional[list] = None  # bbox cener proposals from detector
    interaction_point: Optional[List[int]] = None  # relative interaction
    point_idx_in_proposal: int = 0  # point index in the proposals

    # whether the action is successfully executed
    succeed: bool = True

    def __repr__(self):
        a = f"{self.name[0]}"
        if self.arg1 is not None and self.arg1[0] != "None":
            a += f" {self.arg1[0]}"
        if self.arg2 is not None and self.arg2[0] != "None":
            a += f" {self.arg2[0]}"
        if self.interaction_point is not None:
            a += " %.1f %.1f" % (self.interaction_point[0], self.interaction_point[1])

        # probability
        # a += f"({self.name[2]}"
        # if self.arg1 is not None:
        #     a += f" {self.arg1[2]}"
        # if self.arg2 is not None:
        #     a += f" {self.arg2[2]}"
        # a += ")"
        return a

    def mark_failed(self):
        self.succeed = False


@dataclass
class ETAgentState:
    controller_inputs: dict = field(default_factory=dict)  # encoded input
    navigator_inputs: dict = field(default_factory=dict)  # encoded input
    interaction_history: List[ETAgentAction] = field(default_factory=list)
    subgoal_history: Optional[List[ETAgentAction]] = None  # subgoals
    navigation_history: List[ETAgentAction] = field(default_factory=list)
    last_observation: Optional[Image] = None
    last_detected_objects: Optional[dt2_instances_type] = None
    is_navigating: bool = False  # whether the agent is during a navigation
    location: dict = field(default_factory=dict)
    pose: dict = field(default_factory=dict)
    object_in_hand: Optional[str] = None

    controller_step_local: int = 0  # succ. controller actions before navi.
    controller_step_global: int = 0  # succ. controller actions in total
    navigation_step_local: int = 0  # succ. steps in one navigataion process
    navigation_step_global: int = 0  # succ. navigation steps in total
    env_step: int = 0  # succ. environment steps (i.e. primitive actions)
    mental_activity_step: int = 0  # unexecutable mental actions (e.g. Goto)

    # failed actions at this step
    failed_actions_local: List[ETAgentAction] = field(default_factory=list)

    num_failures_local: int = 0  # number of failed attempts at this step
    num_failures_global: int = 0  # number of failed attempts so far
    num_backtracks: int = 0

    last_action_success: bool = True

    def update(self, frame):
        """update the agent's state"""
        self.last_observation = frame

        last_action = self.get_last_action()
        if last_action is not None:
            assert last_action.succeed
            self.env_step += 1
            self.num_failures_local = 0
            self.mental_activity_step = 0

            if self.is_navigating:
                self.navigation_step_local += 1
                self.navigation_step_global += 1
                self.update_loc_pose(last_action.name[0])
            else:
                self.controller_step_local += 1
                self.controller_step_global += 1

            if last_action.name[0] == "Pickup":
                object_in_hand = last_action.arg1[0]
            if last_action.name[0] == "Place":
                object_in_hand = None
            # TODO: add other rules for object state changes

    def update_loc_pose(self, action_name: str):
        if action_name == "Forward":
            self.location["x"] += 0.25
        elif action_name == "Backward":
            self.location["x"] -= 0.25
        elif action_name == "Pan Left":
            self.location["z"] += 0.25
        elif action_name == "Pan Right":
            self.location["z"] -= 0.25
        elif action_name == "Turn Left":
            self.pose["yaw"] = (self.pose["yaw"] - 90) % 360
        elif action_name == "Turn Right":
            self.pose["yaw"] = (self.pose["yaw"] + 90) % 360
        elif action_name == "Look Up":
            self.pose["pitch"] -= 30
        elif action_name == "Look Down":
            self.pose["pitch"] += 30

    def check_success(self, img: Image, logger=None):
        last_action = self.get_last_action()
        if self.last_observation is None or last_action.name[0] in [
            "[END]",
            "Stop",
            None,
        ]:
            self.last_action_success = True
            return True
        # prev = imagehash.average_hash(self.last_observation)
        # curr = imagehash.average_hash(img)
        # self.last_action_success = False if prev - curr < 2 else True

        diff = np.abs(np.array(img) - np.array(self.last_observation)).sum()
        if last_action.name[0] in ["ToggleOn", "ToggleOff"]:
            self.last_action_success = False if diff < 1e5 else True
        else:
            self.last_action_success = False if diff < 2e6 else True

        # prev = np.array(self.last_observation)
        # curr = np.array(img)

        # prev_cv2 = cv2.cvtColor(np.array(self.last_observation), cv2.COLOR_BGR2RGB)
        # curr_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        # cv2.imshow("prev", prev_cv2)
        # cv2.imshow("curr", curr_cv2)
        if logger is not None:
            logger.debug(str(last_action) + " pixel diff:" + str(diff))
        else:
            print(last_action, diff)

        # self.last_action_success = False if img == self.last_observation else True
        return self.last_action_success

    def get_last_action(self):
        if self.is_navigating:
            return self.navigation_history[-1] if self.navigation_history else None
        return self.interaction_history[-1] if self.interaction_history else None

    def get_prev_interactions(self, number):
        if not self.interaction_history:
            return []
        return [
            (a.name[0], a.arg1[0]) for a in self.interaction_history[: -number - 1 : -1]
        ]

    def record_action(self, action):
        if action.action_type == "navigation":
            self.navigation_history.append(action)
        elif action.action_type == "interaction":
            self.interaction_history.append(action)
        else:
            raise TypeError("Invalid action type: %s" % (action.action_type))

    def reset_failure_state(self):
        self.failed_actions_local = []
        self.agent_state.num_failures_local = 0

    def mark_last_action_failed(self):
        last_action = self.get_last_action()
        last_action.mark_failed()
        self.failed_actions_local.append(last_action)
        self.num_failures_local += 1
        self.num_failures_global += 1
