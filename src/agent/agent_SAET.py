import os, json, copy, random
from datetime import datetime

import torch
import cv2
import numpy as np
import spacy
from attrdict import AttrDict
from pprint import pprint

from .state_saet import ETAgentState, ETAgentAction
from model.model.model import PLModelWrapper
from model.data.teach import TeachDataset
from model.data.navigation import NaviDataset
from model.utils.data_util import load_vocab, process_edh_for_inference
from model.nn.vision import FeatureExtractor
from model.model.object_detector import MaskRCNNDetector
from model.utils.helper_util import create_logger


class TEAChAgentSubgoalAwareET:
    def __init__(self, args, pid, logger=None):
        """ """
        self.args = args
        self.agent_id = pid
        self.gpu_id = pid % args.gpu_number
        self.plot_and_pause = args.plot_and_pause
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
                debug=args.eval_debug,
                output=["file"],
            )
        )
        self.logger.info(args)
        self.logger.info(
            f"Agent No.{self.agent_id} is created on device: {self.device})"
        )
        self.load_modules()

        # some constants
        self.stop_action = ("Stop", 2, 1.0)
        self.none_arg = ("None", 3, 1.0)
        self.end_navi_action = ("[END]", 2, 1.0)

    # def launch_agent(self):

    def load_modules(self):
        self.load_controller()
        self.load_navigator()
        self.load_visfeat_extractor()
        self.load_object_detector()
        self.tokenizer = spacy.load("en_core_web_sm")

        pred2lang_fn = os.path.join(
            self.args.ithor_assets_dir, "predicate_to_language.json"
        )
        self.pred2lang = json.load(open(pred2lang_fn, "r"))
        self.controller_vocab["predicate_to_language"] = self.pred2lang

        afforance_fn = os.path.join(self.args.ithor_assets_dir, "affordance.json")
        self.affordance = json.load(open(afforance_fn, "r"))

    def load_controller(self):
        ctr_root_dir = self.args.controller_dir
        ckpt_dir = os.path.join(ctr_root_dir, self.args.controller_ckpt)
        self.controller_vocab_dir = os.path.join(ctr_root_dir, "vocab")
        self.controller_cfg_dir = os.path.join(ctr_root_dir, "config.json")
        self.controller_vocab = load_vocab(self.controller_vocab_dir)
        self.controller_args = AttrDict(**json.load(open(self.controller_cfg_dir)))
        self.controller_args.device = self.device
        dec_params = {
            "keep_topk_prob": self.args.dec_keep_topk_prob,
            "intent_decoding": self.args.dec_intent_decoding,
            "max_intent_length": self.args.dec_max_intent_length,
        }
        self.controller_args.dec_params = dec_params

        self.controller = PLModelWrapper.load_from_checkpoint(
            ckpt_dir,
            model_cls="controller",
            args=self.controller_args,
            vocabs=self.controller_vocab,
        )
        self.controller = self.controller.to(device=self.device)

        self.controller_data_handler = TeachDataset(
            "Unknown",
            self.controller_args,
            load_data=False,
            vocab_path=self.controller_vocab_dir,
        )
        self.controller.eval()
        self.controller.freeze()
        self.logger.info(f"Controller is loaded from: {ctr_root_dir}")

    def load_navigator(self):
        ckpt_dir = os.path.join(self.args.navigator_dir, self.args.navigator_ckpt)
        self.navigator_vocab_dir = os.path.join(self.args.navigator_dir, "vocab")
        self.navigator_cfg_dir = os.path.join(self.args.navigator_dir, "config.json")
        self.navigator_vocab = load_vocab(self.navigator_vocab_dir)
        self.navigator_args = AttrDict(**json.load(open(self.navigator_cfg_dir)))
        self.navigator_args.device = self.device
        dec_params = {"keep_topk_prob": self.args.dec_keep_topk_prob}
        self.navigator_args.dec_params = dec_params

        self.navigator = PLModelWrapper.load_from_checkpoint(
            ckpt_dir,
            model_cls="navigator",
            args=self.navigator_args,
            vocabs=self.navigator_vocab,
        )
        self.navigator = self.navigator.to(device=self.device)

        self.navigator_data_handler = NaviDataset(
            "Unknown",
            self.navigator_args,
            load_data=False,
            vocab_path=self.navigator_vocab_dir,
        )
        self.navigator.eval()
        self.navigator.freeze()
        self.logger.info(f"Navigator is loaded from: {self.args.navigator_dir}")

    def load_visfeat_extractor(self):
        architecture = self.args.visual_enc_archiecture
        compress_type = self.args.visual_enc_compress_type

        self.visfeat_extractor = FeatureExtractor(
            architecture,
            device=self.device,
            share_memory=True,
            compress_type=compress_type,
        )
        self.logger.info("Pretrained Resnet50 visual feature extractor is loaded")

    def load_object_detector(self):
        ckpt_file = os.path.join(self.args.objdet_dir, self.args.objdet_ckpt)
        obj_list_file = os.path.join(self.args.ithor_assets_dir, "object_to_id.json")
        conf_thresh = self.args.odet_conf_thresh

        self.object_detector = MaskRCNNDetector(
            ckpt_file, obj_list_file, self.device, conf_thresh
        )
        self.logger.info(f"MaskRCNN object detector is loaded from: {ckpt_file}")

    def init_agent_state(self, edh):
        self.agent_state = ETAgentState(
            controller_inputs=edh,
            location={"x": 0, "y": 0, "z": 0},
            pose={"yaw": 0, "pitch": 0},
        )

    def reset(self, edh_instance, edh_history_images, edh_name=None):
        self.logger.info("Start a new edh instance: %s" % edh_name)
        add_intention = self.controller.model.add_intention
        edh = process_edh_for_inference(
            edh_instance,
            edh_history_images,
            self.controller_vocab,
            self.tokenizer,
            add_intention=add_intention,
            controller=self.controller.model if add_intention else None,
            data_handler=self.controller_data_handler if add_intention else None,
        )

        if edh["frames"] and self.controller_data_handler.enable_vision:
            edh["vis_feats"] = self.visfeat_extractor.featurize(
                edh["frames"], batch_size=8
            )

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

        self.logger.debug("Dialog history: \n" + dialogs)
        self.logger.debug("Action history: \n" + history_actions_str)

        self.logger.debug("%d history frames are processed" % len(edh["frames"]))
        self.init_agent_state(edh)
        self.logger.info("Agent state is initialized")

    def predict_controller_action(
        self, frame, invalid_actions=["Text", "[PAD]", "[BEG]"], verbose=True
    ):
        #

        self.agent_state.is_navigating = False
        last_action = self.agent_state.get_last_action()

        new_input = {"frame": frame}
        if last_action is not None:
            assert last_action.action_type == "interaction"
            new_input.update(
                {
                    "action": last_action.name[0],
                    "arg1": last_action.arg1[0],
                    # "arg2": last_action.arg2[0],
                }
            )

        predictions, updated_inputs = self.controller.predict(
            self.agent_state.controller_inputs,
            new_input,
            self.controller_data_handler,
            self.visfeat_extractor,
        )
        self.agent_state.controller_inputs = updated_inputs
        # if verbose:
        #     self.logger.debug("Controller Inputs: ")
        #     self.logger.debug(updated_inputs["action_input"])
        #     self.logger.debug(updated_inputs["arg1_input"])

        # detect objects in the current observation
        instances = self.object_detector.get_predictions(frame)
        self.agent_state.last_detected_objects = instances
        visible_objs = self.object_detector.get_visibile_objects(instances)

        # combine the action and argument predoctions into interaction proposals
        proposals = []
        for aname in predictions["action_output"]:
            if aname[0] in invalid_actions:
                continue

            if aname[0] == "[END]":
                stop = ("Stop", aname[1], aname[2])
                proposals.append(
                    {
                        "action": stop,
                        "arg1": self.none_arg,
                        "arg2": self.none_arg,
                        "prob": aname[2] * predictions["arg1_output"][0][2],
                    }
                )
            else:
                # interaction actions have one argument which should
                # be visible (detected) and meet the affordance requirement
                valid_arg1_predictions = self.get_valid_argument(
                    aname[0], predictions["arg1_output"], visible_objs
                )
                for arg1 in valid_arg1_predictions:
                    proposals.append(
                        {
                            "action": aname,
                            "arg1": arg1,
                            "arg2": self.none_arg,
                            "prob": aname[2] * arg1[2],
                        }
                    )
        proposals.sort(key=lambda x: x["prob"], reverse=True)

        if verbose:
            self.logger.debug("Raw controller predictions:")
            action_str = "     Top 3 actions: "
            for a in predictions["action_output"][:3]:
                action_str += "%s(%.3f) " % (a[0], a[2])
            self.logger.debug(action_str)
            arg1_str = "     Top 3 arguments: "
            for a in predictions["arg1_output"][:3]:
                arg1_str += "%s(%.3f) " % (a[0], a[2])
            self.logger.debug(arg1_str)
            if "intent_done" in predictions:
                self.logger.debug(
                    "     Intent: \n        DONE: %s\n        TODO: %s"
                    % (predictions["intent_done"], predictions["intent_todo"])
                )
            self.logger.debug("Visible objs:" + str(visible_objs))
            self.logger.debug("Top3 controller proposals:")
            for p in proposals[:3]:
                self.logger.debug(
                    "     %s %s %.3f" % (p["action"][0], p["arg1"][0], p["prob"])
                )

        return proposals

    def get_controller_action(
        self,
        frame=None,
        proposals=None,
        idx_in_proposal=0,
        record_log=True,
        select_obj_idx=0,
    ):
        # first see whether we need to make a new prediction
        if proposals is None and frame is not None:
            proposals = self.predict_controller_action(frame)

        # if not even a good prediction exists in the proposal, backtrack
        last_action = self.agent_state.get_last_action()
        if not proposals or proposals[0]["prob"] < 1e-3:
            self.logger.debug("No good prediction: backtrack")
            return self.controller_backtrack(last_action)

        # assume the proposal is good: try to select an action

        # in a regular prediction, idx_in_proposal=0 which corresponds to selecting
        # the predicted action with the highest probability
        selected_action = proposals[idx_in_proposal]

        # manual intervention to handle some special cases:
        check_num = 3
        curr = (selected_action["action"][0], selected_action["arg1"][0])
        prevs = self.agent_state.get_prev_interactions(number=check_num)

        # case 1: change repeated prediction
        # if the action is a vanilla model prediction instead of a intervened prediction
        # from error handling, we manully discard repeated prediction
        # E.g. (pickup Mug, pickup Mug) -> (pickup Mug, Goto xxx)
        if (
            last_action is not None
            and frame is not None
            and curr == prevs[0]
            and curr[0] in ["Pickup", "Place", "Pour"]
        ):
            try:
                selected_action = proposals[idx_in_proposal + 1]
            except:
                selected_action = {"action": self.stop_action, "arg1": self.none_arg}
                idx_in_proposal = select_obj_idx = -1

        # case 2: early stop prediction to escape from endless loops such as
        # pick - place - pick - place - pick - ...
        # goto - open - close - goto - open - close - ...
        if (
            self.agent_state.controller_step_local > 32
            or self.agent_state.controller_step_global > 99
        ):
            selected_action = {"action": self.stop_action, "arg1": self.none_arg}
            idx_in_proposal = select_obj_idx = -1
            self.logger.debug("Terminate: escape from repeated predictions")

        # case 3: stop if goto X repeated for *check_num* times: cannot find X
        if (
            curr[0] == "Goto"
            and len(prevs) == check_num
            and all([a == curr for a in prevs])
        ):
            selected_action = {"action": self.stop_action, "arg1": self.none_arg}
            idx_in_proposal = select_obj_idx = -1
            self.logger.debug("Terminate: cannot find %s %s" % (curr))

        # select the interaction point
        if selected_action["action"][0] in ["Goto", "Stop"]:
            point_proposals = selected_interact_point = None
        else:
            instances = self.agent_state.last_detected_objects
            point_proposals = self.object_detector.get_interaction_points(
                instances, selected_action["arg1"][0]
            )
            assert point_proposals
            # print("proposals")
            # pprint(proposals)
            # print(idx_in_proposal, select_obj_idx)
            # print("selected action:", selected_action)
            # print("point_proposals")
            # pprint(point_proposals)
            # TODO: more complex point selection heruistic
            selected_interact_point = point_proposals[select_obj_idx][0]

        agent_action = ETAgentAction(
            action_type="interaction",
            proposals=proposals,
            name=selected_action["action"],
            idx_in_proposal=idx_in_proposal,
            interaction_point=selected_interact_point,
            point_proposals=point_proposals,
            point_idx_in_proposal=select_obj_idx,
            arg1=selected_action["arg1"],
            # arg2=selected_action["arg2"],
        )
        self.agent_state.record_action(agent_action)

        if agent_action.name[0] in ["Goto", "Stop"]:
            self.agent_state.controller_step_global += 1
            self.agent_state.controller_step_local += 1

        if record_log:
            # pprint(predictions)
            # print(pred_action, pred_arg1, pred_arg2)
            ctrl_step = self.agent_state.controller_step_global
            self.logger.debug(f"Interaction: [{agent_action}] (c_step: {ctrl_step})")
            if agent_action.name[0] == "Stop":
                self.logger.debug("Interaction terminates. ")

        return agent_action

    def get_valid_argument(self, action_name, arg_predictions, visible_objs):
        """filter invalid action arguments according to object visiblility

        :param action_name: action name string
        :param arg_predictions: list of tuples (obj_name, obj_idx, probability)
        :param visible_objs: visible (detected) object class names
        :return: valid argument predictions
        """
        if action_name == "Goto":
            return arg_predictions

        valid_objs = self.affordance[action_name]

        valid_arg_predictions = []
        for arg_pred in arg_predictions:
            obj = arg_pred[0]
            if obj in valid_objs and obj in visible_objs:
                valid_arg_predictions.append(arg_pred)

        return valid_arg_predictions

    def predict_navigation_action(self, frame, init_navigation=False, verbose=True):
        new_input = {"frame": frame}
        last_action = self.agent_state.get_last_action()
        assert last_action is not None
        assert last_action.succeed

        if init_navigation:  # start a new navigation sub-procedure
            assert last_action.action_type == "interaction"
            self.agent_state.is_navigating = True
            self.agent_state.navigation_step_local = 0
            self.agent_state.controller_step_local = 0
            self.agent_state.navigator_inputs = {
                "goal": [
                    last_action.arg1[1],
                    # navi_action.arg2[1],
                ],
                "action_input": [1],  # TODO: encode [BEG] token
            }
        else:  # resume navigating
            last_action = self.agent_state.get_last_action()
            assert last_action.action_type == "navigation"
            new_input["action"] = last_action.name[1]

        predictions, updated_inputs = self.navigator.predict(
            self.agent_state.navigator_inputs,
            new_input,
            self.navigator_data_handler,
            self.visfeat_extractor,
        )
        self.agent_state.navigator_inputs = updated_inputs

        if verbose:
            self.logger.debug("Raw navigation predictions:")
            action_str = "   Top 3 actions: "
            for a in predictions[:3]:
                action_str += "%s(%.3f) " % (a[0], a[2])
            self.logger.debug(action_str)

        return predictions

    def get_navigator_action(self, frame, init_navigation=False):
        predictions = self.predict_navigation_action(
            frame, init_navigation=init_navigation
        )
        # pprint(predictions)
        idx_in_proposal = 0
        pred_action = predictions[idx_in_proposal]
        if self.check_loops_in_navigation(pred_action):
            for idx, pred_action in enumerate(predictions):
                if "Turn" not in pred_action[0]:
                    idx_in_proposal = idx
                    break
            self.logger.debug("Manully escape from loop")

        if self.agent_state.navigation_step_local > 32:
            pred_action = self.end_navi_action
            idx_in_proposal = -1
            self.logger.debug("Manully escape from endless navigation")

        if init_navigation and pred_action[0] != "[END]":
            instances = self.agent_state.last_detected_objects
            visible_objs = self.object_detector.get_visibile_objects(instances)
            target_obj = self.agent_state.interaction_history[-1].arg1[0]
            if target_obj in visible_objs:
                pred_action = self.end_navi_action
                idx_in_proposal = -1
                self.logger.debug(
                    "Alreadly reach the target object thus no need to "
                    "navigate. Manully change %s to [END]"%(pred_action[0])
                )

        agent_action = ETAgentAction(
            action_type="navigation",
            proposals=predictions,
            name=pred_action,
            idx_in_proposal=idx_in_proposal,
        )
        self.agent_state.record_action(agent_action)

        sloc = self.agent_state.navigation_step_local
        sglob = self.agent_state.navigation_step_global
        self.logger.debug(f"Navigation: [{pred_action[0]}] (n_step: {sloc}|{sglob})")
        return agent_action

    def check_loops_in_navigation(self, curr_action):
        curr_aname = curr_action[0]
        is_in_loop = False
        if len(self.agent_state.navigation_history) >= 3:
            last_actions = self.agent_state.navigation_history[:-4:-1]
            if (
                curr_aname in ["Turn Right", "Turn Left"]
                and self.agent_state.navigation_history[-1].name[0] == curr_aname
                and self.agent_state.navigation_history[-2].name[0] == curr_aname
                and self.agent_state.navigation_history[-3].name[0] == curr_aname
            ):
                is_in_loop = True
        return is_in_loop

    def step(self, frame):
        """Given the current observation, predict the action to take"""

        # if the last action fails, try another action from the predictions
        last_action_succeed = self.agent_state.check_success(frame, self.logger)
        if not last_action_succeed:
            self.agent_state.mark_last_action_failed()
            self.logger.debug(
                "[Action Failed] #failures: %d (total: %d)"
                % (
                    self.agent_state.num_failures_local,
                    self.agent_state.num_failures_global,
                )
            )
            self.logger.debug("Try self-correction:")

            agent_action = (
                self.navigation_error_handling()
                if self.agent_state.is_navigating
                else self.interaction_error_handling()
            )
            agent_action = self.forward_one_more_step(frame, agent_action)

            if self.plot_and_pause:
                TEAChAgentSubgoalAwareET.plot_frame_and_pause(frame, agent_action)
            return agent_action

        # The last action is believed to be successfully executed.
        # Update the agent's internal state according to the effect of that action.
        self.agent_state.update(frame)
        self.logger.debug("[Action Succeed] Env step: %d" % self.agent_state.env_step)

        # predict the next action
        if self.agent_state.is_navigating:
            agent_action = self.get_navigator_action(frame, init_navigation=False)
        else:
            agent_action = self.get_controller_action(frame)
        agent_action = self.forward_one_more_step(frame, agent_action)

        if self.plot_and_pause:
            TEAChAgentSubgoalAwareET.plot_frame_and_pause(frame, agent_action)
        return agent_action

    def forward_one_more_step(self, frame, agent_action):
        """predict another action
        Step must return a physical action that is executable in the environment.
        However, in a hierarchical policy, some actions are conceptual such
        as openning/ending a navigation sub-process. For these actions,
        we need to forward another step to get its physical action for
        execution.
        For example, after the controller predicts the "Goto" action, the
        navigator needs to predict the first navigation action as the return
        of the function step.

        :param agent_action: the predicted conceptual action
        """

        if self.agent_state.is_navigating and agent_action.name[0] == "[END]":
            self.logger.debug(
                "Navigation ends. Need to predict the next interaction action"
            )
            # if it is predicted to be the end of a navigation,
            # we need to predict the next interaction action

            select_idx = self.agent_state.mental_activity_step
            try:
                agent_action = self.get_controller_action(
                    frame, idx_in_proposal=select_idx
                )
            except IndexError:
                agent_action = ETAgentAction(
                    action_type="interaction",
                    proposals=agent_action.proposals,
                    name=("Stop", 2, 1.0),
                    idx_in_proposal=-1,
                )
                self.agent_state.record_action(agent_action)
                self.logger.debug("Terminate because failed to move on.")
            except Exception as e:
                print("Debug!")
                import sys, traceback

                traceback.print_exc(file=sys.stdout)
                quit()
            self.agent_state.mental_activity_step += 1

        elif not self.agent_state.is_navigating and agent_action.name[0] == "Goto":
            self.logger.debug("Navigation starts. Need to predict the first movement. ")
            # have to predict the first primitive navigation action for "Goto"
            agent_action = self.get_navigator_action(frame, init_navigation=True)
        else:
            # for execuable actions directly return them
            return agent_action

        # the agent forwarded step may still not be a primitive action
        agent_action = self.forward_one_more_step(frame, agent_action)
        return agent_action

    def navigation_error_handling(self):
        last_action = self.agent_state.get_last_action()
        old_idx_in_proposal = last_action.idx_in_proposal
        new_idx_in_proposal = old_idx_in_proposal + random.randint(1, 2)
        # TODO: modify this; forward and pan right both failed
        invalid_action_mapping = {
            "Forward": ['Forward', 'Backward'],
            "Backward": ['Forward', 'Backward'],
            "Look Up": ['Look Up', 'Look Down'],
            "Look Down": ['Look Up', 'Look Down'],
            "Pan Left": ['Pan Left', 'Pan Right'],
            "Pan Right": ['Pan Left', 'Pan Right'],
        }
        invalid_list = invalid_action_mapping.get(last_action.name[0], [])
        for new_action in last_action.proposals:
            if new_action[0] not in invalid_list:
                break
        # if new_idx_in_proposal < len(last_action.proposals):
        #     new_action = last_action.proposals[new_idx_in_proposal]
        # else:
        #     new_action = self.end_navi_action
        #     new_idx_in_proposal = -1

        new_agent_action = ETAgentAction(
            action_type="navigation",
            proposals=last_action.proposals,
            name=new_action,
            idx_in_proposal=new_idx_in_proposal,
        )
        self.agent_state.record_action(new_agent_action)
        self.logger.debug(
            f"Relpace failed navigation action [{last_action}] by [{new_agent_action}]"
        )
        return new_agent_action

    def interaction_error_handling(self):

        last_action = self.agent_state.get_last_action()
        num_failures = self.agent_state.num_failures_local
        if num_failures >= 5:
            # too many failures, backtrack to the previous navigation action
            return self.controller_backtrack(last_action)

        new_point_idx = last_action.point_idx_in_proposal + 1
        new_action_idx = last_action.idx_in_proposal + 1

        if new_point_idx < len(last_action.point_proposals):
            # if there are multiple instances of the predicted object class, try another
            new_agent_action = self.get_controller_action(
                proposals=last_action.proposals,
                idx_in_proposal=last_action.idx_in_proposal,
                record_log=False,
                select_obj_idx=new_point_idx,
            )
        elif new_action_idx < len(last_action.proposals):
            # try different predicted actions
            new_agent_action = self.get_controller_action(
                proposals=last_action.proposals,
                idx_in_proposal=new_action_idx,
                record_log=False,
            )
        else:
            return self.controller_backtrack(last_action)

        self.logger.debug(
            f"Relpace failed interaction action [{last_action}] by [{new_agent_action}]"
        )

        return new_agent_action

    def controller_backtrack(self, last_action):

        if self.agent_state.num_backtracks < self.args.max_allowed_backtracks:
            self.logger.debug("Try to backtrack")
            bt_step_num = 0
            for a in self.agent_state.interaction_history[::-1]:
                bt_step_num += 1
                if a.name[0] == "Goto":
                    new_action = copy.deepcopy(a)
                    self.agent_state.record_action(new_action)
                    self.agent_state.num_backtracks += 1
                    self.logger.debug(
                        f"Backtrack to previous navigation action "
                        f"[Goto {new_action.arg1[0]}]"
                        # f"[Goto {new_action.arg1[0]} {new_action.arg2[0]}]"
                    )

                    # we need to modify the controller input to exclude actions
                    # after this 'Goto' action (include itself since it will be
                    # added as the new_input in the prediction)
                    ctrl_input = self.agent_state.controller_inputs
                    for k, v in ctrl_input.items():
                        if isinstance(v, list):
                            ctrl_input[k] = ctrl_input[k][:-bt_step_num]
                    return new_action

        # if not find a point to backtrack, stop the interaction
        new_agent_action = ETAgentAction(
            action_type="interaction",
            proposals=last_action.proposals,
            name=self.stop_action,
            idx_in_proposal=-1,
        )
        self.agent_state.record_action(new_agent_action)
        self.logger.debug(
            "Interaction terminates due to too many failures or "
            "no available backtracking point."
        )
        return new_agent_action

    @classmethod
    def plot_frame_and_pause(cls, frame, agent_action):
        print(agent_action)
        frame_np = np.array(frame)
        if agent_action.interaction_point:
            x = int(agent_action.interaction_point[0])
            y = int(agent_action.interaction_point[1])
            cv2.circle(frame_np, (x, y), 5, thickness=-1, color=(255, 0, 0))
        curr_cv2 = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        cv2.imshow("current obs", curr_cv2)
        cv2.waitKey(0)
