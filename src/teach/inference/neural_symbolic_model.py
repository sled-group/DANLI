# SPDX-License-Identifier: MIT-0
import argparse
import numpy as np
from typing import List
from attrdict import AttrDict

from teach.inference.teach_model import TeachModel
from teach.logger import create_logger

from agent.agent_neural_symbolic import NeuralSymbolicAgent
from agent.agent_oracle_navigator import TEAChAgentOracleNavigator

logger = create_logger(__name__)


class NeuralSymbolicModel(TeachModel):
    """
    Wrapper around Subgoal Aware ET Model for inference
    """

    def __init__(self, process_index: int, num_processes: int, model_args: List[str]):
        """Constructor

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of processes launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--eval_name', type=str, )
        parser.add_argument("--eval_device", type=str, default="cuda", help="cpu/cuda")
        parser.add_argument("--gpu_number", type=int, default=1)

        parser.add_argument("--use_teach_logger", action="store_true")
        parser.add_argument("--log_dir", type=str, default="log/")

        parser.add_argument("--depth_model_path", type=str, required=True)
        parser.add_argument("--panoptic_model_path", type=str, required=True)
        parser.add_argument("--panoptic_model_config", type=str, required=True)
        parser.add_argument("--state_estimator_path", type=str, required=True)
        parser.add_argument("--subgoal_predictor_path", type=str, required=True)
        parser.add_argument("--subgoal_predictor_ckpt", type=str, required=True)
        parser.add_argument("--fastdownward_path", type=str, required=True)
        parser.add_argument("--pddl_problem_save_dir", type=str, required=True)
        parser.add_argument("--meta_save_dir", type=str)
        parser.add_argument("--pddl_domain_file", type=str, required=True)
        parser.add_argument("--corenlp_dir", type=str)
        parser.add_argument("--word_embedding_model", type=str)
        parser.add_argument("--gt_subgoals_file", type=str, help="file path to load ground truth subgoals")


        parser.add_argument(
            "--max_steps", type=int, default=300
        )  # Simbot maximum: 1000

        parser.add_argument("--max_allowed_backtracks", type=int, default=5)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--ithor_assets_dir", type=str, default="./ithor_assets/")
        parser.add_argument("--plot", action="store_true")
        parser.add_argument("--save_meta_data", action="store_true")
        parser.add_argument(
            "--pause_at",
            type=str,
            choices=["interaction", "each_step", "no_pause"],
            default="no_pause",
        )

        parser.add_argument("--hlsm_use_gt_seg", action="store_true")
        parser.add_argument("--hlsm_use_gt_obj_det", action="store_true")
        parser.add_argument("--hlsm_use_gt_depth", action="store_true")
        parser.add_argument("--hlsm_use_gt_pose", action="store_true")
        parser.add_argument("--hlsm_use_gt_inventory", action="store_true")
        parser.add_argument("--use_gt_subgoals", action="store_true")
        parser.add_argument("--obj_det_conf_threshold", type=float, default=0.7)
        parser.add_argument("--state_det_conf_threshold", type=float, default=0.8)

        parser.add_argument("--agent_type", type=str, default='pddl', choices=["pddl", "fsm"])
        parser.add_argument("--nlu_spatial_threshold", type=float, default=0.7)
        parser.add_argument("--disable_replan", action="store_true")
        parser.add_argument("--disable_search_belief", action="store_true")
        parser.add_argument("--disable_scene_pruning", action="store_true")
        parser.add_argument("--last_subgoal_only", action="store_true")

        args = parser.parse_args(model_args)
        self.args = args

        logger_ = logger if args.use_teach_logger else None
        # self.agent = TEAChAgentSubgoalAwareET(args, process_index, logger_)
        self.agent = NeuralSymbolicAgent(args, process_index, logger_)
        # self.agent = [('Turn Left', [0.054155584876669605, 0.7492177499700433]), ('Forward', [0.9998105598199051, 0.10992540400328288]), ('Forward', [0.6975331519097145, 0.6378375606099822]), ('Forward', [0.53945742193972, 0.26767026136848615]), ('Forward', [0.9102138514183801, 0.9787085184349054]), ('Turn Right', [0.9099488776107348, 0.2019757216366862]), ('Look Up', [0.599845870586368, 0.08121480713053775]), ('Stop', [0.20826225637030937, 0.25515511356014076])]
        # self.idx = -1

    def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):
        edh_history_images = [np.asarray(img) for img in edh_history_images]
        self.agent.start_new_edh(edh_instance, edh_history_images, edh_name, env=None)
        return True

    # def get_next_action(
    #     self, img, edh_instance, prev_action, img_name=None, edh_name=None
    # ):
    #     self.idx += 1
    #     return self.agent[self.idx]

    def get_next_action(
        self, img, edh_instance, prev_action, img_name=None, edh_name=None
    ):
        """
        Sample function producing random actions at every time step. When running model inference, a model should be
        called in this function instead.
        :param img: PIL Image containing agent's egocentric image
        :param edh_instance: EDH instance
        :param prev_action: One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values
        from a previous call of get_next_action
        :param img_name: image file name
        :param edh_name: EDH instance file name
        :return action: An action name from all_agent_actions
        :return obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1) indicating an object in the image;
        The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the agent's egocentric image, selects
        an object in a 10x10 pixel patch around the pixel indicated by the coordinate if the desired action can be
        performed on it, and executes the action in AI2-THOR.
        """
        return self.agent.step(np.asarray(img))

    def start_new_edh_instance_cheating(
        self, edh_instance, edh_history_images, edh_name=None, env=None, history_events=None
    ):
        # print("SHAPE ================>", np.asarray(edh_history_images[0]).shape)
        edh_history_images = [np.asarray(img) for img in edh_history_images]
        if history_events is not None:
            assert len(edh_history_images) == len(history_events), "%d %d" % (len(edh_history_images), len(history_events))
        self.agent.start_new_edh(edh_instance, edh_history_images, edh_name, env, history_events)
        return True
