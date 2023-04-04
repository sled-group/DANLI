# SPDX-License-Identifier: MIT-0
import argparse
import numpy as np
from typing import List
from attrdict import AttrDict

from teach.inference.teach_model import TeachModel
from teach.logger import create_logger
from agent.agent_SAET import TEAChAgentSubgoalAwareET
from agent.agent_oracle_navigator import TEAChAgentOracleNavigator

logger = create_logger(__name__)


class SAETModel(TeachModel):
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
        parser.add_argument("--eval_device", type=str, default="cuda", help="cpu/cuda")
        parser.add_argument("--gpu_number", type=int, default=1)

        parser.add_argument("--use_teach_logger", action="store_true")
        parser.add_argument("--log_dir", type=str, default="log/")

        parser.add_argument("--controller_dir", type=str, required=True)
        parser.add_argument("--controller_ckpt", type=str, required=True)
        parser.add_argument("--navigator_dir", type=str, required=True)
        parser.add_argument("--navigator_ckpt", type=str, required=True)
        parser.add_argument(
            "--objdet_dir",
            type=str,
            default="/home/ubuntu/ehsd_dev/models/teach_mrcnn/",
        )
        parser.add_argument("--objdet_ckpt", type=str, default="model_final.pth")

        parser.add_argument("--visual_enc_archiecture", type=str, default="fasterrcnn")
        parser.add_argument("--visual_enc_compress_type", type=str, default="4x")

        parser.add_argument("--dec_keep_topk_prob", type=int, default=10)
        parser.add_argument("--dec_intent_decoding", type=str, default="greedy")
        parser.add_argument("--dec_max_intent_length", type=int, default=20)

        parser.add_argument("--odet_conf_thresh", type=float, default=0.5)
        parser.add_argument("--odet_batch_size", type=int, default=16)
        parser.add_argument("--max_allowed_backtracks", type=int, default=5)
        parser.add_argument("--eval_debug", type=bool, default=True)
        parser.add_argument("--ithor_assets_dir", type=str, default="./ithor_assets/")
        parser.add_argument("--plot_and_pause", action="store_true")
        parser.add_argument("--use_oracle_navigator", action="store_true")

        # args = {
        #     # "model_profile": {
        #     #     "controller": {
        #     #         "exp_dir": "/gpfs/accounts/chaijy_root/chaijy1/zhangyic/project/EAI/TEACh-exp/2022-03-23-20-43-01-edh",
        #     #         "ckpt": "ckpt/ckpt-epoch=04-val_unseen_total_loss=4.89.ckpt",
        #     #     },
        #     #     "navigator": {
        #     #         "exp_dir": "/gpfs/accounts/chaijy_root/chaijy1/zhangyic/project/EAI/TEACh-exp/2022-01-27-04-43-43-navi",
        #     #         "ckpt": "ckpt/ckpt-epoch=07-val_unseen_total_loss=1.12.ckpt",
        #     #     },
        #     #     "visual_processor": {
        #     #         "visual_archi": "fasterrcnn",
        #     #         "compress_type": "4x",
        #     #     },
        #     #     "object_detector": {
        #     #         "exp_dir": "/gpfs/accounts/chaijy_root/chaijy1/zhangyic/project/EAI/teach_mrcnn/",
        #     #         "ckpt": "model_final.pth",
        #     #     },
        #     # },
        #     # "model_profile": {
        #     #     "controller": {
        #     #         # "exp_dir": "/data/simbot/teach-exp/2022-02-09-21-04-47-edh",  # non-intent
        #     #         # "ckpt": "ckpt/ckpt-epoch=02-val_unseen_total_loss=5.02.ckpt",
        #     #         "exp_dir": "/data/simbot/teach-exp/2022-03-23-22-28-47-edh-intent",  # add intent
        #     #         "ckpt": "ckpt-epoch=26-avg_accu=0.7775.ckpt",
        #     #     },
        #     #     "navigator": {
        #     #         "exp_dir": "/data/simbot/teach-exp/2022-01-27-04-43-43-navi",
        #     #         "ckpt": "ckpt/ckpt-epoch=07-val_unseen_total_loss=1.12.ckpt",
        #     #     },
        #     #     "visual_processor": {
        #     #         "visual_archi": "fasterrcnn",
        #     #         "compress_type": "4x",
        #     #     },
        #     #     "object_detector": {
        #     #         "exp_dir": "/data/simbot/teach-exp/teach_mrcnn/",
        #     #         "ckpt": "model_final.pth",
        #     #     },
        #     # },
        #     "model_profile": {
        #         "controller": {
        #             # "exp_dir": "/data/simbot/teach-exp/2022-02-09-21-04-47-edh",  # non-intent
        #             # "ckpt": "ckpt/ckpt-epoch=02-val_unseen_total_loss=5.02.ckpt",
        #             "exp_dir": "/home/ubuntu/ehsd_dev/models/2022-03-23-22-28-47-edh-intent",  # add intent
        #             "ckpt": "ckpt-epoch=26-avg_accu=0.7775.ckpt",
        #         },
        #         "navigator": {
        #             "exp_dir": "/home/ubuntu/ehsd_dev/models/2022-01-27-04-43-43-navi",
        #             "ckpt": "ckpt/ckpt-epoch=07-val_unseen_total_loss=1.12.ckpt",
        #         },
        #         "visual_processor": {
        #             "visual_archi": "fasterrcnn",
        #             "compress_type": "4x",
        #         },
        #         "object_detector": {
        #             "exp_dir": "/home/ubuntu/ehsd_dev/models/teach_mrcnn/",
        #             "ckpt": "model_final.pth",
        #         },
        #     },
        #     # which split to use ('train', 'valid_seen', 'valid_unseen')
        #     "split": "valid_seen",
        #     # evalulation devices
        #     "eval_device": "cuda",
        #     "gpu_number": 8,
        #     # sequence decoding strategy for intent decoding
        #     "dec_params": {
        #         "keep_topk_prob": 10,
        #         "intent_decoding": "greedy",
        #         "max_intent_length": 20,
        #     },
        #     # object detection confidence threshold
        #     "odet_conf_thresh": 0.5,
        #     # object detection batch size
        #     "odet_batch_size": 16,
        #     "eval_debug": True,
        #     "max_allowed_backtracks": 5,
        #     # "ithor_assets_dir": "/home/zhangyic/project/EAI/ehsd_dev/ithor_assets/",
        #     "ithor_assets_dir": "./ithor_assets/",
        #     "plot_and_pause": False,
        # }

        args = parser.parse_args(model_args)
        self.args = args

        logger_ = logger if args.use_teach_logger else None
        if not args.use_oracle_navigator:
            self.agent = TEAChAgentSubgoalAwareET(args, process_index, logger_)
        else:
            self.agent = TEAChAgentOracleNavigator(args, process_index, logger_)
        # self.agent = [('Turn Left', [0.054155584876669605, 0.7492177499700433]), ('Forward', [0.9998105598199051, 0.10992540400328288]), ('Forward', [0.6975331519097145, 0.6378375606099822]), ('Forward', [0.53945742193972, 0.26767026136848615]), ('Forward', [0.9102138514183801, 0.9787085184349054]), ('Turn Right', [0.9099488776107348, 0.2019757216366862]), ('Look Up', [0.599845870586368, 0.08121480713053775]), ('Stop', [0.20826225637030937, 0.25515511356014076])]
        # self.idx = -1

    def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):
        self.agent.reset(edh_instance, edh_history_images, edh_name)
        # self.idx = -1
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
        agent_action = self.agent.step(img)

        action = agent_action.name[0]
        if agent_action.interaction_point is not None:
            # obj_relative_coord = agent_action.interaction_point

            # Note: have to use y,x instead of x,y
            obj_relative_coord = [
                agent_action.interaction_point[1] / 900,
                agent_action.interaction_point[0] / 900,
            ]
            # obj_relative_coord = [i / 900 for i in agent_action.interaction_point]
        else:
            obj_relative_coord = None

        return action, obj_relative_coord

    def start_new_edh_instance_cheating(
        self, edh_instance, edh_history_images, edh_name=None, env=None
    ):
        # self.idx = -1
        self.agent.reset(edh_instance, edh_history_images, edh_name, env)
        return True
