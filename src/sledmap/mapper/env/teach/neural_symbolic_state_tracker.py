import os
import json
from typing import Union, List, Dict, Tuple, Optional
import torch
import numpy as np

import cv2
from PIL.Image import Image

from definitions.teach_objects import ithor_oid_to_object_class

from sledmap.mapper.env.teach.teach_action import TeachAction, InteractionActions
from sledmap.mapper.env.teach.teach_observation import TeachObservation
from sledmap.mapper.models.teach.spatial_state_repr import TeachSpatialStateRepr
from sledmap.mapper.env.teach.teach_observation import TeachObservation

from sledmap.mapper.env.teach.state_tracker import StateTracker
from sledmap.mapper.models.teach.teach_observation_function import (
    TeachObservationFunction,
)
from sledmap.mapper.models.teach.symbolic_world_repr import TeachSymbolicWorld
from sledmap.mapper.models.teach.teach_depth_model import TeachDepthModel
from sledmap.mapper.models.teach.teach_perception_model_new import PerceptionModel
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState
import sledmap.mapper.utils.render3d as r3d
import open3d as o3d

class NeuralSymbolicAgentStateTracker:
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device
        self.logger = logger
        self.log_func = logger.info if logger is not None else print
        # self.log_func = lambda x:x

        self._build(build_3d_visualizer=args.plot)

    def _build(self, build_3d_visualizer):
        
        if not self.args.hlsm_use_gt_depth:
            self.depth_model = TeachDepthModel()
            self.depth_model.load_state_dict(
                torch.load(self.args.depth_model_path, map_location=self.device)
            )
            self.depth_model.to(self.device)
            self.depth_model.eval()
        else:
            self.depth_model = None

        # self.perception_model = PerceptionModel(
        #     self.args.panoptic_model_path, self.args.state_estimator_path, self.device# maskrcnn_ckpt=self.args.mask_rcnn_path,
        # )
        if not self.args.hlsm_use_gt_seg and not self.args.hlsm_use_gt_obj_det:
            self.perception_model = PerceptionModel(
                self.args.panoptic_model_config, self.args.panoptic_model_path, self.args.state_estimator_path, self.device# maskrcnn_ckpt=self.args.mask_rcnn_path,
            )
            self.perception_model.to(self.device)
            self.perception_model.eval()
        else:
            self.perception_model = None

        self.state_tracker = StateTracker(
            args=self.args,
            device=self.device,
            seg_model=self.perception_model,
            depth_model=self.depth_model,
            logger=self.logger,
        )
        self.observation_function = TeachObservationFunction()
        self.symbolic_world_repr = TeachSymbolicWorld(self.logger)

        if build_3d_visualizer:
            self.voxel_grid_visualizer = o3d.visualization.Visualizer()
            self.voxel_grid_visualizer.create_window(window_name='Voxel Map', width=1920, height=1080)

    def reset(self, env=None):
        self.state_tracker.reset(event=env.last_event if env is not None else None)
        self.spatial_state_repr = None
        self.symbolic_world_repr.reset()
        self.observation_function.clear_trace()

    def step(
        self, frame: np.ndarray, last_action: TeachAction, event=None, verbose=False
    ) -> NeuralSymbolicAgentState:

        print("=============================> Observation Step")

        self.log_func("-------- Make Observation --------")
        self.state_tracker.update_state(frame=frame, event=event)
        observation = self.state_tracker.get_observation()
        agent_pose_m = self.state_tracker.get_agent_pos(tensorize=True)

        if not self.state_tracker.last_action_failed:
            self.log_func("-------- Update Spatial State Representation --------")
            self.spatial_state_repr, current_visible_voxels = self.observation_function(
                observation, self.spatial_state_repr
            )

            self.log_func("-------- Update Symbolic World Representation --------")
            self.symbolic_world_repr = self.symbolic_world_repr.update(
                voxel_observability=current_visible_voxels,
                object_detections=observation.object_detections,
                last_succeed_action=last_action,
                agent_pose_m=agent_pose_m,
                verbose=verbose
            )
                  
            # from pprint import pprint
            # print('agent position:', agent_pose_m)
            # print("-------------- all existing objects ----------------")
            # pprint(self.symbolic_world_repr.get_all_instances())

        current_state = NeuralSymbolicAgentState(
            observation=observation,
            spatial_state_repr=self.spatial_state_repr,
            symbolic_world_repr=self.symbolic_world_repr,
            last_action=last_action,
            last_action_failed=self.state_tracker.last_action_failed,
        )
        return current_state

    def replay_trajectory(
        self,
        observations: List[np.ndarray],
        edh_driver_action_history: List[Dict],
        events: Optional[List] = None,
        verbose=False,
        visualize=False,
        meta_data=None,
        meta_save_dir=None,
    ) -> NeuralSymbolicAgentState:

        replay_step_num = 0
        current_state = None
        last_action = TeachAction.create_empty_action()
        for idx, edh_action in enumerate(edh_driver_action_history):
            self.log_func("-------- Replay Step: %d --------" % idx)

            if not TeachAction.is_valid(edh_action["action_name"]) or (
                InteractionActions.has_action(edh_action["action_name"])
                and edh_action["oid"] is None
            ):
                self.log_func(
                    "[%r %r] is not a valid TEACh action. Skip it!"
                    % (edh_action["action_name"], edh_action["oid"])
                )
                continue

            frame = observations[idx]
            event = events[idx] if events is not None else None
            current_state = self.step(frame=frame, last_action=last_action, event=event, verbose=verbose)

            self.log_func("-------- Get the Next TeachAction --------")
            object_type = (
                None
                if edh_action["oid"] is None
                else ithor_oid_to_object_class(edh_action["oid"])
            )
            teach_api_action = {
                "action_name": edh_action["action_name"],
                "object_type": object_type,
                "oid": edh_action["oid"],
                "x": edh_action["x"],
                "y": edh_action["y"],
            }
            last_action_success = not last_action.is_failed()
            last_action_tmp = last_action
            last_action = TeachAction.create_from_teach_api(
                teach_api_action=teach_api_action,
                symbolic_world_repr=current_state.symbolic_world_repr,
            )
            self.state_tracker.log_action(last_action)

            if self.args.save_meta_data:
                symbolic_dict = current_state.get_symbolic_state_dict()
                instance_match_log_info = current_state.get_instance_match_log_info()
                symbolic_meta = {
                    'world_state': symbolic_dict,
                    'match_info': instance_match_log_info
                }
                save_dir = os.path.join(meta_save_dir, 'symbolic')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, '%d.json' %(replay_step_num)), 'w') as f:
                    json.dump(symbolic_meta, f, indent=2)

                meta_data['steps'].append({
                'img_idx': replay_step_num,
                'step_idx': -1,
                "time_start": edh_action["time_start"],
                'stage': 'replay',
                'status': None,
                'action_to_take': last_action.to_dict(),
                'last_action': last_action_tmp.to_dict(),
                'last_action_success': last_action_success,
            })

            if visualize:
                current_state.visualize(
                    action=last_action, 
                    waitkey=None,#0 if last_action.is_interaction() else None,
                    voxel_grid_visualizer=self.voxel_grid_visualizer,
                    save_dir=meta_save_dir if self.args.save_meta_data else None,
                    img_idx=replay_step_num,
                )
            
            replay_step_num += 1
            
        
        self.last_action = last_action
        self.log_func("-------- Replay Completed --------")
        return current_state, replay_step_num

    def log_action(self, action):
        self.state_tracker.log_action(action)