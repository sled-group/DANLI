from dataclasses import dataclass
from typing import Union, List, Dict, Tuple, Optional
import os
import json
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
from sledmap.mapper.utils.viz import show_image, show_image_with_bbox_and_point
import sledmap.mapper.utils.render3d as r3d


class NeuralSymbolicAgentState:
    def __init__(self,
            observation: TeachObservation,
            spatial_state_repr: TeachSpatialStateRepr,
            symbolic_world_repr: TeachSymbolicWorld,
            last_action: TeachAction,
            last_action_failed: bool) -> None:
        self.observation = observation
        self.spatial_state_repr = spatial_state_repr
        self.symbolic_world_repr = symbolic_world_repr
        self.inventory_instance_ids = self.symbolic_world_repr.inventory_instance_ids
        self.last_action=last_action
        self.last_action_failed = last_action_failed
        self.spatial_relations = {}

    
    def get_2D_detection_of_instance(self, instance_id):
        return self.symbolic_world_repr.get_2D_detection_of_instance(instance_id)
    
    def get_3D_instance_id_of_detection(self, detection):
        return self.symbolic_world_repr.instance_mapping_2to3[detection.tmp_unique_id]

    def get_instance_by_id(self, instance_id):
        return self.symbolic_world_repr.get_instance(instance_id)

    def get_all_instances(self):
        return self.symbolic_world_repr.get_all_instances()
    
    def get_all_instance_ids(self):
        return self.symbolic_world_repr.get_all_instance_ids()
    
    def get_symbolic_state_dict(self):
        return self.symbolic_world_repr.get_symbolic_state_dict()

    def get_agent_pos_m(self):
        return self.spatial_state_repr.get_agent_pos_m()
    
    def set_spatial_relations(self, spatial_relations):
        self.spatial_relations = spatial_relations
    
    def get_search_hints(self, search_types):
        return self.symbolic_world_repr.get_hint_list(search_types, self.spatial_relations)
    
    def get_instances_with_state_change(self):
        return self.symbolic_world_repr.instances_with_state_change
    
    def get_instance_match_log_info(self):
        return self.symbolic_world_repr.instance_match_log_info
    
    def visualize(self, action=None, waitkey=None, voxel_grid_visualizer=None, interactive=False, save_dir=None, img_idx=0):
        # "/data/simbot/teach-eval/neural_symbolic/viz/images_gt"

        images = {}
        observation = show_image_with_bbox_and_point(
            self.observation.represent_as_image(semantic=True, rgb=True, depth=False, horizontal=True)[0].detach().cpu(), 
            self.observation.object_detections, 
            None, # action.interaction_point if action is not None else None,
            "observation", 
            waitkey=waitkey
        )
        images['observation'] = observation

        if "navigation_map" in self.spatial_state_repr.annotations:
            images['navigation_map'] = self.spatial_state_repr.annotations["navigation_map"]
        
        
        # show_image(self.spatial_state_repr.represent_as_image_with_inventory(
        #     animate=False,
        #     observability=False,
        #     topdown2d=True,
        #     inventory=False), "Top-down 2D Map", scale=16, waitkey=waitkey)

        if voxel_grid_visualizer is not None:
            rgb_voxelgrid = self.spatial_state_repr.annotate_with_agent_and_goal()
            # r3d.update_and_render_new_voxel_grid(voxel_grid_visualizer, new_voxel_grid=rgb_voxelgrid)
            geometry, centroid = r3d.voxelgrid_to_geometry(rgb_voxelgrid)
            topdown_map = r3d.render_geometries(geometry, camera_angle='default')
            show_image(topdown_map, "topdown_map_3d", scale=1, waitkey=None)
            images['topdown_map'] = topdown_map[:, :, [2, 1, 0]]

            sideview_map = r3d.render_geometries(geometry, camera_angle='side2')
            show_image(sideview_map, "sideview_map_3d", scale=1, waitkey=None)
            images['sideview_map'] = sideview_map[:, :, [2, 1, 0]]
            
            if interactive:
                r3d.view_voxel_grid(rgb_voxelgrid)
        

        if save_dir:
            for img_type, img in images.items():
                save_dir_img_type = os.path.join(save_dir, img_type)
                if not os.path.exists(save_dir_img_type):
                    os.makedirs(save_dir_img_type)
                fn = os.path.join(save_dir_img_type, '%d.jpg' %(img_idx))
                cv2.imwrite(fn, cv2.convertScaleAbs(img, alpha=(255.0)))