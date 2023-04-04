from typing import Union
import torch

from sledmap.mapper.utils.base_cls import ObservationFunction

from sledmap.mapper.env.alfred.alfred_observation import AlfredObservation
from sledmap.mapper.env.alfred.alfred_subgoal import AlfredSubgoal
from sledmap.mapper.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr

from sledmap.mapper.models.alfred.projection.image_to_voxels import ImageToVoxels
from sledmap.mapper.models.alfred.projection.voxel_3d_observability import Voxel3DObservability
from sledmap.mapper.models.alfred.voxel_grid import VoxelGrid

from sledmap.mapper.ops.depth_estimate import DepthEstimate

from sledmap.mapper.utils.viz import show_image
from sledmap.mapper.flags import GLOBAL_VIZ, BIG_TRACE

import sledmap.mapper.utils.render3d as r3d

from sledmap.mapper.parameters import Hyperparams

VIZ_OBSERVATION = GLOBAL_VIZ


class HlsmObservationFunction(ObservationFunction):
    def __init__(self, params: Hyperparams):
        super().__init__()
        self.image_to_voxels = ImageToVoxels()
        self.voxel_3d_observability = Voxel3DObservability()
        self.trace = {}

        self.use_mle_depth = params.get("use_mle_depth", False)
        print(f"USING MLE DEPTH: {self.use_mle_depth}")


        if VIZ_OBSERVATION:
            import open3d as o3d
            self.voxel_grid_visualizer = o3d.visualization.Visualizer()
            self.voxel_grid_visualizer.create_window(window_name='Voxel Map', width=852, height=480)
            r3d.initialize_visualizer_view_angle(self.voxel_grid_visualizer)

    def get_trace(self, device):
        for k, v in self.trace.items():
            if isinstance(v, list):
                self.trace[k] = [x.to(device) for x in v]
            else:
                self.trace[k] = v.to(device)
        return self.trace

    def clear_trace(self):
        self.trace = {}

    def forward(self, observation: AlfredObservation, prev_state: Union[AlfredSpatialStateRepr, None], goal: Union[AlfredSubgoal, None]) -> AlfredSpatialStateRepr:
        scene_image = observation.semantic_image
        depth_image = observation.depth_image
        hfov_deg = observation.hfov_deg
        extrinsics4f = observation.pose

        if isinstance(depth_image, DepthEstimate):
            # If the system is currently trying to achieve a high-level goal, make sure to always project points
            # that correspond to the high-level goal into the voxel map
            if goal is not None:
                arg_obj_id = goal.arg_intid()
                include_mask = scene_image[:, arg_obj_id:arg_obj_id+1] > 0.5
            else:
                include_mask = None
            if self.use_mle_depth:
                proj_depth = depth_image.mle()
            else:
                proj_depth = depth_image.get_trustworthy_depth(include_mask=include_mask)
            obs_depth = depth_image.mle()
        else:
            proj_depth = depth_image
            obs_depth = depth_image

        voxel_grid : VoxelGrid = self.image_to_voxels(scene_image, proj_depth, extrinsics4f, hfov_deg, mark_agent=True)
        voxel_observability_grid, voxel_ray_depths = self.voxel_3d_observability(voxel_grid, extrinsics4f, obs_depth, hfov_deg)

        inventory_vector = observation.inventory_vector

        # Override data with depths to debug the mapping between depth image and voxel grid
        REPLACE_SEMANTICS_WITH_DEPTH_DEBUG = False
        if REPLACE_SEMANTICS_WITH_DEPTH_DEBUG:
            voxel_grid.data = voxel_ray_depths.repeat((1, 3, 1, 1, 1)) / 5.0

        # First observation in a sequence
        if prev_state is None:
            next_state = AlfredSpatialStateRepr(voxel_grid, voxel_observability_grid, inventory_vector, observation)
            if VIZ_OBSERVATION:
                rgb_voxelgrid = next_state.make_rgb_voxelgrid(observability=False)
                r3d.update_and_render_new_voxel_grid(self.voxel_grid_visualizer, new_voxel_grid=rgb_voxelgrid, first_time=True)

        # Subsequent observation - integrate it
        else:
            next_voxel_grid_data = prev_state.data.data * (1 - voxel_observability_grid.data) + voxel_grid.data * voxel_observability_grid.data
            next_voxel_grid_occupancy = prev_state.data.occupancy * (1 - voxel_observability_grid.data) + voxel_grid.occupancy * voxel_observability_grid.data
            next_observability_mask = torch.max(voxel_observability_grid.data, prev_state.obs_mask.data)

            #next_observability_voxels = VoxelGrid(next_observability_mask, next_voxel_grid_occupancy, voxel_grid.voxel_size, voxel_grid.origin)
            next_voxel_grid             = VoxelGrid(next_voxel_grid_data,    next_voxel_grid_occupancy, voxel_grid.voxel_size, voxel_grid.origin)
            next_observability_voxels   = VoxelGrid(next_observability_mask, next_observability_mask,   voxel_grid.voxel_size, voxel_grid.origin)
            next_state = AlfredSpatialStateRepr(next_voxel_grid, next_observability_voxels, inventory_vector, observation)

            if VIZ_OBSERVATION:
                show_image(observation.represent_as_image(semantic=True, rgb=True, depth=True, horizontal=False)[0].detach().cpu(), "Observation", scale=1, waitkey=1)
                show_image(proj_depth[0], "Projected Depth", scale=1, waitkey=1)
                show_image(next_state.represent_as_image_with_inventory(animate=False,
                                                                        observability=False,
                                                                        topdown2d=True,
                                                                        inventory=False), "Top-down 2D Map", scale=8, waitkey=1)
                rgb_voxelgrid = next_state.make_rgb_voxelgrid(observability=False)
                r3d.update_and_render_new_voxel_grid(self.voxel_grid_visualizer, new_voxel_grid=rgb_voxelgrid)




        if BIG_TRACE:
            # Don't save observations here: they're already saved elsewhere
            self.trace["state_repr"] = AlfredSpatialStateRepr(next_state.data, next_state.obs_mask, next_state.inventory_vector, observation)
            self.trace["observed_repr"] = AlfredSpatialStateRepr(voxel_grid, voxel_observability_grid, inventory_vector, None)
            if not observation.extra_rgb_frames:
                self.trace["extra_rgb_frames"] = [observation.rgb_image]
            else:
                self.trace["extra_rgb_frames"] = observation.extra_rgb_frames


        # TODO: Update the inventory vector if necessary
        return next_state
