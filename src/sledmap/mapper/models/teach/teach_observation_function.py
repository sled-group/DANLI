from typing import Union, Optional
import torch
import torchvision

from sledmap.mapper.utils.base_cls import ObservationFunction

from sledmap.mapper.env.teach.teach_observation import TeachObservation
from sledmap.mapper.models.teach.spatial_state_repr import TeachSpatialStateRepr

from sledmap.mapper.models.alfred.projection.image_to_voxels import ImageToVoxels
from sledmap.mapper.models.alfred.projection.voxel_3d_observability import Voxel3DObservability
from sledmap.mapper.models.alfred.voxel_grid import VoxelGrid

from sledmap.mapper.ops.depth_estimate import DepthEstimate

from sledmap.mapper.utils.viz import show_image, show_image_with_bbox_and_point
from sledmap.mapper.flags import GLOBAL_VIZ, BIG_TRACE

import sledmap.mapper.utils.render3d as r3d

from sledmap.mapper.parameters import Hyperparams

VIZ_OBSERVATION = GLOBAL_VIZ


class TeachObservationFunction(ObservationFunction):
    def __init__(self, params: Optional[Hyperparams] = None):
        super().__init__()
        self.image_to_voxels = ImageToVoxels()
        self.voxel_3d_observability = Voxel3DObservability()
        self.trace = {}

        if params is None:
            params = Hyperparams({"use_mle_depth": False})

        self.use_mle_depth = params.get("use_mle_depth", False)
        print(f"USING MLE DEPTH: {self.use_mle_depth}")

        # if VIZ_OBSERVATION:
        #     import open3d as o3d
        #     self.voxel_grid_visualizer = o3d.visualization.Visualizer()
        #     self.voxel_grid_visualizer.create_window(window_name='Voxel Map', width=852, height=480)
        #     r3d.initialize_visualizer_view_angle(self.voxel_grid_visualizer, angle=0.0)

    def get_trace(self, device):
        for k, v in self.trace.items():
            if isinstance(v, list):
                self.trace[k] = [x.to(device) for x in v]
            else:
                self.trace[k] = v.to(device)
        return self.trace

    def clear_trace(self):
        self.trace = {}

    def forward(self, observation: TeachObservation, prev_state: Union[TeachSpatialStateRepr, None]) -> TeachSpatialStateRepr:
        scene_image = observation.semantic_image
        depth_image = observation.depth_image
        hfov_deg = observation.hfov_deg
        extrinsics4f = observation.pose

        if isinstance(depth_image, DepthEstimate):
            # If the system is currently trying to achieve a high-level goal, make sure to always project points
            # that correspond to the high-level goal into the voxel map
            proj_depth = depth_image.mle() if self.use_mle_depth else depth_image.get_trustworthy_depth()
            obs_depth = depth_image.mle()
        else:
            proj_depth = obs_depth = depth_image

        voxel_grid : VoxelGrid = self.image_to_voxels(scene_image, proj_depth, extrinsics4f, hfov_deg, min_depth=0.8)
        voxel_observability_grid, voxel_ray_depths = self.voxel_3d_observability(voxel_grid, extrinsics4f, proj_depth, hfov_deg) # modify obs_depth to proj_depth

        inventory_vector = observation.inventory_vector

        # Override data with depths to debug the mapping between depth image and voxel grid
        REPLACE_SEMANTICS_WITH_DEPTH_DEBUG = False
        if REPLACE_SEMANTICS_WITH_DEPTH_DEBUG:
            voxel_grid.data = voxel_ray_depths.repeat((1, 3, 1, 1, 1)) / 5.0

        # First observation in a sequence
        if prev_state is None:
            next_state = TeachSpatialStateRepr(voxel_grid, voxel_observability_grid, inventory_vector, observation)
            if VIZ_OBSERVATION:
                rgb_voxelgrid = next_state.make_rgb_voxelgrid(observability=False)
                # r3d.update_and_render_new_voxel_grid(self.voxel_grid_visualizer, new_voxel_grid=rgb_voxelgrid, first_time=True)

        # Subsequent observation - integrate it
        else:
            next_voxel_grid_data = prev_state.data.data * (1 - voxel_observability_grid.data) + voxel_grid.data * voxel_observability_grid.data
            next_voxel_grid_occupancy = prev_state.data.occupancy * (1 - voxel_observability_grid.data) + voxel_grid.occupancy * voxel_observability_grid.data
            next_observability_mask = torch.max(voxel_observability_grid.data, prev_state.obs_mask.data)

            #next_observability_voxels = VoxelGrid(next_observability_mask, next_voxel_grid_occupancy, voxel_grid.voxel_size, voxel_grid.origin)
            next_voxel_grid             = VoxelGrid(next_voxel_grid_data,    next_voxel_grid_occupancy, voxel_grid.voxel_size, voxel_grid.origin)
            next_observability_voxels   = VoxelGrid(next_observability_mask, next_observability_mask,   voxel_grid.voxel_size, voxel_grid.origin)
            next_state = TeachSpatialStateRepr(next_voxel_grid, next_observability_voxels, inventory_vector, observation)
            

            if VIZ_OBSERVATION:
                # show_image(observation.represent_as_image(semantic=True, rgb=False, depth=True, horizontal=False)[0].detach().cpu(), "Observation", scale=1, waitkey=1)
                # show_image(proj_depth[0], "Projected Depth", scale=1, waitkey=None)
                vis_observation = observation.represent_as_image(semantic=True, rgb=True, depth=False, horizontal=False)[0].detach().cpu()
                show_image_with_bbox_and_point(
                    vis_observation, 
                    observation.object_detections, 
                    None,
                    "Observation with BBox", 
                    waitkey=None
                )
                show_image(next_state.represent_as_image_with_inventory(animate=False,
                                                                        observability=False,
                                                                        topdown2d=True,
                                                                        inventory=False), "Top-down 2D Map", scale=16, waitkey=None)
                # rgb_voxelgrid = next_state.make_rgb_voxelgrid(observability=False)
                # r3d.update_and_render_new_voxel_grid(self.voxel_grid_visualizer, new_voxel_grid=rgb_voxelgrid)
                # r3d.view_voxel_grid(rgb_voxelgrid)
        
        next_state.annotations['observed_repr'] = TeachSpatialStateRepr(voxel_grid, voxel_observability_grid, inventory_vector, None)

        # if BIG_TRACE:
            # Don't save observations here: they're already saved elsewhere
            # self.trace["state_repr"] = TeachSpatialStateRepr(next_state.data, next_state.obs_mask, next_state.inventory_vector, observation)
            # self.trace["observed_repr"] = TeachSpatialStateRepr(voxel_grid, voxel_observability_grid, inventory_vector, None)
            # if not observation.extra_rgb_frames:
            #     self.trace["extra_rgb_frames"] = [observation.rgb_image]
            # else:
            #     self.trace["extra_rgb_frames"] = observation.extra_rgb_frames
        if "agent_trajectory" not in self.trace:
            self.trace["agent_trajectory"] = [next_state.get_pos_xyz_vx()]
        else:
            self.trace["agent_trajectory"] += [next_state.get_pos_xyz_vx()]
        next_state.annotations['agent_trajectory'] = self.trace["agent_trajectory"]
        

        current_visible_voxels = voxel_observability_grid
        # get the 3D masks in voxel for each 2D object detections
        if observation.object_detections:
            # object instance visibility should be computed without masking out close voxels
            voxel_grid_fully_visible  = self.image_to_voxels(scene_image, obs_depth, extrinsics4f, hfov_deg, min_depth=0.1)
            current_visible_voxels, voxel_ray_depths = self.voxel_3d_observability(voxel_grid_fully_visible, extrinsics4f, proj_depth, hfov_deg)

            instance_2d_masks = [ins.instance_mask for ins in observation.object_detections]
            instance_2d_masks = torch.stack(instance_2d_masks, dim=0).unsqueeze(1).half()  # (N, 1, H, W)
            instance_3d_masks_in_voxel = self.image_to_voxels(instance_2d_masks, obs_depth, extrinsics4f, hfov_deg, min_depth=0.1)
            observation.assign_3d_voxel_masks(instance_3d_masks_in_voxel)

        # Debug 3D voxel mask
        # idxxx = 5
        # if prev_state is not None:
        #     print(observation.object_detections[idxxx].tmp_unique_id)
        #     print(instance_2d_masks[idxxx:idxxx+1].shape)
        #     mask_3d_in_voxel = self.image_to_voxels(instance_2d_masks[idxxx:idxxx+1], proj_depth, extrinsics4f, hfov_deg)
        #     rgb_data = 1 - mask_3d_in_voxel.data.repeat((1, 3, 1, 1, 1))
        #     occupancy = mask_3d_in_voxel.occupancy
        #     rgb_voxelgrid = VoxelGrid(rgb_data, occupancy, mask_3d_in_voxel.voxel_size, mask_3d_in_voxel.origin)
        #     r3d.update_and_render_new_voxel_grid(self.voxel_grid_visualizer, new_voxel_grid=rgb_voxelgrid)
        #     r3d.view_voxel_grid(rgb_voxelgrid)


        # TODO: Update the inventory vector if necessary
        return next_state, current_visible_voxels
        