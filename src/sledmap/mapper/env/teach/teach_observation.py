from typing import Iterable, Union, List

import torch

import sledmap.mapper.env.teach.segmentation_definitions as segdef
from sledmap.mapper.utils.base_cls import Observation
from sledmap.mapper.env.teach.teach_object_instance import ObjectInstanceDetection2D 
from sledmap.mapper.models.teach.utils import voxel_idx_coord_to_meter_coord
from sledmap.mapper.models.alfred.voxel_grid import VoxelGrid

VISUALIZE_AUGMENTATIONS = False

class TeachObservation(Observation):
    def __init__(
            self,
            rgb_image: torch.tensor,
            depth_image: torch.tensor,
            semantic_image: torch.tensor,
            object_detections: List[ObjectInstanceDetection2D],
            inventory_vector: torch.tensor,
            pose: torch.tensor,
            hfov_deg: float,
            cam_horizon_deg: List[float]
        ):
        super().__init__()
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.semantic_image = semantic_image
        self.object_detections = object_detections
        self.inventory_vector = inventory_vector
        self.pose = pose
        self.hfov_deg = hfov_deg
        self.cam_horizon_deg = cam_horizon_deg

        # This is used only during test time. TODO: Make sure pose is correct and use it instead.
        self.agent_pos = None

    def __getitem__(self, item):
        rgb_image = self.rgb_image[item]
        depth_image = self.depth_image[item]
        semantic_image = self.semantic_image[item]
        object_detections = self.object_detections[item]
        inventory_vector = self.inventory_vector[item]
        pose = self.pose[item]
        hfov_deg = self.hfov_deg
        cam_horizon_deg = [self.cam_horizon_deg[item]]
        assert self.error_causing_action is None, "Cannot treat observations with error_causing_action as batches"
        return TeachObservation(
            rgb_image,
            depth_image,
            semantic_image,
            object_detections,
            inventory_vector,
            pose,
            hfov_deg,
            cam_horizon_deg,
        )

    def set_agent_pos(self, agent_pos):
        self.agent_pos = agent_pos

    def get_agent_pos(self, device="cpu"):
        # TODO: This is a temporary workaround. Figure it out and standardize it:
        if self.agent_pos is None:
            raise ValueError("Requesting agent_pos from observation, but set_agent_pos hasn't been called")
        return self.agent_pos.to(device)

    def get_depth_image(self):
        if isinstance(self.depth_image, torch.Tensor):
            depth_img = self.depth_image
        # depth image is a DepthEstimate
        else:
            depth_img = self.depth_image.get_trustworthy_depth()
        return depth_img

    def is_compressed(self):
        return self.semantic_image.shape[1] == 1

    def compress(self):
        # If semantic image is in a one-hot representation, convert to a more space-saving integer representation
        if not self.is_compressed():
            # TODO: This doesn't support anything bigger than 128!!
            self.semantic_image = self.semantic_image.type(torch.uint8).argmax(dim=1, keepdim=True)

    def uncompress(self):
        # If semantic image is in an integer representation, convert it to a one-hot representation
        if self.is_compressed():
            n_c = segdef.get_num_objects()
            rng = torch.arange(0, n_c, 1, device=self.semantic_image.device, dtype=torch.uint8)
            onehotrepr = (rng[None, :, None, None] == self.semantic_image).type(torch.uint8)
            self.semantic_image = onehotrepr


    def to(self, device) -> "TeachObservation":
        obs_o = TeachObservation(
            self.rgb_image.to(device),
            self.depth_image.to(device),
            self.semantic_image.to(device),
            [i.to(device) for i in self.object_detections],
            self.inventory_vector.to(device),
            self.pose.to(device),
            self.hfov_deg,
            self.cam_horizon_deg,
        )
        obs_o.agent_pos = self.agent_pos.to(device) if self.agent_pos is not None else None
        return obs_o

    @classmethod
    def collate(cls, observations: Iterable["TeachObservation"]) -> "TeachObservation":
        rgb_images = torch.cat([o.rgb_image for o in observations], dim=0)
        depth_images = torch.cat([o.depth_image for o in observations], dim=0)
        semantic_images = torch.cat([o.semantic_image for o in observations], dim=0)
        object_detectionss = [o.object_detections for o in observations]
        inventory_vectors = torch.cat([o.inventory_vector for o in observations], dim=0)
        poses = torch.cat([o.pose for o in observations], dim=0)
        hfov_deg = next(iter(observations)).hfov_deg
        cam_horizon_deg = [o.cam_horizon_deg[0] for o in observations]
        return TeachObservation(
            rgb_image=rgb_images,
            depth_image=depth_images,
            semantic_image=semantic_images,
            object_detections=object_detectionss,
            inventory_vector=inventory_vectors,
            pose=poses,
            hfov_deg=hfov_deg,
            cam_horizon_deg=cam_horizon_deg,
        )

    def represent_as_image(self, semantic=True, rgb=True, depth=True, horizontal=False) -> torch.tensor:
        imglist = []
        if rgb:
            imglist.append(self.rgb_image)
        if semantic:
            was_compressed = False
            if self.is_compressed():
                was_compressed = True
                self.uncompress()
            imglist.append(segdef.intid_tensor_to_rgb(self.semantic_image))
            if was_compressed:
                self.compress()
        if depth:
            imglist.append(self.get_depth_image().repeat((1, 3, 1, 1)) / 5.0)
        return torch.cat(imglist, dim=3 if horizontal else 2)

    def assign_3d_voxel_masks(self, voxel_mask_grid: VoxelGrid):
        """
        Update the 3d voxel mask for each 2d object detection.
        
        :param voxel_masks: tensor of size (N, 1, 61, 61, 10)
        """
        voxel_masks = voxel_mask_grid.data.bool()
        vx_size = voxel_mask_grid.voxel_size
        vx_origin = voxel_mask_grid.origin

        points = voxel_masks.nonzero()
        for idx, instance in enumerate(self.object_detections):
            centroid = points[points[:, 0] == idx].float().mean(dim=0)[-3:]
            centroid = centroid * vx_size + vx_origin + 0.5 * vx_size
            instance.assign_3d_voxel_mask(voxel_masks[idx, 0, :, :, :], centroid.cpu().numpy())
