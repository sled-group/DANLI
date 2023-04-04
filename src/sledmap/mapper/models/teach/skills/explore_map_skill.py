from typing import Dict

import torch
import torch.nn.functional as F

from typing import Dict

from sledmap.mapper.utils.base_cls import Skill

import sledmap.mapper.env.teach.segmentation_definitions as segdef

from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.models.teach.spatial_state_repr import TeachSpatialStateRepr
from sledmap.mapper.models.teach.skills.go_to import GoToSkill

class ExploreMapSkill(Skill):
    def __init__(self, max_nav_count: int, simple_explore_baseline: bool):
        super().__init__()
        self.goto_skill = GoToSkill()

        self.max_nav_count = max_nav_count
        self.simple_explore_baseline = simple_explore_baseline
        self._reset()

    def start_new_rollout(self):
        self._reset()

    def _reset(self):
        self.pre_rotation_finished = False
        self.navigate_finished = False
        self.post_rotation_finished = False
        self.rotation_count = 0
        self.count = 0
        self.trace = {}

        self.rewardmap = None

    def _build_floor_edge_mask(self, state_repr):
        floor_ids = [segdef.object_string_to_intid(s) for s in ["Floor", "Carpet"]]
        wall_ids = [segdef.object_string_to_intid(s) for s in ["Wall", "Door", "Window"]]
        map_floor = state_repr.data.data[:, floor_ids].max(1).values.max(3).values.float()
        map_wall = state_repr.data.data[:, wall_ids].max(1).values.max(3).values.float()
        map_occupied = state_repr.data.occupancy.max(1).values.max(3).values.float()
        map_free = 1 - map_occupied

        map_observed = state_repr.obs_mask.data.max(1).values.max(3).values.float()
        map_unobserved = 1 - map_observed

        kern1 = torch.ones((1, 1, 3, 3), device=map_free.device)
        expanded_unobserved = (F.conv2d(map_unobserved[None, :, :, :], kern1, padding=1)[0] > 0.5).float()
        expanded_walls = (F.conv2d(map_wall[None, :, :, :], kern1, padding=1)[0] > 0.5).float()
        not_expanded_walls = 1 - expanded_walls

        unobserved_floor_boundary = expanded_unobserved * map_floor * not_expanded_walls
        return unobserved_floor_boundary

    def _construct_cost_function(self, state_repr : TeachSpatialStateRepr):
        b, c, w, l, h = state_repr.data.data.shape

        ground_2d = (state_repr.data.occupancy[:, :, :, :, 0:4].sum(4) == 1).float()

        filt_pos_prob = self._build_floor_edge_mask(state_repr)
        # Proposals on ground if ground_edge doesn't exist
        if filt_pos_prob.sum() < 0.5:
            filt_pos_prob = ground_2d + 1e-10

        filt_pos_prob = filt_pos_prob / filt_pos_prob.sum()

        # TODO: Figure out why sometimes there are 3 dimensions, sometimes 4
        if len(filt_pos_prob.shape) == 3:
            filt_pos_prob = filt_pos_prob[None, :, :, :]

        idx = torch.distributions.categorical.Categorical(probs=filt_pos_prob.view([-1])).sample().item()
        x = idx // l
        y = idx % l
        self.rewardmap = torch.zeros_like(filt_pos_prob)
        self.rewardmap[:, :, x, y] = 1.0

        return self.rewardmap

    def get_trace(self, device="cpu") -> Dict:
        return self.trace

    def clear_trace(self):
        self.trace = {}

    def has_failed(self) -> bool:
        return False

    def set_goal(self):
        self._reset()

    def act(self, state_repr: TeachSpatialStateRepr) -> TeachAction:
        if self.rewardmap is None:
            self.rewardmap = self._construct_cost_function(state_repr)
            self.goto_skill.set_goal(self.rewardmap)

        # Exclude extra 3 steps for pre-rotation
        if self.count > self.max_nav_count - 3:
            self.navigate_finished = True
            self.pre_rotation_finished = True

        # Baseline with 4 in-place rotations
        # if self.simple_explore_baseline:
        #     self.pre_rotation_finished = True
        #     self.navigate_finished = True
        print("====================== ExploreMapSkill")
        print("navigate_finished", self.navigate_finished)
        print("pre_rotation_finished", self.pre_rotation_finished)
        print("post_rotation_finished", self.post_rotation_finished)

        # Always look up and rotate around, even if we think that the object is found.
        # This costs very little, but can significantly improve the map representation between
        # failed navigation retries.
        if not self.pre_rotation_finished:

            
            
            action : TeachAction = TeachAction(action_type="Turn Left")
            # if self.rotation_count == 0:
            #     action : TeachAction = TeachAction(action_type="Look Down")
            # if self.rotation_count == 2:
            #     action : TeachAction = TeachAction(action_type="Look Up")
            
            # if self.rotation_count == 1:
            #     action : TeachAction = TeachAction(action_type="Look Down")
            # if self.rotation_count == 2:
            #     action : TeachAction = TeachAction(action_type="Look Down")
            if self.rotation_count == 2:
                # action : TeachAction = TeachAction(action_type="Look Up")
                self.rotation_count = -1
                self.pre_rotation_finished = True
            
            
            if state_repr.get_camera_pitch_deg() > 31:
                action : TeachAction = TeachAction(action_type="Look Up")
            elif state_repr.get_camera_pitch_deg() <= 1:
                action : TeachAction = TeachAction(action_type="Look Down")
            else:
                self.rotation_count += 1

        elif not self.navigate_finished:
            # Sample a non-stop action, unless the object is found or time limit exceeded
            count = 0
            while True:
                if self.rewardmap is None:
                    self.rewardmap = self._construct_cost_function(state_repr)
                    self.goto_skill.set_goal(self.rewardmap)
                action : TeachAction = self.goto_skill.act(state_repr)
                if action.is_stop():
                    self.rewardmap = None
                    count += 1
                    if count > 2:
                        self.navigate_finished = True
                        break
                else:
                    break

        # elif not self.post_rotation_finished:
        #     # First do 4x RotateLeft to look around
        #     action : TeachAction = TeachAction(action_type="Turn Left")
        #     self.rotation_count += 1

        #     # Finish rotating
        #     if self.rotation_count == 3:
        #         self.rotation_count = 0
        #         self.post_rotation_finished = True

        else:
            action = TeachAction(action_type="Stop")

        self.count += 1
        return action
