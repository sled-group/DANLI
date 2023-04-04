from typing import Dict
import math

from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.models.teach.spatial_state_repr import TeachSpatialStateRepr
from sledmap.mapper.utils.base_cls import Skill


DEBUG_DISABLE_PITCHING = False
MIN_PITCH = -1.30899694
MAX_PITCH = 1.30899694


class TiltToPitchSkill(Skill):
    def __init__(self):
        super().__init__()
        self.target_pitch = None
        self.last_diff = None

    def start_new_rollout(self):
        self._reset()

    def _reset(self):
        self.target_pitch = None
        self.last_diff = None

    def get_trace(self, device="cpu") -> Dict:
        return {}

    def clear_trace(self):
        ...

    def set_goal(self, target_pitch):
        self.target_pitch = max(min(target_pitch, MAX_PITCH), MIN_PITCH)
        self.last_diff = None


    def has_failed(self) -> bool:
        return False

    def act(self, state_repr : TeachSpatialStateRepr) -> TeachAction:
        # Debug what happens if we disable pitching
        if DEBUG_DISABLE_PITCHING:
            return TeachAction(action_type="Stop")

        pitch = state_repr.get_camera_pitch_deg()
        pitch = math.radians(pitch)

        # Allow control error to be between -pi and +pi
        ctrl_diff = pitch - self.target_pitch
        ctrl_diff = (ctrl_diff + math.pi) % (math.pi * 2) - math.pi

        step_size = math.pi * (30 / 180)

        # Rotate to the correct angle
        if ctrl_diff < -step_size/2:
            action_type = "Look Down"
        elif ctrl_diff > step_size/2:
            action_type = "Look Up"
        else:
            action_type = "Stop"

        # Sometimes tilting gets stuck due to collisions. This is to abort tilting and avoid getting stuck in an infinte loop.
        if action_type in ["LookDown", "LookUp"] and self.last_diff == ctrl_diff:
            print("TiltToPitch: Seems to be stuck. Stopping!")
            action_type = "Stop"
        self.last_diff = ctrl_diff

        return TeachAction(action_type=action_type)
