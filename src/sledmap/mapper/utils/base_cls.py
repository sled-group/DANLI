from typing import Union, Iterable, Tuple, Dict
from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class Observation(ABC):
    def __init__(self):
        ...

    @classmethod
    @abstractmethod
    def collate(cls, observations: Iterable["Observation"]) -> "Observation":
        """
        Creates a single "observation" that represents a batch of observations
        """
        ...

    @abstractmethod
    def represent_as_image(self) -> torch.tensor:
        ...

    @abstractmethod
    def to(self, device) -> "Observation":
        """Moves self to the given Torch device, and returns self"""
        ...


class StateRepr(ABC):

    def __init__(self):
        ...

    @classmethod
    @abstractmethod
    def collate(cls, states: Iterable["StateRepr"]) -> "StateRepr":
        """
        Creates a single StateRepresentation that represents a batch of states
        """
        ...

    @abstractmethod
    def represent_as_image(self) -> torch.tensor:
        ...

    def cast(self, cls, device="cpu"):
        raise NotImplementedError(f"Casting of {type(self)} not implemented")


class Subgoal(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def type_id(self):
        ...

    @abstractmethod
    def is_stop(self) -> bool:
        """
        :return: True if this is a STOP-action, False otherwise.
        """
        ...

class Task(ABC):
    def __init__(self):
        ...
        
class Action(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def is_stop(self) -> bool:
        """
        :return: True if this is a STOP-action, False otherwise.
        """
        ...

class Env(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def reset(self) -> (Observation, Task):
        ...

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        ...

class ObservationFunction(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, observation: Observation, prev_state: Union[StateRepr, None], goal: Union[Subgoal, None]) -> StateRepr:
        ...

class StateRepr(ABC):

    def __init__(self):
        ...

    @classmethod
    @abstractmethod
    def collate(cls, states: Iterable["StateRepr"]) -> "StateRepr":
        """
        Creates a single StateRepresentation that represents a batch of states
        """
        ...

    @abstractmethod
    def represent_as_image(self) -> torch.tensor:
        ...

    def cast(self, cls, device="cpu"):
        raise NotImplementedError(f"Casting of {type(self)} not implemented")

class LearnableModel(nn.Module, ABC):
    """
    Represents a model that can be trained on batches of transitions.
    """

    @abstractmethod
    def loss(self, batch: Dict) -> (torch.tensor, Dict):
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Name to identify this model as opposed to other models used within the same agent"""
        ...

class Agent(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def get_trace(self, device="cpu") -> Dict:
        return {}

    @abstractmethod
    def clear_trace(self):
        ...

    def action_execution_failed(self):
        # Tell the agent that the most recently predicted action has failed
        ...

    @abstractmethod
    def start_new_rollout(self, task: Task, state_repr: StateRepr = None):
        # Optionally take state_repr argument to allow switching instructions while keeping the map
        ...

    def finalize(self, total_reward: float):
        """A chance for the agent to wrap up after a task is done (e.g. by saving trace data or what not)"""
        ...

    @abstractmethod
    def act(self, observation_or_state_repr: Union[Observation, StateRepr]) -> Action:
        ...


class Skill(ABC):
    """
    Skills differ from Agents in that
    """
    def __init__(self):
        super().__init__()

    def start_new_rollout(self):
        ...

    @abstractmethod
    def get_trace(self, device="cpu") -> Dict:
        # Return a dictionary of outputs (e.g. tensors, arrays) that illustrate internal reasoning of the skill
        ...

    def clear_trace(self):
        # Clear any traces collected in this rollout to have a clean slate for next rollout or sample
        ...

    @abstractmethod
    def set_goal(self, goal):
        ...

    @abstractmethod
    def act(self, state_repr: StateRepr) -> Action:
        ...

    @abstractmethod
    def has_failed(self) -> bool:
        ...
