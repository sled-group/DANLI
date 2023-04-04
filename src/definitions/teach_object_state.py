"""
Objects physical states and relations
"""
import os
import json
import copy
from typing import Union, Optional, Dict

from attrdict import AttrDict

PRINT_FUNCTION = None

ITHOR_PHYSICAL_STATE = [
    "isToggled",  # bool
    "isBroken",  # bool
    "isFilledWithLiquid",  # bool
    "isDirty",  # bool
    "isUsedUp",  # bool
    "isCooked",  # bool
    "isOpen",  # bool
    "isPickedUp",  # bool
]
# Note: we do not have `isSliced` since once an object is sliced,  it no longer exists
# in the scene. And for the spawned objects such as AppleSlicd, they are not slicable
# any more thus do not have the state `isSliced`.

ITHOR_OBJECT_RELATIONS = [
    "parentReceptacles",  # none or list of receptacles that hold the object
    "receptacleObjectIds",  # none or list of objects that the object hold
]
# Note: these properties are maintained by the simulator. The behaviour might be
# weird and undecidable, e.g., sometimes Floor is in an object's parentReceptacles
# although it is not directly hold the object: e.g. CreditCard -> GarbageCan, Floor

SIMBOT_CUSTOM_STATE = [
    "simbotIsBoiled",
    "simbotIsFilledWithWater",
    "simbotIsFilledWithCoffee",
]

SIMBOT_CUSTOM_OTHER = [
    "simbotPickedUp",
    "simbotIsCooked",  # used to mark cooked object that is not identified by ithor
    "simbotLastParentReceptacle",
    "simbotIsReceptacleOf",
    "simbotObjectClass",
]  # Other simbot custom properties used for goal check


class Unary:
    def __init__(
        self,
        name: str,
        value: Union[bool, str, None],
        confidence: Union[float, None, str] = "Default",
    ):
        self.name = name
        self.value = value
        self.confidence = confidence
    
    def __call__(self):
        return self.get_value()
    
    
    def get_value(self):
        return self.value

    def set_value(self, value, confidence=None, verbose=True):
        if PRINT_FUNCTION is not None and verbose:
            PRINT_FUNCTION(" - %s: %r -> %r" % (self.name, self.value, value))
        self.value = value
        self.confidence = confidence

    def update_from_estimation(self, new_state: "Unary", verbose=True):
        old_value, old_conf = self.value, self.confidence

        if self.confidence is None:
            # is known determinstically, do not update
            pass
        elif self.confidence == "Default":
            self.value = new_state.value
            self.confidence = new_state.confidence
        else:
            if isinstance(self.value, bool):
                score = self.confidence if self.value else 1 - self.confidence
                new_score = (
                    new_state.confidence if new_state.value else 1 - new_state.confidence
                )
                merged_score = (score + new_score) / 2

                if merged_score >= 0.5:
                    self.value = True
                    self.confidence = merged_score
                else:
                    self.value = False
                    self.confidence = 1 - merged_score
            else:
                if new_state.confidence > self.confidence:
                    self.value = new_state.value
                    self.confidence = new_state.confidence

        if PRINT_FUNCTION is not None and verbose:
            PRINT_FUNCTION(
                " - %s: %r(%r) -> %r(%r)"
                % (
                    self.name,
                    old_value,
                    "%.2f" % old_conf if isinstance(old_conf, float) else old_conf,
                    self.value,
                    "%.2f" % self.confidence
                    if isinstance(self.confidence, float)
                    else self.confidence,
                )
            )

    def __repr__(self) -> str:
        return "%s: %r (%r)" % (
            self.name,
            self.value,
            "%.2f" % self.confidence
            if isinstance(self.confidence, float)
            else self.confidence,
        )
    
    def __eq__(self, other):
        return self.name == other.name and self.value == other.value


class Relation:
    def __init__(
        self,
        name: str,
        value_with_conf: Dict[str, Union[float, None]],
    ):
        self.name = name
        self.value_with_conf = copy.deepcopy(value_with_conf)

    def __call__(self):
        return self.get_values()

    def get_values(self):
        return list(self.value_with_conf.keys())

    def get_values_with_confident(self, conf_threshold):
        values_conf = []
        for v, conf_score in self.value_with_conf.items():
            if conf_score >= conf_threshold:
                values_conf.append(v)
        return values_conf

    def update(self, new_value_with_conf):
        if PRINT_FUNCTION is not None:
            PRINT_FUNCTION(
                " - update: %s -> %s"
                % (str(self.value_with_conf), str(new_value_with_conf))
            )
        self.value_with_conf = copy.deepcopy(new_value_with_conf)

    def add(self, value, confidence=None, verbose=True):
        if PRINT_FUNCTION is not None and verbose:
            PRINT_FUNCTION(" - add: %s to %r" % (value, self.value_with_conf))
        self.value_with_conf[value] = confidence

    def remove(self, value, verbose=True):
        if value in self.value_with_conf:
            if PRINT_FUNCTION is not None and verbose:
                PRINT_FUNCTION(" - remove: " + str(value))
            del self.value_with_conf[value]

    def update_from_estimation(self, new_state: "Relation", verbose=True):
        old_value_with_conf = copy.deepcopy(self.value_with_conf)
        for value, conf in old_value_with_conf.items():
            if conf is None:
                # is known determinstically, do not update
                continue
            if value not in new_state.value_with_conf:
                # remove not detected objects
                self.remove(value, verbose=verbose)

        # add new objects
        for value, conf in new_state.value_with_conf.items():
            if value not in self.value_with_conf:
                self.value_with_conf[value] = conf
            elif self.value_with_conf[value] is None:
                # is known determinstically, do not update
                pass
            else:
                old_conf = self.value_with_conf[value]
                new_conf = (old_conf + conf) / 2
                self.value_with_conf[value] = new_conf

        if PRINT_FUNCTION is not None and verbose:
            PRINT_FUNCTION(
                " - %s: %r -> %r"
                % (
                    self.name,
                    old_value_with_conf,
                    self.value_with_conf,
                )
            )

    def __repr__(self) -> str:
        return "%s: %r" % (self.name, self.get_values())
    
    def __eq__(self, other):
        return self.name == other.name and set(self.value_with_conf.keys()) == set(other.value_with_conf.keys())


# we define the default object state
DEFAULT_OBJECT_STATE = {
    "visible": Unary("visible", True),  # when we add an instance it must be visible
    "interactable": Unary("interactable", False),  # visible and close enough
    "distance": Unary("distance", None),
    "isToggled": Unary("isToggled", False),
    "isBroken": Unary("isBroken", False),
    "isFilledWithLiquid": Unary("isFilledWithLiquid", False),
    "isDirty": Unary("isDirty", True),  # prepare for the worst
    "isUsedUp": Unary("isUsedUp", False),
    "isCooked": Unary("isCooked", False),
    "isSliced": Unary("isSliced", False),
    "isOpen": Unary("isOpen", False),
    "isPickedUp": Unary("isPickedUp", False),
    "receptacleObjectIds": Relation("receptacleObjectIds", dict()),
    "parentReceptacles": Relation("parentReceptacles", dict()),
    "simbotIsBoiled": Unary("simbotIsBoiled", False),
    "simbotIsFilledWithWater": Unary("simbotIsFilledWithWater", False),
    "simbotIsFilledWithCoffee": Unary("simbotIsFilledWithCoffee", False),
    "isPlacedOn": Unary(
        "isPlacedOn", None
    ),  # record the receptacle id where the object is placed
    "isInsideClosed": Unary(
        "isInsideClosed", False
    ),  # whether the object is inside a closed receptacle
    "simbotIsPickedUp": Unary("simbotIsPickedUp", False), # whether it is/was picked up
    "isObserved": Unary("isObserved", True), # whether it has been observed in the scene or is only a dummpy object that might exist 
    "sliceParent": Unary("sliceParent", None),
}

PRINT_STATE = [
    "visible",
    "isPickedUp",
    "isCooked",
    "isToggled",
    "simbotIsFilledWithWater",
    "simbotIsFilledWithCoffee",
    "isDirty",
    "isSliced",
    "simbotIsBoiled",
    "isPlacedOn",
    "isInsideClosed",
    "simbotIsPickedUp",
]


def create_default_object_state():
    return AttrDict(copy.deepcopy(DEFAULT_OBJECT_STATE))


def state_to_dict(state: Dict[str, Union[Unary, Relation]]) -> Dict:
    """
    Convert object state to dict
    """
    state_dict = dict()
    for k, v in state.items():
        if isinstance(v, Unary):
            state_dict[k] = v.get_value()
        elif isinstance(v, Relation):
            state_dict[k] = v.get_values()
        else:
            raise ValueError("Unknown object state type: {}".format(type(v)))
    return state_dict


# Mapping between stove burners and stove knobs
# Adopted from: https://github.com/rowanz/piglet/blob/main/data/knob_to_burner.json
# Warning: developed in ai2thor == 2.4.6 -> incomplete for ai2thor 3.1.0
ENV_DIR = os.path.dirname(os.path.realpath(__file__))
MAP_FILE_PATH = os.path.join(ENV_DIR, "meta_data", "ithor_knob_to_burner.json")
with open(MAP_FILE_PATH, "r") as f:
    KNOB_TO_BURNER = json.load(f)
    BURNER_TO_KNOB = {v: k for k, v in KNOB_TO_BURNER.items()}
