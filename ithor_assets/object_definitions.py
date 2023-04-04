import json, os
from pprint import pprint

DOMAIN_FILES_DIR = "./ithor_assets"
PRINT_OBJECT_DEFINITIONS = True

_INTERACTABLE_OBJECTS = []  # Objects whose physical state can change
_STRUCTURAL_OBJECTS = []  # Background objects that can not be interacted with
_STANDING_RECEPTACLES = []  # Receptacles that do not move in general
_PICKABLE_RECEPTACLES = []  # Containers that can be picked up
_OPENABLE_RECEPTACLES = []  # Receptacles that can be closed to store things
_GROUND = ["Carpet", "Floor"]  # Objects the agent can stand on
_GLASS = [
    "Mirror",
    "Window",
    "ShowerGlass",
]  # Glass objects that may lead to poor perception estimation

# Undocumented structural objects that present in ground truth semantic segmentations
# Such objects are not fully supported in the simulator (as opposed to documented ones).
# Their role should only be enrich the background of scenes.
# Warning: this set is manully summarized therefore incomplete
_ADDITIONAL_STRUCTURAL_OBJECTS = [
    "AirConditioner",
    "Bag",
    "Bookcase",
    "CabinetBody",
    "Carpet",
    "Ceiling",
    "CounterSide",
    "Cube",
    "Cylinder",
    "Dishwasher" "DomeLight",
    "Door",
    "LightFixture",
    "StoveBase",
    "Wall",
]

_OBJECT_AFFORDANCE = {}  # Affordance of each object
_ACTION_APPLICABILITY = {}  # Applicable object lists for each interaction action


# Add objects to the list/dict above
with open(os.path.join(DOMAIN_FILES_DIR, "affordance.json")) as f:
    _OBJECT_AFFORDANCE = json.load(f)
with open(os.path.join(DOMAIN_FILES_DIR, "action_applicability.json")) as f:
    _ACTION_APPLICABILITY = json.load(f)
for o, aff in _OBJECT_AFFORDANCE.items():
    if aff and aff != ["moveable"]:
        _INTERACTABLE_OBJECTS.append(o)
    else:
        _STRUCTURAL_OBJECTS.append(o)

    if "receptacle" in aff and "pickupable" not in aff:
        _STANDING_RECEPTACLES.append(o)
    if "receptacle" in aff and "pickupable" in aff:
        _PICKABLE_RECEPTACLES.append(o)
    if "receptacle" in aff and "openable" in aff:
        _OPENABLE_RECEPTACLES.append(o)

_STRUCTURAL_OBJECTS.extend(_ADDITIONAL_STRUCTURAL_OBJECTS)
_STRUCTURAL_OBJECTS.sort()


UNK_OBJ_STR = "OTHERS"
OBJECT_CLASSES = sorted(list(set(_INTERACTABLE_OBJECTS + _STRUCTURAL_OBJECTS)))
OBJECT_CLASSES += [UNK_OBJ_STR]
OBJECT_INT_TO_STR = {i: o for i, o in enumerate(OBJECT_CLASSES)}
OBJECT_STR_TO_INT = {o: i for i, o in enumerate(OBJECT_CLASSES)}

# Save OBJECT_STR_TO_INT into a json
with open(os.path.join(DOMAIN_FILES_DIR, "object_to_id.json"), "w") as f:
    json.dump(OBJECT_STR_TO_INT, f, indent=2)

if PRINT_OBJECT_DEFINITIONS:
    print("-" * 20, "_OBJECT_AFFORDANCE", "-" * 20)
    pprint(_OBJECT_AFFORDANCE)
    print("-" * 20, "_INTERACTABLE_OBJECTS", "-" * 20)
    pprint(_INTERACTABLE_OBJECTS)
    print("-" * 20, "_STRUCTURAL_OBJECTS", "-" * 20)
    pprint(_STRUCTURAL_OBJECTS)
    print("-" * 20, "_STANDING_RECEPTACLES", "-" * 20)
    pprint(_STANDING_RECEPTACLES)
    print("-" * 20, "_PICKABLE_RECEPTACLES", "-" * 20)
    pprint(_PICKABLE_RECEPTACLES)
    print("-" * 20, "_OPENABLE_RECEPTACLES", "-" * 20)
    pprint(_OPENABLE_RECEPTACLES)
    print("-" * 20, "OBJECT_CLASSES", "-" * 20)
    pprint(OBJECT_CLASSES)


# APIs
def get_num_objects():
    return len(OBJECT_CLASSES)


def object_str_to_intid(object_class_str) -> int:
    global OBJECT_STR_TO_INT, UNK_OBJ_STR
    unk_id = OBJECT_STR_TO_INT[UNK_OBJ_STR]
    return OBJECT_STR_TO_INT.get(object_class_str, unk_id)


def object_intid_to_string(intid: int) -> str:
    global OBJECT_INT_TO_STR, UNK_OBJ_STR
    return OBJECT_INT_TO_STR.get(intid, UNK_OBJ_STR)


def get_openable_receptacle_ids():
    return [object_str_to_intid(s) for s in _OPENABLE_RECEPTACLES]


def get_ground_ids():
    return [object_str_to_intid(s) for s in _GROUND]


def get_glass_ids():
    return [object_str_to_intid(s) for s in _GLASS]


def ithor_oid_to_object_class(objectId_str) -> str:
    """Map object `objectId` in iTHOR metadata to its class name
    examples: 'Book|1|2|3' -> 'Book'
              'Apple|0|1|2' -> 'Apple'
              'Apple|0|1|2|AppleSliced_0' -> 'AppleSliced'
              'Apple|0|1|2|AppleSliced_5' -> 'AppleSliced'

    :param object_str: objectId string
    :return: object class name string
    """
    splt = objectId_str.split("|")
    return splt[0] if len(splt) == 4 else splt[-1].split("_")[0]


def normalize_semantic_object_class(object_str) -> str:
    """Normalize object class names in iTHOR semantic segmentations
    iTHOR simulator has some random named background objects in its
    ground truth segmentation masks. We try our best to normalize
    these names into its object category defined in OBJECT_CLASSES.

    examples: 'FP228:Cube.1196' -> 'Cube'
              'StandardWall -> 'Wall'
              'Door 1' -> 'Door'

    :param object_str: object class names from ground truth instance information
                       such as ai2thor_controller_event.class_masks.keys() or
                       ai2thor_controller_event.instance_detections2D.keys()
    :return: normalized object class string
    """

    # if input is an instance objectId, first map it to its class name
    if "|" in object_str:
        object_str = ithor_oid_to_object_class(object_str)

    # Remove numbers: e.g. 'FP228:Cube.1196' -> 'Cube'
    object_str = object_str.split(":")[-1].split(".")[0]

    # Case-by-case normalizations
    # Warning: these rules are manully summarized therefore incomplete
    if "StandardDoor" in object_str or (
        object_str[:4] == "Door" and len(object_str) in [5, 6]
    ):
        object_str = "Door"
    elif any(
        [i in object_str for i in ["StandardWall", "polySurface"]]
    ) or object_str in [
        "Walls",
        "BackSplash",
        "Decals_2",
        "Room",
    ]:
        object_str = "Wall"
        # Note: polySurface may be doors or another surfaces, but we simply classify
        # them as walls as they are just obstacles.
    elif "LightFixture" in object_str:
        object_str = "LightFixture"
    elif any(
        [i in object_str for i in ["StoveBase", "StoveBottomDoor", "StoveTopDoor"]]
    ) or object_str in ["OVENDOOR", "Stove", "StoveTopGas"]:
        object_str = "StoveBase"
    elif object_str == "Rug":
        object_str = "Carpet"
    elif object_str == "Books":
        object_str = "Book"
    elif object_str in [
        "StandardIslandHeight",
        "IslandMesh",
        "StandardCounterHeightWidth",
        "KitchenIsland",
    ]:
        object_str = "CounterSide"
    elif object_str in ["UpperCabinets", "CabinetsShell"]:
        object_str = "Cabinet"
    elif "Ceiling" in object_str:
        object_str = "Ceiling"

    return object_str
