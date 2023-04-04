"""
Object class and affordance definitions in TEACh and basic APIs for SEAGULL.
"""
import copy
from re import I
from typing import Dict
from enum import Enum, unique
from .teach_receptacle_compatibility import RECEPTACLE_COMPATIBILITY


@unique
class ObjectClass(Enum):
    """
    All the object class name symbols and their integer indexes
    """

    # index 0 is kept for the unknown class in segmantic segmantation
    AlarmClock = 1
    AluminumFoil = 2
    Apple = 3
    AppleSliced = 4
    ArmChair = 5
    BaseballBat = 6
    BasketBall = 7
    Bathtub = 8
    BathtubBasin = 9
    Bed = 10
    Blinds = 11
    Book = 12
    Boots = 13
    Bottle = 14
    Bowl = 15
    Box = 16
    Bread = 17
    BreadSliced = 18
    ButterKnife = 19
    CD = 20
    Cabinet = 21
    Candle = 22
    CellPhone = 23
    Chair = 24
    Cloth = 25
    CoffeeMachine = 26
    CoffeeTable = 27
    CounterTop = 28
    CreditCard = 29
    Cup = 30
    Desk = 31
    DeskLamp = 32
    Desktop = 33
    DiningTable = 34
    DishSponge = 35
    DogBed = 36
    Drawer = 37
    Dresser = 38
    Dumbbell = 39
    Egg = 40
    EggCracked = 41
    Faucet = 42
    FloorLamp = 43
    Footstool = 44
    Fork = 45
    Fridge = 46
    GarbageBag = 47
    GarbageCan = 48
    HandTowel = 49
    HandTowelHolder = 50
    HousePlant = 51
    Kettle = 52
    KeyChain = 53
    Knife = 54
    Ladle = 55
    Laptop = 56
    LaundryHamper = 57
    Lettuce = 58
    LettuceSliced = 59
    LightSwitch = 60
    Microwave = 61
    Mirror = 62
    Mug = 63
    Newspaper = 64
    Ottoman = 65
    Pan = 66
    PaperTowelRoll = 67
    Pen = 68
    Pencil = 69
    PepperShaker = 70
    Pillow = 71
    Plate = 72
    Plunger = 73
    Pot = 74
    Potato = 75
    PotatoSliced = 76
    RemoteControl = 77
    RoomDecor = 78
    Safe = 79
    SaltShaker = 80
    ScrubBrush = 81
    Shelf = 82
    ShelvingUnit = 83
    ShowerCurtain = 84
    ShowerDoor = 85
    ShowerGlass = 86
    ShowerHead = 87
    SideTable = 88
    Sink = 89
    SinkBasin = 90
    SoapBar = 91
    SoapBottle = 92
    Sofa = 93
    Spatula = 94
    Spoon = 95
    SprayBottle = 96
    Statue = 97
    Stool = 98
    StoveBurner = 99
    StoveKnob = 100
    TVStand = 101
    TableTopDecor = 102
    TeddyBear = 103
    Television = 104
    TennisRacket = 105
    TissueBox = 106
    Toaster = 107
    Toilet = 108
    ToiletPaper = 109
    ToiletPaperHanger = 110
    Tomato = 111
    TomatoSliced = 112
    Towel = 113
    TowelHolder = 114
    VacuumCleaner = 115
    Vase = 116
    Watch = 117
    WateringCan = 118
    Window = 119
    WineBottle = 120  # above: interactable objects
    AirConditioner = 121  # below: structural objects
    Bag = 122
    Bookcase = 123
    CabinetBody = 124
    Carpet = 125
    Ceiling = 126
    CounterSide = 127
    Cube = 128
    Curtains = 129
    Cylinder = 130
    Dishwasher = 131
    DomeLight = 132
    Door = 133
    Floor = 134
    LightFixture = 135
    Painting = 136
    Poster = 137
    StoveBase = 138
    TargetCircle = 139
    Wall = 140
    OTHERS = 141  # all other structural objects

    @classmethod
    def has_object(cls, obj_cls_name: str) -> bool:
        return obj_cls_name in cls.__members__

    @classmethod
    def get_num_objects(cls):
        return len(cls) + 1

    @classmethod
    def name_to_id(cls, obj_cls_name: str) -> int:
        if cls.has_object(obj_cls_name):
            return cls[obj_cls_name].value
        return cls.OTHERS.value

    @classmethod
    def id_to_name(cls, obj_int_id: int) -> str:
        if obj_int_id >= cls.get_num_objects():
            return cls.OTHERS.name
        return cls(obj_int_id).name

    @classmethod
    def get_all_names(cls):
        return [n.name for n in cls]


# Object affordance defined in the official document
_OBJECT_AFFORDANCE_OFFICIAL: Dict[str, set] = {
    "AlarmClock": {"pickupable"},
    "AluminumFoil": {"pickupable"},
    "Apple": {"pickupable", "sliceable"},
    "AppleSliced": {"pickupable"},
    "ArmChair": {"receptacle", "moveable"},
    "BaseballBat": {"pickupable"},
    "BasketBall": {"pickupable"},
    "Bathtub": {"receptacle"},
    "BathtubBasin": {"receptacle"},
    "Bed": {"receptacle", "dirtyable"},
    "Blinds": {"openable"},
    "Book": {"openable", "pickupable"},
    "Boots": {"pickupable"},
    "Bottle": {"pickupable", "canFillWithLiquid", "breakable"},
    "Bowl": {"pickupable", "receptacle", "canFillWithLiquid", "breakable", "dirtyable"},
    "Box": {"openable", "pickupable", "receptacle"},
    "Bread": {"pickupable", "sliceable"},
    "BreadSliced": {"pickupable", "cookable"},
    "ButterKnife": {"pickupable"},
    "CD": {"pickupable"},
    "Cabinet": {"openable", "receptacle"},
    "Candle": {"pickupable", "toggleable"},
    "CellPhone": {"pickupable", "toggleable", "breakable"},
    "Chair": {"moveable", "receptacle"},
    "Cloth": {"pickupable", "dirtyable"},
    "CoffeeMachine": {"toggleable", "receptacle", "moveable"},
    "CoffeeTable": {"receptacle", "moveable"},
    "CounterTop": {"receptacle"},
    "CreditCard": {"pickupable"},
    "Cup": {"pickupable", "receptacle", "canFillWithLiquid", "breakable", "dirtyable"},
    "Curtains": set(),
    "Desk": {"receptacle", "moveable"},
    "DeskLamp": {"toggleable", "moveable"},
    "Desktop": {"moveable"},
    "DiningTable": {"receptacle", "moveable"},
    "DishSponge": {"pickupable"},
    "DogBed": {"moveable"},
    "Drawer": {"openable", "receptacle"},
    "Dresser": {"receptacle", "moveable"},
    "Dumbbell": {"pickupable"},
    "Egg": {"pickupable", "sliceable", "breakable"},
    "EggCracked": {"pickupable", "cookable"},
    "Faucet": {"toggleable"},
    "Floor": set(),
    "FloorLamp": {"toggleable", "moveable"},
    "Footstool": {"moveable"},
    "Fork": {"pickupable"},
    "Fridge": {"openable", "receptacle"},
    "GarbageBag": {"moveable"},
    "GarbageCan": {"receptacle", "moveable"},
    "HandTowel": {"pickupable"},
    "HandTowelHolder": {"receptacle"},
    "HousePlant": {"canFillWithLiquid", "moveable"},
    "Kettle": {"openable", "pickupable", "canFillWithLiquid"},
    "KeyChain": {"pickupable"},
    "Knife": {"pickupable"},
    "Ladle": {"pickupable"},
    "Laptop": {"openable", "pickupable", "toggleable", "breakable"},
    "LaundryHamper": {"receptacle", "moveable"},
    "Lettuce": {"pickupable", "sliceable"},
    "LettuceSliced": {"pickupable"},
    "LightSwitch": {"toggleable"},
    "Microwave": {"openable", "toggleable", "receptacle", "moveable"},
    "Mirror": {"breakable", "dirtyable"},
    "Mug": {"pickupable", "receptacle", "canFillWithLiquid", "breakable", "dirtyable"},
    "Newspaper": {"pickupable"},
    "Ottoman": {"receptacle", "moveable"},
    "Painting": set(),
    "Pan": {"pickupable", "receptacle", "dirtyable"},
    "PaperTowelRoll": {"pickupable", "canBeUsedUp"},
    "Pen": {"pickupable"},
    "Pencil": {"pickupable"},
    "PepperShaker": {"pickupable"},
    "Pillow": {"pickupable"},
    "Plate": {"pickupable", "receptacle", "breakable", "dirtyable"},
    "Plunger": {"pickupable"},
    "Poster": set(),
    "Pot": {"pickupable", "receptacle", "canFillWithLiquid", "dirtyable"},
    "Potato": {"pickupable", "sliceable", "cookable"},
    "PotatoSliced": {"pickupable", "cookable"},
    "RemoteControl": {"pickupable"},
    "RoomDecor": {"moveable"},
    "Safe": {"openable", "receptacle", "moveable"},
    "SaltShaker": {"pickupable"},
    "ScrubBrush": {"pickupable"},
    "Shelf": {"receptacle"},
    "ShelvingUnit": {"moveable"},
    "ShowerCurtain": {"openable"},
    "ShowerDoor": {"openable", "breakable"},
    "ShowerGlass": {"breakable"},
    "ShowerHead": {"toggleable"},
    "SideTable": {"receptacle", "moveable"},
    "Sink": {"receptacle"},
    "SinkBasin": {"receptacle"},
    "SoapBar": {"pickupable"},
    "SoapBottle": {"pickupable", "canBeUsedUp"},
    "Sofa": {"receptacle", "moveable"},
    "Spatula": {"pickupable"},
    "Spoon": {"pickupable"},
    "SprayBottle": {"pickupable"},
    "Statue": {"pickupable", "breakable"},
    "Stool": {"moveable", "receptacle"},
    "StoveBurner": {"toggleable", "receptacle"},
    "StoveKnob": {"toggleable"},
    "TVStand": {"receptacle", "moveable"},
    "TableTopDecor": {"pickupable"},
    "TargetCircle": set(),
    "TeddyBear": {"pickupable"},
    "Television": {"toggleable", "breakable", "moveable"},
    "TennisRacket": {"pickupable"},
    "TissueBox": {"pickupable", "canBeUsedUp"},
    "Toaster": {"toggleable", "receptacle", "moveable"},
    "Toilet": {"openable", "receptacle"},
    "ToiletPaper": {"pickupable", "canBeUsedUp"},
    "ToiletPaperHanger": {"receptacle"},
    "Tomato": {"pickupable", "sliceable"},
    "TomatoSliced": {"pickupable"},
    "Towel": {"pickupable"},
    "TowelHolder": {"receptacle"},
    "VacuumCleaner": {"moveable"},
    "Vase": {"pickupable", "breakable"},
    "Watch": {"pickupable"},
    "WateringCan": {"pickupable", "canFillWithLiquid"},
    "Window": {"breakable"},
    "WineBottle": {"pickupable", "canFillWithLiquid", "breakable"},
}

# Custom affordance we add for contextualized interactions
_OBJECT_AFFORDANCE_CUSTOM: Dict[str, set] = {
    "Mug": {"canFillWithCoffee"},
    "BreadSliced": {"stovecookable", "microwavable", "boilable", "toastable"},
    "EggCracked": {"stovecookable", "microwavable", "boilable"},
    "Potato": {"stovecookable", "microwavable", "boilable"},
    "PotatoSliced": {"stovecookable", "microwavable", "boilable"},
}

# Defintion for all the object affordance
OBJECT_AFFORDANCE = copy.deepcopy(_OBJECT_AFFORDANCE_OFFICIAL)
for o, aff in _OBJECT_AFFORDANCE_CUSTOM.items():
    OBJECT_AFFORDANCE[o].update(aff)  # set union


# Categorize objects according to their affordance
_INTERACTABLE_OBJECTS = set()  # Objects whose physical state can change
_STRUCTURAL_OBJECTS = set()  # Background objects that can not be interacted with

# Undocumented structural objects that present in ground truth semantic segmentations
# Such objects are not fully supported in the simulator (as opposed to documented ones).
# Their role should only be enrich the background of scenes.
# Warning: this set is manully summarized therefore incomplete
_ADDITIONAL_STRUCTURAL_OBJECTS = {
    "AirConditioner",
    "Bag",
    "Bookcase",
    "CabinetBody",
    "Carpet",
    "Ceiling",
    "CounterSide",
    "Cube",
    "Cylinder",
    "Dishwasher",
    "DomeLight",
    "Door",
    "LightFixture",
    "StoveBase",
    "Wall",
}

for o, aff in OBJECT_AFFORDANCE.items():
    if aff and aff != ["moveable"]:
        _INTERACTABLE_OBJECTS.add(o)
    else:
        _STRUCTURAL_OBJECTS.add(o)
_STRUCTURAL_OBJECTS.update(_ADDITIONAL_STRUCTURAL_OBJECTS)

# Object name list for panoptic segmantation
THING_NAMES = sorted(list(_INTERACTABLE_OBJECTS))
STUFF_NAMES = sorted(list(_STRUCTURAL_OBJECTS)) + ["OTHERS"]


recogniziable_predicate = {
    "toggleable": "isToggled",
    "canFillWithLiquid": "isFilledWithWater",
    "canFillWithCoffee": "simbotIsFilledWithCoffee",
    "dirtyable": "isDirty",
    "cookable": "isCooked",
    "openable": "isOpen",
}

# create a mapping form affordance to objects
AFFORDANCE_TO_OBJECTS = {}
OBJECT_STATE_OF_INTEREST = {}
for obj, aff in OBJECT_AFFORDANCE.items():
    for a in aff:
        if a not in AFFORDANCE_TO_OBJECTS:
            AFFORDANCE_TO_OBJECTS[a] = set()
        AFFORDANCE_TO_OBJECTS[a].add(obj)
        if obj != "StoveKnob" and a in recogniziable_predicate:
            pred = recogniziable_predicate[a]
            if obj not in OBJECT_STATE_OF_INTEREST:
                OBJECT_STATE_OF_INTEREST[obj] = []
            OBJECT_STATE_OF_INTEREST[obj].append(pred)


def get_object_state_of_interest(instance_meata_data: dict) -> list:
    """
    Get the state of interest for the object instance.

    :param instance_meata_data: dict of iTHOR instance meta data
    :return: list of tuples of object states of interest and their values
            E.g. Alarmclock -> []
                 Pot -> [('isDirty', False), ('isFilledWithWater', True)]
                 Potato -> [('isCooked', False)]
    """
    obj_class_name = instance_meata_data["objectType"]
    if obj_class_name not in OBJECT_STATE_OF_INTEREST:
        return []

    object_state = []
    state_of_interest = OBJECT_STATE_OF_INTEREST[obj_class_name]
    for state in state_of_interest:
        if state == "isFilledWithWater":
            assert instance_meata_data["canFillWithLiquid"]
            isfilledcoffee = instance_meata_data.get("simbotIsFilledWithCoffee", False)
            if instance_meata_data["isFilledWithLiquid"] and not isfilledcoffee:
                object_state.append((state, True))
            else:
                object_state.append((state, False))
        elif state == "simbotIsFilledWithCoffee":
            isfilledcoffee = instance_meata_data.get("simbotIsFilledWithCoffee", False)
            object_state.append(state, isfilledcoffee)
        elif state == "isCooked":
            iscooked = instance_meata_data.get("isCooked", False)
            simbotiscooked = instance_meata_data.get("simbotIsCooked", False)
            simbotisboiled = instance_meata_data.get("simbotIsBoiled", False)
            if iscooked or simbotiscooked or simbotisboiled:
                object_state.append((state, iscooked))
        else:
            assert (
                state in instance_meata_data
            ), "state {} not in instance_meata_data???".format(state)
            object_state.append((state, instance_meata_data[state]))

    assert object_state, "object_state is empty???"
    return object_state


# Affordance APIs
def is_interactable(obj_cls_name: str) -> bool:
    return obj_cls_name in _INTERACTABLE_OBJECTS


def is_structural(obj_cls_name: str) -> bool:
    return obj_cls_name in _STRUCTURAL_OBJECTS


def get_object_affordance(obj_cls_name: str) -> set:
    """
    Get the object's affordance

    :param obj_cls_name: object class name string
    :return: set of affordance
    """
    if obj_cls_name in OBJECT_AFFORDANCE:
        return OBJECT_AFFORDANCE[obj_cls_name]
    return set()


def get_object_receptacle_compatibility(obj_cls_name: str) -> set:
    """
    Get the object's receptacle compatibility

    :param obj_cls_name: object class name string
    :return: set of compatible receptacles
    """
    if obj_cls_name not in RECEPTACLE_COMPATIBILITY:
        return set()
    return RECEPTACLE_COMPATIBILITY[obj_cls_name]


# Object string handling APIs
def ithor_oid_to_object_class(objectId_str: str) -> str:
    """
    Map object `objectId` in iTHOR metadata to its class name
    examples: 'Book|1|2|3' -> 'Book'
              'Apple|0|1|2' -> 'Apple'
              'Apple|0|1|2|AppleSliced_0' -> 'AppleSliced'
              'Apple|0|1|2|AppleSliced_5' -> 'AppleSliced'

    :param object_str: objectId string
    :return: object class name string
    """
    splt = objectId_str.split("|")
    return splt[0] if len(splt) == 4 else splt[-1].split("_")[0]


def normalize_semantic_object_class(object_str: str) -> str:
    """
    Normalize object class names in iTHOR semantic segmentations
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


SIMILAR_OBJECT_SET = {
    "Tables": {"CounterTop", "DiningTable", "CoffeeTable", "Desk", "SideTable"},
    "Cups": {"Cup", "Mug"},
}


def objects_are_similar(obj_type1, obj_type2):
    for v in SIMILAR_OBJECT_SET.values():
        if obj_type1 in v and obj_type2 in v:
            return True
    return obj_type1 in obj_type2 or obj_type2 in obj_type1
