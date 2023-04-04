"""
Object semantic class definitions
"""
import logging
from enum import Enum, unique
from typing import Dict, Set
from definitions.teach_objects import OBJECT_AFFORDANCE, ObjectClass


TEACH_CUSTOM_SEMANTIC_CLASSES = {
    "Appliances": {
        "CoffeeMachine",
        "Fridge",
        "Microwave",
        "StoveBurner",
        "StoveKnob",
        "Toaster",
        "VacuumCleaner",
    },
    "Chairs": {"ArmChair", "Chair", "Sofa", "Stool"},
    "CleaningProducts": {
        "Cloth",
        "DishSponge",
        "GarbageBag",
        "PaperTowelRoll",
        "Plunger",
        "ScrubBrush",
        "SoapBar",
        "SoapBottle",
        "SprayBottle",
        "TissueBox",
        "ToiletPaper",
        "VacuumCleaner",
    },
    "Computers": {"Desktop", "Laptop"},
    "Condiments": {"PepperShaker", "SaltShaker"},
    "Cookware": {"Kettle", "Knife", "Ladle", "Pan", "Pot", "Spatula"},
    "Dishware": {"Bowl", "Plate"},
    "Drinkware": {"Bottle", "Cup", "Mug"},
    "Electronics": {
        "AlarmClock",
        "CellPhone",
        "Desktop",
        "Laptop",
        "RemoteControl",
        "Television",
    },
    "Food": {
        "Apple",
        "AppleSliced",
        "Bread",
        "BreadSliced",
        "Egg",
        "EggCracked",
        "Lettuce",
        "LettuceSliced",
        "Potato",
        "PotatoSliced",
        "Tomato",
        "TomatoSliced",
    },
    "Fruit": {"Apple", "AppleSliced", "Tomato", "TomatoSliced"},
    "Furniture": {
        "ArmChair",
        "Bed",
        "Chair",
        "CoffeeTable",
        "Desk",
        "DiningTable",
        "Dresser",
        "Footstool",
        "Ottoman",
        "Shelf",
        "SideTable",
        "Sofa",
        "Stool",
        "TVStand",
    },
    "HygieneProducts": {"HandTowel", "SoapBar", "SoapBottle", "TissueBox", "Towel"},
    "KitchenUtensils": {"ButterKnife", "Fork", "Knife", "Ladle", "Spatula", "Spoon"},
    "Knives": {"ButterKnife", "Knife"},
    "Lamps": {"DeskLamp", "FloorLamp"},
    "Lights": {"DeskLamp", "FloorLamp", "LightSwitch"},
    "MediaEntertainment": {
        "Book",
        "CD",
        "CellPhone",
        "Desktop",
        "Laptop",
        "Newspaper",
        "RemoteControl",
        "Television",
    },
    "RoomDecorations": {
        "AlarmClock",
        "Candle",
        "HousePlant",
        "RoomDecor",
        "Statue",
        "TableTopDecor",
        "TeddyBear",
        "Vase",
        "WineBottle",
    },
    "Silverware": {"ButterKnife", "Fork", "Spoon"},
    "SmallHandheldObjects": {
        "Book",
        "CD",
        "CellPhone",
        "CreditCard",
        "KeyChain",
        "Pen",
        "Pencil",
        "RemoteControl",
        "Watch",
    },
    "Soap": {"SoapBar", "SoapBottle"},
    "SportsEquipment": {"BaseballBat", "BasketBall", "Dumbbell", "TennisRacket"},
    "Tables": {"CoffeeTable", "Desk", "DiningTable", "Shelf", "SideTable"},
    "Tableware": {
        "Bottle",
        "Bowl",
        "Cup",
        "Mug",
        "PepperShaker",
        "Plate",
        "SaltShaker",
        "WineBottle",
    },
    "Vegetables": {"Lettuce", "LettuceSliced", "Potato", "PotatoSliced"},
    "WallDecor": {
        "Blinds",
        "Curtains",
        "Painting",
        "Poster",
        "ShowerCurtain",
        "Window",
    },
    "WaterSources": {"Faucet", "ShowerHead", "Toilet"},
    "WritingUtensils": {"Pen", "Pencil"},
}


# Obtain some semantic classes from object affordance
_STANDING_RECEPTACLES = set()  # Receptacles that do not move in general
_PICKABLE_RECEPTACLES = set()  # Containers that can be picked up
_OPENABLE_RECEPTACLES = set()  # Receptacles that can be closed to store things
_WATER_CONTAINERS = set()  # Containers that can hold water for watering plant
for o, aff in OBJECT_AFFORDANCE.items():
    if "receptacle" in aff and "pickupable" not in aff:
        _STANDING_RECEPTACLES.add(o)
    if "receptacle" in aff and "pickupable" in aff:
        _PICKABLE_RECEPTACLES.add(o)
    if "receptacle" in aff and "openable" in aff:
        _OPENABLE_RECEPTACLES.add(o)
    if "pickupable" in aff and "canFillWithLiquid" in aff:
        _WATER_CONTAINERS.add(o)

# Additional semantic classes
FUNCTIONAL_SEMANTIC_CLASSES = {
    "ManipulableKitchenUtils": {
        "CoffeeMachine",
        "Microwave",
        "StoveKnob",
        "Toaster",
    },  # Toggleable objects for food/drink preparation
    "FoodCookers": {
        "Bowl",
        "Pan",
        "Pot",
        "Microwave",
        "StoveBurner",
    },  # Receptacles used for cooking food
    "StoveTopCookers": {
        "Kettle",
        "Pan",
        "Pot",
    },  # Cookwares used on top of stove burners
    "BoilContainers": {
        "Bowl",
        "Pot",
    },  # Containers used for boiling
    "WaterTaps": {
        "Faucet",
        "ShowerHead",
    },  # Toggleable objects to make running water
    "WaterBottomReceptacles": {
        "Bathtub",
        "BathtubBasin",
        "Sink",
        "SinkBasin",
    },  # Receptacles for placing objects below water
    "WaterContainers": _WATER_CONTAINERS,  # water plant / move water
    # Include: Bottle, Bowl, Cup, Kettle, Mug, Pot, WateringCan, WineBottle
    "OpenableReceptacles": _OPENABLE_RECEPTACLES,  # store objects
    # Include: Box, Cabinet, Drawer, Fridge, Microwave, Safe, Toilet
    "StandingReceptacles": _STANDING_RECEPTACLES,  # 'big' landmarks
    "PickableReceptacles": _PICKABLE_RECEPTACLES,  # enable stack
    # Include: Bowl, Box, Cup, Mug, Pan, Plate, Pot
    "OccupiableReceptacles": {
        "Bathtub",
        "BathtubBasin",
        "Microwave",
        "Pan",
        "Pot",
        "Sink",
        "SinkBasin",
        "StoveBurner",
    },  # May need to clear before placing. Warning: incomplete
    "Ground": {
        "Carpet",
        "Floor",
    },  # Places where the agent can stand
    "Glass": {
        "Mirror",
        "Window",
        "ShowerGlass",
    },  # May lead to noisy perception estimation (e.g. due to reflections)
}

# Gather all the semantic class definitions
SEMANTIC_CLS_TO_OBJECTS: Dict[str, Set[str]] = {}
SEMANTIC_CLS_TO_OBJECTS.update(TEACH_CUSTOM_SEMANTIC_CLASSES)
SEMANTIC_CLS_TO_OBJECTS.update(FUNCTIONAL_SEMANTIC_CLASSES)

# Construct a mapping from object string to its semantic classes
OBJECT_TO_SEMANTIC_CLASSES: Dict[str, Set[str]] = {o.name: set() for o in ObjectClass}
for semcls, objs in SEMANTIC_CLS_TO_OBJECTS.items():
    for obj in objs:
        OBJECT_TO_SEMANTIC_CLASSES[obj].add(semcls)


@unique
class SemanticClass(Enum):
    """
    All the semantic class names and their integer indexes
    """

    Appliances = 0
    BoilContainers = 1
    Chairs = 2
    CleaningProducts = 3
    Computers = 4
    Condiments = 5
    Cookware = 6
    Dishware = 7
    Drinkware = 8
    Electronics = 9
    Food = 10
    FoodCookers = 11
    Fruit = 12
    Furniture = 13
    Glass = 14
    Ground = 15
    HygieneProducts = 16
    KitchenUtensils = 17
    Knives = 18
    Lamps = 19
    Lights = 20
    ManipulableKitchenUtils = 21
    MediaEntertainment = 22
    OccupiableReceptacles = 23
    OpenableReceptacles = 24
    PickableReceptacles = 25
    RoomDecor = 26
    Silverware = 27
    SmallHandheldObjects = 28
    Soap = 29
    SportsEquipment = 30
    StandingReceptacles = 31
    StoveTopCookers = 32
    Tables = 33
    Tableware = 34
    Vegetables = 35
    WallDecor = 36
    WaterBottomReceptacles = 37
    WaterContainers = 38
    WaterSources = 39
    WaterTaps = 40
    WritingUtensils = 41

    @classmethod
    def has_semcls(cls, sem_cls_name: str) -> bool:
        return sem_cls_name in cls.__members__

    @classmethod
    def get_num_sem_classes(cls):
        return len(cls)

    @classmethod
    def name_to_id(cls, sem_cls_name: str) -> int:
        if cls.has_semcls(sem_cls_name):
            return cls[sem_cls_name].value
        raise ValueError("Invalid semantic class name: %s" % sem_cls_name)

    @classmethod
    def id_to_name(cls, sem_int_id: int) -> str:
        if sem_int_id < cls.get_num_sem_classes():
            return cls(sem_int_id).name
        raise ValueError("Invalid semantic class index: %d" % sem_int_id)

    @classmethod
    def get_all_objs_in_semcls(cls, semcls_name: str) -> set:
        """
        Get all the objects for a semantic class

        :param sem_cls_name: object class name string
        :return: a set of all the objects in the semantic class
        """

        if semcls_name not in SEMANTIC_CLS_TO_OBJECTS:
            logging.warning("Unknown semantic class name: %s" % semcls_name)
        return SEMANTIC_CLS_TO_OBJECTS.get(semcls_name, set())

    @classmethod
    def get_all_semcls_for_obj(cls, obj_cls_name: str) -> set:
        """
        Get the associated semantic classes for the object

        :param obj_cls_name: object class name string
        :return: a set of semantic classes the object belongs to
        """
        if obj_cls_name not in OBJECT_TO_SEMANTIC_CLASSES:
            logging.warning("Unknown object class name: %s" % obj_cls_name)
        return OBJECT_TO_SEMANTIC_CLASSES.get(obj_cls_name, set())


def get_common_semantic_classes(obj_class1: str, obj_class2: str) -> set:
    """
    Find common semantic classes for two objects

    :param obj_class1: _description_
    :param obj_class2: _description_
    :return: list of comman semantic classes between these objects
    """
    if obj_class1 == obj_class2:
        return SemanticClass.get_all_objs_in_semcls(obj_class1)
    sem_classes1 = SemanticClass.get_all_objs_in_semcls(obj_class1)
    sem_classes2 = SemanticClass.get_all_objs_in_semcls(obj_class2)
    return sem_classes1.intersection(sem_classes2)


def check_obj_X_is_Y(name_str_X: str, name_str_Y: str) -> bool:
    """
    Check whether object X is Y where Y can be an object or a semantic class

    :param name_str_X: name string of X
    :param name_str_Y: name string of Y
    :return: true or false
    """
    if name_str_X == name_str_Y:
        return True
    elif (
        (name_str_X == "SinkBasin" and name_str_Y == "Sink")
        or (name_str_X == "Sink" and name_str_Y == "SinkBasin")
        or (name_str_X == "BathtubBasin" and name_str_Y == "Bathtub")
        or (name_str_X == "Bathtub" and name_str_Y == "BathtubBasin")
    ):
        return True
    elif name_str_Y in SemanticClass.get_all_semcls_for_obj(name_str_X):
        return True
    return False
