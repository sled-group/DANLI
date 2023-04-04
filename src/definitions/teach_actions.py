"""
Action definitions and basic APIs for SEAGULL.
"""
from typing import Union
from enum import Enum, unique


ACTION_NAME_MAPPING = {
    "TurnRight": "Turn Right",
    "TurnLeft": "Turn Left",
    "LookUp": "Look Up",
    "LookDown": "Look Down",
    "PanRight": "Pan Right",
    "PanLeft": "Pan Left",
}  # Mapping to TEACH action names


class IndexedActions(Enum):
    @classmethod
    def has_action(cls, action_name: str) -> bool:
        action_name = action_name.replace(" ", "")
        return action_name in cls.__members__

    @classmethod
    def get_num_actions(cls):
        return len(cls)

    @classmethod
    def name_to_id(cls, action_name: str) -> int:
        action_name = action_name.replace(" ", "")
        if cls.has_action(action_name):
            return cls[action_name].value
        raise ValueError(
            "Invalid %s action name: %s" % (cls.__name__[:-7], action_name)
        )

    @classmethod
    def id_to_name(cls, action_id: int) -> str:
        if action_id < cls.get_num_actions():
            return ACTION_NAME_MAPPING.get(cls(action_id).name, cls(action_id).name)
        raise ValueError("Invalid %s action index: %d" % (cls.__name__[:-7], action_id))


@unique
class NavigationActions(IndexedActions):
    """
    Navigation action names and their integer indexes.
    """

    Forward = 0
    Backward = 1
    TurnLeft = 2
    TurnRight = 3
    LookUp = 4
    LookDown = 5
    PanLeft = 6
    PanRight = 7


@unique
class InteractionActions(IndexedActions):
    """
    Interaction action names and their integer indexes.

    An interaction action consists both an action name defined here and a target
    object (a object/semantic class name) as its argument. During execution, the
    name has to be further mapped to a visual object presented in the bot's view.
    """

    Stop = 0  # decide to end session
    Pickup = 1
    Place = 2
    Open = 3
    Close = 4
    ToggleOn = 5
    ToggleOff = 6
    Slice = 7
    Pour = 8


@unique
class BotPrimitiveActions(IndexedActions):
    """
    All the bot's primitive actions executable in the environment
    """

    Forward = 0
    Backward = 1
    TurnLeft = 2
    TurnRight = 3
    LookUp = 4
    LookDown = 5
    PanLeft = 6
    PanRight = 7
    EndNavi = 8  # custom action added to indicate the end of a navigation
    Pickup = 9
    Place = 10
    Open = 11
    Close = 12
    ToggleOn = 13
    ToggleOff = 14
    Slice = 15
    Pour = 16
    Text = 17  # speak
    Stop = 18  # decide to end session
    Failed = 19  # Indicate a failure


# Applicable objects for each interaction action
INTERACTION_APPLICABILITY = {
    "Pickup": {
        "AlarmClock",
        "AluminumFoil",
        "Apple",
        "AppleSliced",
        "BaseballBat",
        "BasketBall",
        "Book",
        "Boots",
        "Bottle",
        "Bowl",
        "Box",
        "Bread",
        "BreadSliced",
        "ButterKnife",
        "Candle",
        "CD",
        "CellPhone",
        "Cloth",
        "CreditCard",
        "Cup",
        "DishSponge",
        "Dumbbell",
        "Egg",
        "EggCracked",
        "Fork",
        "HandTowel",
        "Kettle",
        "KeyChain",
        "Knife",
        "Ladle",
        "Laptop",
        "Lettuce",
        "LettuceSliced",
        "Mug",
        "Newspaper",
        "Pan",
        "PaperTowelRoll",
        "Pen",
        "Pencil",
        "PepperShaker",
        "Pillow",
        "Plate",
        "Plunger",
        "Pot",
        "Potato",
        "PotatoSliced",
        "RemoteControl",
        "SaltShaker",
        "ScrubBrush",
        "SoapBar",
        "SoapBottle",
        "Spatula",
        "Spoon",
        "SprayBottle",
        "Statue",
        "TableTopDecor",
        "TeddyBear",
        "TennisRacket",
        "TissueBox",
        "ToiletPaper",
        "Tomato",
        "TomatoSliced",
        "Towel",
        "Vase",
        "Watch",
        "WateringCan",
        "WineBottle",
    },
    "Place": {
        "ArmChair",
        "Bathtub",
        "BathtubBasin",
        "Bed",
        "Bowl",
        "Box",
        "Cabinet",
        "Chair",
        "CoffeeMachine",
        "CoffeeTable",
        "CounterTop",
        "Cup",
        "Desk",
        "DiningTable",
        "Drawer",
        "Dresser",
        "Fridge",
        "GarbageCan",
        "HandTowelHolder",
        "LaundryHamper",
        "Microwave",
        "Mug",
        "Ottoman",
        "Pan",
        "Plate",
        "Pot",
        "Safe",
        "Shelf",
        "SideTable",
        "Sink",
        "SinkBasin",
        "Sofa",
        "Stool",
        "StoveBurner",
        "Toaster",
        "Toilet",
        "ToiletPaperHanger",
        "TowelHolder",
        "TVStand",
    },
    "Open": {
        "Blinds",
        "Book",
        "Box",
        "Cabinet",
        "Drawer",
        "Fridge",
        "Kettle",
        "Laptop",
        "Microwave",
        "Safe",
        "ShowerCurtain",
        "ShowerDoor",
        "Toilet",
    },
    "Close": {
        "Blinds",
        "Book",
        "Box",
        "Cabinet",
        "Drawer",
        "Fridge",
        "Kettle",
        "Laptop",
        "Microwave",
        "Safe",
        "ShowerCurtain",
        "ShowerDoor",
        "Toilet",
    },
    "ToggleOn": {
        "Candle",
        "CellPhone",
        "CoffeeMachine",
        "DeskLamp",
        "Faucet",
        "FloorLamp",
        "Laptop",
        "LightSwitch",
        "Microwave",
        "ShowerHead",
        "StoveBurner",
        "StoveKnob",
        "Television",
        "Toaster",
    },
    "ToggleOff": {
        "Candle",
        "CellPhone",
        "CoffeeMachine",
        "DeskLamp",
        "Faucet",
        "FloorLamp",
        "Laptop",
        "LightSwitch",
        "Microwave",
        "ShowerHead",
        "StoveBurner",
        "StoveKnob",
        "Television",
        "Toaster",
    },
    "Slice": {"Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato"},
    "Pour": {
        "Bottle",
        "Bowl",
        "Cup",
        "HousePlant",
        "Kettle",
        "Mug",
        "Pot",
        "WateringCan",
        "WineBottle",
    },
}


def get_valid_objects_for_interaction(action_str: str) -> Union[set, None]:
    """
    Return applicable objects for a given interaction action

    :param action_str: name string of an interaction action
    :return: either a set of applicable objects or `None` for actions that do
             not take object as argument or can be applied to any objects.
    """
    if not InteractionActions.has_action(action_str):
        raise ValueError("`%s` is not a valid interaction action" % action_str)
    if action_str in ["Stop", "Goto", "Find"]:
        return None
    return INTERACTION_APPLICABILITY[action_str]
