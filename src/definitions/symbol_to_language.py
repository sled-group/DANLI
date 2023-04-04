# map object name with multiple words into split strings
MultiWordObjectsToLang = {
    "AirConditioner": "air conditioner",
    "AlarmClock": "alarm clock",
    "AluminumFoil": "aluminum foil",
    "AppleSliced": "apple slice",
    "ArmChair": "arm chair",
    "BaseballBat": "baseball bat",
    "BathtubBasin": "bathtub basin",
    "BreadSliced": "bread slice",
    "ButterKnife": "butter knife",
    "CabinetBody": "cabinet body",
    "CoffeeMachine": "coffee machine",
    "CoffeeTable": "coffee table",
    "CounterSide": "counter side",
    "CreditCard": "credit card",
    "DeskLamp": "desk lamp",
    "DiningTable": "dining table",
    "DishSponge": "dish sponge",
    "DogBed": "dog bed",
    "DomeLight": "dome light",
    "EggCracked": "egg cracked",
    "FloorLamp": "floor lamp",
    "GarbageBag": "garbage bag",
    "GarbageCan": "garbage can",
    "HandTowel": "hand towel",
    "HandTowelHolder": "hand towel holder",
    "HousePlant": "house plant",
    "KeyChain": "key chain",
    "LaundryHamper": "laundry hamper",
    "LettuceSliced": "lettuce slice",
    "LightFixture": "light fixture",
    "LightSwitch": "light switch",
    "PaperTowelRoll": "paper towel roll",
    "PepperShaker": "pepper shaker",
    "PotatoSliced": "potato slice",
    "RemoteControl": "remote control",
    "RoomDecor": "room decoration",
    "SaltShaker": "salt shaker",
    "ScrubBrush": "scrub brush",
    "ShelvingUnit": "shelving unit",
    "ShowerCurtain": "shower curtain",
    "ShowerDoor": "shower door",
    "ShowerGlass": "shower glass",
    "ShowerHead": "shower head",
    "SideTable": "side table",
    "SinkBasin": "sink basin",
    "SoapBar": "soap bar",
    "SoapBottle": "soap bottle",
    "SprayBottle": "spray bottle",
    "StoveBase": "stove base",
    "StoveBurner": "stove burner",
    "StoveKnob": "stove knob",
    "TVStand": "television stand",
    "TableTopDecor": "table top decoration",
    "TargetCircle": "target circle",
    "TeddyBear": "teddy bear",
    "TennisRacket": "tennis racket",
    "TissueBox": "tissue box",
    "ToiletPaper": "toilet paper",
    "ToiletPaperHanger": "toilet paper hanger",
    "TomatoSliced": "tomato slice",
    "TowelHolder": "towel holder",
    "VacuumCleaner": "vacuum cleaner",
    "WateringCan": "watering can",
    "WineBottle": "wine bottle",
}

MultiWordSemanticClassesToLang = {
    "BoilContainers": "containers suitable for boiling",
    "CleaningProducts": "cleaning products",
    "FoodCookers": "food cookers",
    "HygieneProducts": "hygiene products",
    "KitchenUtensils": "kitchen utensils",
    "ManipulableKitchenUtils": "manipulable kitchen utils",
    "MediaEntertainment": "media entertainment",
    "OccupiableReceptacles": "receptacles suitable for occupancy",
    "OpenableReceptacles": "openable receptacles",
    "PickableReceptacles": "pickable containers",
    "RoomDecor": "room decorations",
    "SmallHandheldObjects": "small handheld objects",
    "SportsEquipment": "sports equipment",
    "StandingReceptacles": "big unmoveable furnitures",
    "StoveTopCookers": "stove top cookers",
    "WallDecor": "wall decorations",
    "WaterBottomReceptacles": "sink or bathtub",
    "WaterContainers": "water containers",
    "WaterSources": "water sources",
    "WaterTaps": "water taps",
    "WritingUtensils": "writing utensils",
}

MultiWordTaskObjectsToLang = {
    "BoiledPotato": "boiled potato",
    "BowlOfAppleSlices": "bowl of apple slices",
    "BowlOfCookedPotatoSlices": "bowl of cooked potato slices",
    "BowlOfTomatoSlices": "bowl of tomato slices",
    "CookedPotatoSlice": "cooked potato slice",
    "PlateOfAppleSlices": "plate of apple slices",
    "PlateOfCookedPotatoSlices": "plate of cooked potato slices",
    "PlateOfLettuceSlices": "plate of lettuce slices",
    "PlateOfToast": "plate of toast",
    "PlateOfTomatoSlices": "plate of tomato slices",
}

ActionToLang = {
    "Goto": "go for",
    "Pickup": "pick up",
    "Place": "place {} to {}",
    "Open": "open",
    "Close": "close",
    "ToggleOn": "turn on",
    "ToggleOff": "turn off",
    "Slice": "slice",
    "Pour": "pour water in {} to {}",
    "Stop": "stop",
    "Forward": "move forward",
    "Backward": "move backward",
    "Turn Left": "turn left",
    "Turn Right": "turn right",
    "Look Up": "look up",
    "Look Down": "look down",
    "Pan Left": "pan left",
    "Pan Right": "pan right",
    "Text": "speak",
}


def obj_cls_symbol_to_language(obj_or_sem_cls_name: str) -> str:
    if obj_or_sem_cls_name in MultiWordObjectsToLang:
        return MultiWordObjectsToLang[obj_or_sem_cls_name]
    elif obj_or_sem_cls_name in MultiWordSemanticClassesToLang:
        return MultiWordSemanticClassesToLang[obj_or_sem_cls_name]
    elif obj_or_sem_cls_name in MultiWordTaskObjectsToLang:
        return MultiWordTaskObjectsToLang[obj_or_sem_cls_name]
    return obj_or_sem_cls_name.lower()


def action_to_language(action_name: str) -> str:
    if action_name in ActionToLang:
        return ActionToLang[action_name]
    raise ValueError("Unknown action: {}".format(action_name))


# This is used for convered a subgoal tuple to natural langauge form
GoalConditionToLang = {
    "isCooked": "cook {}",
    "isClean": "rinse {}",
    "isPickedUp": "get {}",
    "isFilledWithLiquid": "fill {} with water",
    "isEmptied": "empty {}",
    "isSliced": "slice {}",
    "simbotIsBoiled": "boil {}",
    "simbotIsCooked": "cook {}",
    "simbotIsFilledWithCoffee": "fill {} with coffee",
    "simbotIsFilledWithWater": "fill {} with water",
    "parentReceptacles": "place {} to {}",
    "isToggleOn": "turn on {}",
    "isToggleOff": "turn off {}",
    "isBroken": "break {}",
    "isUsedUp": "use up {}",
    "isOpen": "open {}",
    "isClosed": "close {}",
}


def state_to_predicate(goal_condition: tuple) -> str:
    subj, predicate, obj = goal_condition
    if predicate == "isToggled" and obj == 1:
        predicate = "isToggleOn"
    if predicate == "isToggled" and obj == 0:
        predicate = "isToggleOff"
    if predicate == "isDirty" and obj == 0:
        predicate = "isClean"
    if predicate == "isFilledWithLiquid" and obj == 0:
        predicate = "isEmptied"
    if predicate == "isOpen" and obj == 0:
        predicate = "isClosed"
    if predicate == "simbotIsCooked":
        predicate = "isCooked"
    return predicate


def subgoal_to_language(goal_condition: tuple) -> str:
    """
    Convert a subgoal tuple to natural language form.

    :param goal_condition: tuple of (subject, predicate, object)
                           For unary predicates, object is 0 or 1
    :return: languge form of the subgoal
    """
    subj, predicate, obj = goal_condition
    predicate = state_to_predicate(goal_condition)

    template = GoalConditionToLang[predicate]
    subj_name = obj_cls_symbol_to_language(subj)
    if predicate == "parentReceptacles":
        obj_name = obj_cls_symbol_to_language(obj)
        return template.format(subj_name, obj_name)
    return template.format(subj_name)


def task_to_language(task_name_with_id: str) -> str:
    task_id, task_name = task_name_with_id.split(": ")
    make_kw = ["Breakfast", "Toast", "Coffee", "Slice", "Sandwich", "Salad"]
    if any([kw in task_name for kw in make_kw]):
        return "make " + task_name.lower()
    return task_name.lower()


PredicateToLang = {
    "NONE": " none",
    "isCooked": " cook",
    "isClean": " rinse",
    "isPickedUp": " get",
    "isFilledWithLiquid": " fill water",
    "isEmptied": " empty",
    "isSliced": " slice",
    "simbotIsBoiled": " boil",
    "simbotIsFilledWithCoffee": " fill coffee",
    "parentReceptacles": " place",
    "EOS": " completed",
}

# Convert symbols to verb/noun phrases readly for pretrained language model encoding
def symbol_to_phrase(symbol: str) -> str:
    if symbol in PredicateToLang:
        return PredicateToLang[symbol]
    lang = obj_cls_symbol_to_language(symbol)
    return " " + lang

ConditionToVerb = {
    ("isToggled", 0): "Turn Off",
    ("isToggled", 1): "Turn On",
    ("isBroken", 1): "Break",
    ("isFilledWithLiquid", 1): "Fill",
    ("isFilledWithLiquid", 0): "Empty",
    ("isDirty", 0): "Rinse",
    ("isUsedUp", 1): "Use Up",
    ("isCooked", 1): "Cook",
    ("isSliced", 1): "Slice",
    ("isOpen", 1): "Open",
    ("isOpen", 0): "Close",
    ("isPickedUp", 1): "Get",
    ("simbotIsFilledWithCoffee", 1): "Fill Coffee",
    ("parentReceptacles", 1): "Place",
    ("simbotIsCooked", 1): "Cook",
    ("simbotIsBoiled", 1): "Boiling",  # To distinguish with the task name "Boil X"
    ("simbotIsFilledWithWater", 1): "Fill Water",
    ("isGettingClear", 1): "Clear",
}