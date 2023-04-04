from enum import Enum, unique
from typing import List, Dict
from dataclasses import dataclass


@unique
class GoalArguments(Enum):
    NONE = 0  # for pad or when we don't care about the object
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
    WineBottle = 120  # above: object class names
    Appliances = 121  # below: semantic class names
    BoilContainers = 122
    Chairs = 123
    CleaningProducts = 124
    Computers = 125
    Condiments = 126
    Cookware = 127
    Dishware = 128
    Drinkware = 129
    Electronics = 130
    Food = 131
    FoodCookers = 132
    Fruit = 133
    Furniture = 134
    Glass = 135
    Ground = 136
    HygieneProducts = 137
    KitchenUtensils = 138
    Knives = 139
    Lamps = 140
    Lights = 141
    ManipulableKitchenUtils = 142
    MediaEntertainment = 143
    OccupiableReceptacles = 144
    OpenableReceptacles = 145
    PickableReceptacles = 146
    RoomDecorations = 147
    Silverware = 148
    SmallHandheldObjects = 149
    Soap = 150
    SportsEquipment = 151
    StandingReceptacles = 152
    StoveTopCookers = 153
    Tables = 154
    Tableware = 155
    Vegetables = 156
    WallDecor = 157
    WaterBottomReceptacles = 158
    WaterContainers = 159
    WaterSources = 160
    WaterTaps = 161
    WritingUtensils = 162  # above: semantic class names
    BoiledPotato = 163  # below: special goal object mentions
    BowlOfAppleSlices = 164
    BowlOfCookedPotatoSlices = 165
    BowlOfTomatoSlices = 166
    Coffee = 167
    CookedPotatoSlice = 168
    PlateOfAppleSlices = 169
    PlateOfCookedPotatoSlices = 170
    PlateOfLettuceSlices = 171
    PlateOfToast = 172
    PlateOfTomatoSlices = 173
    Salad = 174
    Sandwich = 175
    Toast = 176

SPECIAL_GOAL_ARG_MENTIONS = {
    "BoiledPotato"
    "BowlOfAppleSlices"
    "BowlOfCookedPotatoSlices"
    "BowlOfTomatoSlices"
    "Coffee"
    "CookedPotatoSlice"
    "PlateOfAppleSlices"
    "PlateOfCookedPotatoSlices"
    "PlateOfLettuceSlices"
    "PlateOfToast"
    "PlateOfTomatoSlices"
    "Salad"
    "Sandwich"
    "Toast"
}


@unique
class GoalConditions(Enum):
    NONE = 0  # for pad or when we don't care about the condition
    isCooked = 1
    isClean = 2
    isPickedUp = 3
    isFilledWithLiquid = 4
    isEmptied = 5
    isSliced = 6
    simbotIsBoiled = 7
    simbotIsFilledWithCoffee = 8
    parentReceptacles = 9
    EOS = 10


@unique
class GoalReceptacles(Enum):
    NONE = 0  # for pad or when we don't care about the receptacle
    ArmChair = 1
    Bathtub = 2
    BathtubBasin = 3
    Bed = 4
    Bowl = 5
    Box = 6
    Cabinet = 7
    Chair = 8
    CoffeeMachine = 9
    CoffeeTable = 10
    CounterTop = 11
    Cup = 12
    Desk = 13
    DiningTable = 14
    Drawer = 15
    Dresser = 16
    Fridge = 17
    GarbageCan = 18
    HandTowelHolder = 19
    LaundryHamper = 20
    Microwave = 21
    Mug = 22
    Ottoman = 23
    Pan = 24
    Plate = 25
    Pot = 26
    Safe = 27
    Shelf = 28
    SideTable = 29
    Sink = 30
    SinkBasin = 31
    Sofa = 32
    Stool = 33
    StoveBurner = 34
    TVStand = 35
    Toaster = 36
    Toilet = 37
    ToiletPaperHanger = 38
    TowelHolder = 39  # above: object class names
    Chairs = 40  # below: semantic class names
    Tables = 41
    Furniture = 42


# @unique
# class TaskNames(Enum):
#     pass


# @dataclass
# class TaskDefinitions:
#     # 101: Toast, 106: Plate Of Toast
#     MakeToast = TEAChTask(
#         taskId=101,
#         name="MakeToast",
#         Args={"number": ["1", "2"]},
#     )
#     # 102: Coffee
#     MakeCoffe = TEAChTask(
#         taskId=102,
#         name="MakeCoffe",
#         Args={"number": ["1", "2"]},
#     )
#     # 103: Clean X, 115: CleanAll X
#     CleanX = TEAChTask(
#         taskId=103,
#         name="CleanX",
#         Args={
#             "number": ["1", "2", "all"],
#             "patient": [
#                 "Bowl",
#                 "Cloth",
#                 "Cookware",
#                 "Cup",
#                 "Dishware",
#                 "Drinkware",
#                 "Mug",
#                 "Pan",
#                 "Plate",
#                 "Pot",
#                 "Tableware",
#             ],
#         },
#     )
#     # 104: Sliced X
#     MakeSlicedX = TEAChTask(
#         taskId=104,
#         name="MakeSlicedX",
#         Args={
#             "number": ["1", "2", "3", "4", "5"],
#             "patient": ["Apple", "Lettuce", "Tomato"],
#         },
#     )


# @dataclass
# class TEAChTask:
#     """Class for keeping track of an item in inventory."""

#     taskId: int
#     name: str
#     Args: Dict[str, List[str]]


################################################################
#####     code for generating the object argument list     #####
################################################################
# special_args = {
#     'BoiledPotato': 10,
#     'BowlOfAppleSlices': 2,
#     'BowlOfCookedPotatoSlices': 8,
#     'BowlOfTomatoSlices': 4,
#     'Coffee': 315,
#     'CookedPotatoSlice': 856,
#     'Plate Of Toast': 26,
#     'PlateOfAppleSlices': 6,
#     'PlateOfCookedPotatoSlices': 7,
#     'PlateOfLettuceSlices': 1,
#     'PlateOfTomatoSlices': 3,
#     'Salad': 80,
#     'Sandwich': 78,
#     'Toast': 1255
# }

# all_args = sorted(_INTERACTABLE_OBJECTS)
# all_args += sorted(list(SEMANTIC_CLS_TO_OBJECTS.keys()))
# all_args += sorted(list(special_args.keys()))

# strr = ''
# for idx, o in enumerate(all_args):
#     strr += "%s = %d\n"%(o, idx)
# print(strr)

# for o, aff in OBJECT_AFFORDANCE.items():
#     if aff and aff != ["moveable"]:
#         _INTERACTABLE_OBJECTS.add(o)

# receps = sorted([o for o, aff in OBJECT_AFFORDANCE.items() if 'receptacle' in aff])
# receps += ['Chairs', 'Tables', 'Furniture'] # semantic class as receptacles
# strr = ''
# for idx, o in enumerate(receps):
#     strr += "%s = %d\n"%(o, idx)
# print(strr)
