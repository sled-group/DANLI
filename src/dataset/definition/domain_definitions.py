import json
import copy
from pprint import pprint
from tkinter.filedialog import Open
from anytree import PreOrderIter, RenderTree
from anytree.search import findall

from ..utils.tree_utils import TreeNode
from definitions.teach_object_semantic_class import (
    SEMANTIC_CLS_TO_OBJECTS,
    OBJECT_TO_SEMANTIC_CLASSES,
)
from definitions.teach_objects import OBJECT_AFFORDANCE

# ITHOR_ASSET_DIR = '/home/zhangyic/project/EAI/ehsd_dev/ithor_assets/'
ITHOR_ASSET_DIR = "./ithor_assets/"

# import pptree
NAVI_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 300]
MANI_ACTIONS = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212]
DIAL_ACTIONS = [100, 101, 102]


OBJECT_TO_ATTR = json.load(open(ITHOR_ASSET_DIR + "property_dict.json", "r"))

KNOB_TO_BURNER = json.load(open(ITHOR_ASSET_DIR + "knob_to_burner.json", "r"))
BURNER_TO_KNOB = {v: k for k, v in KNOB_TO_BURNER.items()}

PROPS_TO_CHECK = {
    "isToggled",
    "isBroken",
    "isDirty",
    "isUsedUp",
    "isSliced",
    "isOpen",
    "isPickedUp",  # 'simbotPickedUp', <-bug
    "receptacleObjectIds",
    "simbotIsReceptacleOf",
    "simbotLastParentReceptacle",
    "parentReceptacles",
    "isCooked",
    "simbotIsCooked",
    "simbotIsBoiled",
    "isFilledWithLiquid",
    "simbotIsFilledWithCoffee",
    "simbotIsFilledWithWater",
}


CONDITION_TO_SUBGOAL = {
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

# CONDITION_TO_COMPLETION = {
#     ('isToggled', 0): 'Turned Off',
#     ('isToggled', 1): 'Turned On',
#     ('isBroken', 1): 'is Broken',
#     ('isFilledWithLiquid', 1): 'Filled',
#     ('isFilledWithLiquid', 0): 'Emptied',
#     ('isDirty', 0): 'Rinsed',
#     ('isUsedUp', 1): 'Used Up',
#     ('isCooked', 1): 'Cooked',
#     ('isSliced', 1): 'Sliced',
#     ('isOpen', 1): 'Opened',
#     ('isPickedUp', 1): 'Picked',
#     ('simbotIsFilledWithCoffee', 1): 'Filled Coffee',
#     ('parentReceptacles', 1): 'Placed',
#     ('simbotIsCooked', 1): 'Cooked',
#     ('simbotIsBoiled', 1): 'Boiled',
#     ('simbotIsFilledWithWater', 1): 'Filled Water',
#     ('isGettingClear', 1): 'Cleared'
# }


def get_goal_instance_name(task_full_name, params):
    task_id, task_name = task_full_name.split(": ")
    if task_id == "103":
        return f"{params[0]}"
    elif task_id == "104":
        return f"{params[0]}Sliced"
    elif task_id == "105":
        return f"Cooked{params[0]}Slice"
    elif task_id == "106":
        return f"PlateOfToast"
    elif task_id == "107":
        # return f"{params[3]} of {params[0]} Cooked {params[1]} Slices"
        return f"{params[3]}OfCooked{params[1]}Slices"
    elif task_id == "108":
        # return f"{params[3]} of {params[0]} {params[1]} Slices"
        return f"{params[3]}Of{params[1]}Slices"
    elif task_id == "110":
        return params[2]
    elif task_id == "111":
        return params[2]
    elif task_id == "112":
        return f"Boiled{params[0]}"
    elif task_id == "113":
        return "EggCracked"
    elif task_id == "114":
        return "WateredPlant"
    elif task_id == "115":
        return f"{params[0]}"
    else:
        return task_name


def get_prop_to_check(obj_props):
    props = obj_props.intersection(PROPS_TO_CHECK)
    # if 'simbotPickedUp' in props and 'isPickedUp' in props:
    #     props.remove('isPickedUp')
    if "simbotIsCooked" in props and "isCooked" in props:
        props.remove("isCooked")
    # if ('simbotIsFilledWithWater' in props or'simbotIsFilledWithCoffee' in props) and \
    #     'isFilledWithLiquid' in props:
    #     props.remove('isFilledWithLiquid')
    if "simbotIsBoiled" in props:
        if "isCooked" in props:
            props.remove("isCooked")
        if "simbotIsCooked" in props:
            props.remove("simbotIsCooked")
    return props


def add_extra_goal_conditions(goal_tree):
    goal_tree_aug = copy.deepcopy(goal_tree)
    original_subgoal_nodes = list(goal_tree_aug.leaves)

    for sg in original_subgoal_nodes:
        parent = sg.parent
        obj, prop, val = sg.name

        add_subgoals = []

        if (
            obj in OBJECT_AFFORDANCE and "canFillWithLiquid" in OBJECT_AFFORDANCE[obj]
        ) or (
            obj in SEMANTIC_CLS_TO_OBJECTS
            and "canFillWithLiquid"
            in OBJECT_AFFORDANCE[list(SEMANTIC_CLS_TO_OBJECTS[obj])[0]]
        ):
            add_subgoals.append((obj, "isFilledWithLiquid", 0))

        # add task-specific subgoals (e.g. using tools)
        if "101: Toast" in parent.name:
            add_subgoals.append(("Knives", "isPickedUp", 1))
            add_subgoals.append((obj[:-6], "isSliced", 1))
            add_subgoals.append((obj[:-6], "isPickedUp", 1))
            add_subgoals.append(("BreadSliced", "parentReceptacles", "Toaster"))

        elif "102: Coffee" in parent.name:
            add_subgoals.append(("Mug", "isPickedUp", 1))
            add_subgoals.append(("Mug", "parentReceptacles", "CoffeeMachine"))

        elif "103: Clean" in parent.name:
            add_subgoals.append((obj, "isPickedUp", 1))
            add_subgoals.append((obj, "parentReceptacles", "Sink"))

        elif "104: Sliced" in parent.name:
            add_subgoals.append(("Knives", "isPickedUp", 1))
            add_subgoals.append((obj[:-6], "isSliced", 1))
            add_subgoals.append((obj[:-6], "isPickedUp", 1))

        elif "105: Cooked Slice Of" in parent.name:
            assert "Potato" in obj
            add_subgoals.append(("Knives", "isPickedUp", 1))
            add_subgoals.append((obj[:-6], "isSliced", 1))
            add_subgoals.append((obj[:-6], "isPickedUp", 1))
            add_subgoals.append((obj, "parentReceptacles", "Microwave"))
            add_subgoals.append((obj, "parentReceptacles", "Pan"))
            add_subgoals.append(("Pan", "parentReceptacles", "StoveBurner"))

        elif "110: Put All" in parent.name:
            add_subgoals.append((obj, "isPickedUp", 1))

        elif "111: Put All" in parent.name:
            add_subgoals.append((obj, "isPickedUp", 1))

        elif "112: Boil" in parent.name:
            assert "Potato" in obj
            add_subgoals.append((obj, "isPickedUp", 1))
            # add_subgoals.append((obj, "parentReceptacles", "BoilContainers"))
            # add_subgoals.append(("BoilContainers", "isFilledWithLiquid", 1))
            # add_subgoals.append(("BoilContainers", "isPickedUp", 1))

            add_subgoals.append((obj, "parentReceptacles", "Bowl"))
            add_subgoals.append(("Bowl", "isFilledWithLiquid", 1))
            add_subgoals.append(("Bowl", "isPickedUp", 1))
            add_subgoals.append(("Bowl", "parentReceptacles", "Microwave"))

            add_subgoals.append((obj, "parentReceptacles", "Pot"))
            add_subgoals.append(("Pot", "isFilledWithLiquid", 1))
            add_subgoals.append(("Pot", "isPickedUp", 1))
            add_subgoals.append(("Pot", "parentReceptacles", "StoveBurner"))

        elif "114: Water Plant" in parent.name:
            add_subgoals.append(("WaterContainers", "isPickedUp", 1))
            add_subgoals.append(("WaterContainers", "parentReceptacles", "Sink"))
            add_subgoals.append(("WaterContainers", "isFilledWithLiquid", 1))

        elif "115: Clean All" in parent.name:
            add_subgoals.append((obj, "isPickedUp", 1))
            add_subgoals.append((obj, "parentReceptacles", "Sink"))

        for sg in add_subgoals:
            if not findall(goal_tree_aug, filter_=lambda node: node.name == sg):
                TreeNode(name=sg, parent=parent, data={"count": 1, "extra": 1})

    return goal_tree_aug