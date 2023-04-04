"""
This file extracts structural information of objects and actions from meta data.
"""
import os
import json
from typing import Dict, List

# from pprint import pprint

LOAD_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(LOAD_DIR, "processed")
AI2THOR_RESOURCE_DIR = os.path.join(LOAD_DIR, "ai2thor_resources")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Metadata of object and their affordance are copied from the official doc at:
# https://ai2thor.allenai.org/ithor/documentation/objects/object-types (Dec 2021)
with open(os.path.join(LOAD_DIR, "ithor_documented_object_list.json")) as f:
    documented_object_list = json.load(f)
with open(os.path.join(LOAD_DIR, "ithor_documented_object_affordance.json")) as f:
    documented_affordance = json.load(f)
assert set(documented_object_list) == set(documented_affordance.keys())

# ------------------------------ Process affordance ------------------------------
# Get the affordance for each object in ithor
affordance_list = [
    "breakable",
    "cookable",
    "dirtyable",
    "canFillWithLiquid",
    "moveable",
    "openable",
    "pickupable",
    "receptacle",
    "sliceable",
    "toggleable",
    "canBeUsedUp",
]
affordance: Dict[str, List[str]] = {}
for k in sorted(documented_object_list):
    v = documented_affordance[k]
    affordance[k] = []
    if v == "":
        continue
    v = v.replace(" (Some)", "").split(", ")
    for aff in v:
        aff = aff.lower()
        if aff == "usedup":
            aff = "canBeUsedUp"
        if aff == "fillable":
            aff = "canFillWithLiquid"
        affordance[k].append(aff)
with open(os.path.join(SAVE_DIR, "object_affordance.json"), "w") as f:
    json.dump(affordance, f, indent=2)


# ------------------------------ Process TEACh actions ------------------------------
# Copy the default definitions
with open(os.path.join(LOAD_DIR, "default_definitions.json")) as f:
    teach_original_definitions = json.load(f)
with open(os.path.join(SAVE_DIR, "teach_original_definitions.json"), "w") as f:
    json.dump(teach_original_definitions, f, indent=2)
# Copy the original mapping from teach action integer id to name strings
with open(os.path.join(AI2THOR_RESOURCE_DIR, "action_idx_to_action_name.json")) as f:
    teach_original_action_id_to_name = json.load(f)
with open(os.path.join(SAVE_DIR, "teach_original_action_id_to_name.json"), "w") as f:
    json.dump(teach_original_action_id_to_name, f, indent=2)


# Map the object affordances to action applicabilities
actions = ["Pickup", "Place", "Open", "Close", "ToggleOn", "ToggleOff", "Slice", "Pour"]
action_applicability: Dict[str, List[str]] = {a: [] for a in actions}
for o, props in documented_affordance.items():
    props = props.lower()
    if "pick" in props:
        action_applicability["Pickup"].append(o)
    if "receptacle" in props:
        action_applicability["Place"].append(o)
    if "open" in props:
        action_applicability["Open"].append(o)
        action_applicability["Close"].append(o)
    if "toggle" in props:
        action_applicability["ToggleOn"].append(o)
        action_applicability["ToggleOff"].append(o)
    if "slice" in props:
        action_applicability["Slice"].append(o)
    if "fill" in props:
        action_applicability["Pour"].append(o)

with open(os.path.join(SAVE_DIR, "action_applicability.json"), "w") as f:
    json.dump(action_applicability, f, indent=2)


# ------------------------- Process object's semantic classes  -------------------------
with open(os.path.join(AI2THOR_RESOURCE_DIR, "custom_object_classes.json")) as f:
    teach_custom_obj_name_to_sem_cls = json.load(f)
teach_custom_sem_cls: Dict[str, List[str]] = {}
for obj_name, sem_classes in teach_custom_obj_name_to_sem_cls.items():
    if obj_name == "PaperTowel":
        obj_name = "PaperTowelRoll"  # Fix an inconsist naming bug
    for sem_cls in sem_classes:
        if sem_cls not in teach_custom_sem_cls:
            teach_custom_sem_cls[sem_cls] = []
        teach_custom_sem_cls[sem_cls].append(obj_name)
for sem_cls in teach_custom_sem_cls:
    teach_custom_sem_cls[sem_cls].sort()
with open(os.path.join(SAVE_DIR, "teach_custom_semantic_classes.json"), "w") as f:
    json.dump(teach_custom_sem_cls, f, sort_keys=True, indent=2)
