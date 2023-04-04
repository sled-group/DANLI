import os
import re
import copy
import pickle
import ujson as json
import spacy
import numpy as np
from tqdm import tqdm
from pprint import pprint, pformat
from collections import OrderedDict, Counter
from multiprocessing import Pool
from anytree import RenderTree, PostOrderIter
from anytree.importer import DictImporter
from anytree.search import findall

from .utils import ai2thor_utils as ai2
from .utils.tree_utils import TreeNode
from definitions.teach_object_semantic_class import SEMANTIC_CLS_TO_OBJECTS
from .definition.domain_definitions import MANI_ACTIONS, BURNER_TO_KNOB
from .definition.domain_definitions import CONDITION_TO_SUBGOAL, PROPS_TO_CHECK
from .definition.domain_definitions import add_extra_goal_conditions
from .definition.domain_definitions import get_prop_to_check
from .definition.domain_definitions import get_goal_instance_name
from .definition.action_definitions import HIGH_ACTIONS, NAVI_ACTIONS, ALL_ACTIONS


def link_game_file_to_edh_tfd(args):
    gamefiles_path = os.path.join(args.raw_data_dir, "all_game_files")
    edh_path = os.path.join(args.raw_data_dir, "edh_instances")
    tfd_path = os.path.join(args.raw_data_dir, "tfd_instances")

    game_id_to_split = {}
    game_id_to_edh_fn = {}
    game_id_to_tfd_fn = {}

    # create a mapping from game ids to edh instances file names
    # e.g. 0a122fa3133a2384_9492 -> [train/0a122fa3133a2384_9492.edh0.json, ...]
    for split in ["train", "valid_seen", "valid_unseen"]:
        edh_split_path = os.path.join(edh_path, split)
        cnt = 0
        for edh_fn in os.listdir(edh_split_path):
            game_id = edh_fn.split(".")[0]
            game_id_to_split[game_id] = split
            if game_id in game_id_to_edh_fn:
                game_id_to_edh_fn[game_id].append(os.path.join(split, edh_fn))
            else:
                game_id_to_edh_fn[game_id] = [os.path.join(split, edh_fn)]
            cnt += 1
        print(f"edh {split} {cnt}")
    # sort according to edh instance id
    for edhs in game_id_to_edh_fn.values():
        edhs.sort(key=lambda x: int(x.split(".")[-2][3:]))
    # pprint([(k, game_id_to_edh_fn[k]) for idx, k in enumerate(game_id_to_edh_fn) if idx < 3])

    # create a mapping from game ids to tfd instances file names
    for split in ["train", "valid_seen", "valid_unseen"]:
        tfd_split_path = os.path.join(tfd_path, split)
        cnt = 0
        for tfd_fn in os.listdir(tfd_split_path):
            game_id = tfd_fn.split(".")[0]
            if game_id in game_id_to_split:
                assert game_id_to_split[game_id] == split
            else:
                game_id_to_split[game_id] = split
            game_id_to_tfd_fn[game_id] = os.path.join(split, tfd_fn)
            cnt += 1
        print(f"tfd {split} {cnt}")

    for game_fn in os.listdir(gamefiles_path):
        game_id = game_fn.split(".")[0]
        if game_id not in game_id_to_split:
            game_id_to_split[game_id] = "test"

    with open(args.out_data_dir + "game_id_to_split.json", "w") as f:
        json.dump(game_id_to_split, f, sort_keys=False, indent=4)
    with open(args.out_data_dir + "game_id_to_tfd_fn.json", "w") as f:
        json.dump(game_id_to_tfd_fn, f, sort_keys=False, indent=4)
    with open(args.out_data_dir + "game_id_to_edh_fn.json", "w") as f:
        json.dump(game_id_to_edh_fn, f, sort_keys=False, indent=4)

    return game_id_to_split, game_id_to_edh_fn, game_id_to_tfd_fn


def get_task_goals(task, goal_flat, goal_tree, objects_meta, count_multiplier=1):
    # we represent the task goals in two ways:
    # 1. in a flat object-centric form stored in a dict. Keys are all relevant objects
    #    and values are each object's goal conditions
    # 2. in a structured tree using the anytree implementation: anytree.readthedocs.io

    def get_node_name(task):
        task_name, task_params = task["task_name"], task["task_params"]
        param_idx_base = 0 if "N" not in task_name else 1
        if "N " in task_name:
            task_name = task_name.replace("N ", "%s " % task_params[0])
        if "X" in task_name:
            task_name = task_name.replace("X", task_params[param_idx_base])
        if "Y" in task_name:
            task_name = task_name.replace("Y", task_params[param_idx_base + 2])
        return str(task["task_id"]) + ": " + task_name

    task_full_name = get_node_name(task)
    goal_instance_name = get_goal_instance_name(task_full_name, task["task_params"])
    goal_tree = TreeNode(task_full_name, parent=goal_tree)
    goal_tree.data = {
        "desc": task["desc"],
        "params": task["task_params"],
        "anchor": task["task_anchor_object"],
        "count": count_multiplier,
        "instance_name": goal_instance_name,
    }

    for subtask_v in task["components"].values():
        # first determine the number of this doing subtask
        try:
            count = int(subtask_v["determiner"]) * count_multiplier
        except:
            if subtask_v["determiner"] == "a":
                count = 1 * count_multiplier
            elif subtask_v["determiner"] == "all":
                count = 99
            else:
                raise ValueError("invalid determiner:", subtask_v["determiner"])
        if count == 0:  # if zero, skip the subtask
            continue

        if "task" in subtask_v:
            # the subtask can be further divided into lower level subtasks
            goal_flat, _ = get_task_goals(
                subtask_v["task"], goal_flat, goal_tree, objects_meta, count
            )
        else:
            # the subtask is a primitive one whose goal condition is defined by the
            # simulator object state conditions
            conditions = subtask_v["conditions"]
            if "objectType" in conditions:
                obj = conditions["objectType"]
            elif "simbotObjectClass" in conditions:
                obj = conditions["simbotObjectClass"]
            else:
                raise KeyError("Did not find a valid object name in: ", conditions)

            if obj not in goal_flat:
                goal_flat[obj] = {"count": 0}

            if subtask_v["instance_shareable"]:
                obj_cnt = 1
                goal_flat[obj]["count"] = 1
            else:
                obj_cnt = count
                goal_flat[obj]["count"] += count
            if obj_cnt >= 99:
                obj_cnt = 0
                for obj_meta in objects_meta:
                    if ai2.check_object_type_match(obj_meta["objectType"], obj):
                        obj_cnt += 1

            if "Sliced" in obj:
                obj_parent = obj[:-6]
                if obj_parent not in goal_flat:
                    # only consider slicing one object
                    goal_flat[obj_parent] = {"count": 1, "isSliced": 1}
                    # TreeNode(name=(obj_parent, 'isSliced', 1), parent=goal_tree,
                    #          data= {'count': 1, 'shareable': 1})

                    # just need to check the existence of a {obj}Sliced instance
                    TreeNode(
                        name=(obj, "sliceable", False),
                        parent=goal_tree,
                        data={"count": obj_cnt, "shareable": 0},
                    )

            for ck, cv in conditions.items():
                if ck in ["objectType", "simbotObjectClass"]:
                    continue
                if ck in goal_flat[obj] and goal_flat[obj][ck] != cv:
                    print(
                        "conflict condition for %s.%s: %s, %s"
                        % (obj, ck, goal_flat[obj][ck], cv)
                    )
                    raise ValueError
                goal_flat[obj][ck] = cv
                # for i in range(obj_cnt):
                if ck != "receptacle":
                    TreeNode(
                        name=(obj, ck, cv),
                        parent=goal_tree,
                        data={
                            "count": obj_cnt,
                            "shareable": int(subtask_v["instance_shareable"]),
                        },
                    )

    if task["relations"]:
        for rel in task["relations"]:
            assert len(rel["tail_entity_list"]) == 1
            for eid in range(len(rel["head_entity_list"])):
                if rel["head_determiner_list"][eid] == "0":
                    continue

                # resolve head object
                anchor_obj = rel["head_entity_list"][eid]
                if "task" in task["components"][anchor_obj]:
                    head_name = get_node_name(task["components"][anchor_obj]["task"])
                else:
                    head_name = None  # anchor_obj
                subtask = task["components"][anchor_obj]
                while "task" in subtask:
                    subtasks = subtask["task"]
                    anchor_obj = subtasks["task_anchor_object"]
                    subtask = subtasks["components"][anchor_obj]

                if "objectType" in subtask["conditions"]:
                    head = subtask["conditions"]["objectType"]
                elif "simbotObjectClass" in subtask["conditions"]:
                    head = subtask["conditions"]["simbotObjectClass"]
                else:
                    raise KeyError("Did not find a valid object name in: ", conditions)
                # for ck, cv in subtask['conditions'].items():
                #     if ck in ['objectType', 'simbotObjectClass']:
                #         continue
                #     else:
                #         head_req[ck] = cv

                # resolve tail object
                anchor_obj = rel["tail_entity_list"][0]
                if "task" in task["components"][anchor_obj]:
                    tail_name = get_node_name(task["components"][anchor_obj]["task"])
                else:
                    tail_name = None  # anchor_obj
                subtask = task["components"][anchor_obj]
                while "task" in subtask:
                    subtasks = subtask["task"]
                    anchor_obj = subtasks["task_anchor_object"]
                    subtask = subtasks["components"][anchor_obj]
                if "objectType" in subtask["conditions"]:
                    tail = subtask["conditions"]["objectType"]
                elif "simbotObjectClass" in subtask["conditions"]:
                    tail = subtask["conditions"]["simbotObjectClass"]
                else:
                    raise KeyError("Did not find a valid object name in: ", conditions)

                assert head in goal_flat
                assert tail in goal_flat

                # add the tail to head's parentReceptacles
                if "parentReceptacles" not in goal_flat[head]:
                    goal_flat[head]["parentReceptacles"] = []

                goal_flat[head]["parentReceptacles"].append(tail)

                # add the head to tail's receptacleObjectIds
                if "receptacleObjectIds" not in goal_flat[tail]:
                    goal_flat[tail]["receptacleObjectIds"] = []

                goal_flat[tail]["receptacleObjectIds"].append(head)

                # add a leaf node
                cnt = rel["head_determiner_list"][eid]
                if cnt == "all":
                    cnt = 0
                    for obj_meta in objects_meta:
                        if ai2.check_object_type_match(obj_meta["objectType"], head):
                            cnt += 1
                else:
                    cnt = 1 if cnt == "a" else int(cnt)
                TreeNode(
                    name=(head, "parentReceptacles", tail),
                    parent=goal_tree,
                    data={
                        "count": cnt,
                        "shareable": 0,
                        "head_name": head_name,
                        "tail_name": tail_name,
                    },
                )

                # cond_str = '%s->%s'%(rel['head_entity_list'][eid],
                #                         rel['tail_entity_list'][0])

    return goal_flat, goal_tree


def register_object(oid, oid_to_otype_id, otype_cnt):
    if oid is not None:
        poid = None if len(oid.split("|")) == 4 else "|".join(oid.split("|")[:-1])
        if oid not in oid_to_otype_id:
            # if poid is not None and poid in oid_to_otype_id:
            #     # register a child object according to its parent object id
            #     # (e.g. sliced/cracked)
            #     oid_to_otype_id[oid] = oid_to_otype_id[poid]
            # else:
            # ergister a non-child object
            otype = ai2.get_obj_type_from_oid(oid)  # oid.split('|')[0]
            if otype in otype_cnt:
                otype_cnt[otype] += 1
            else:
                otype_cnt[otype] = 1
            oid_to_otype_id[oid] = otype_cnt[otype] - 1

    return oid_to_otype_id, otype_cnt


def custom_world_state_update(world_state, aname, oid, picked_oid):
    # As replaying the trahjectories in the simulator is kind of slow
    # we manully keep tracking of a simplified world state here.
    # We only track objects locations, i.e., parentReceptacles and receptacleObjectIds,
    # whose changes are caused by pick up, place and slice actions.
    # *Note* our manully maintained world state might be different from the real metadata

    if "Pickup" in aname:
        # change the state of the receptacle of the picked object
        if world_state[oid]["parentReceptacles"] is not None:
            for recep in world_state[oid]["parentReceptacles"]:
                if oid in world_state[recep]["receptacleObjectIds"]:
                    world_state[recep]["receptacleObjectIds"].remove(oid)

        # change the state of the picked object
        world_state[oid]["lastParentReceptacles"] = world_state[oid][
            "parentReceptacles"
        ]
        world_state[oid]["parentReceptacles"] = None
        world_state[oid]["isPickedUp"] = True
        picked_oid = oid

    if "Place" in aname:
        # first check and delete task irrelevant actions, e.g., pick up something
        # and put it back.
        # From my observation, TEACh has such actions due to the turks' curiosity
        # of playing with the simulator when they are not clear what do do next.
        # delete = False
        # if 'Pickup' in prev_bot_action_event['aname'] and \
        #     world_state[picked_oid]["lastParentReceptacles"] is not None:
        #     for recep in world_state[picked_oid]["lastParentReceptacles"]:
        #         if recep == oid:
        #             delete = True
        # if delete:
        #     high_actions = high_actions[:-2]

        # change the state of the picked object
        world_state[picked_oid]["isPickedUp"] = False
        new_receps = world_state[oid]["parentReceptacles"]
        if new_receps is None:
            new_receps = [oid]
        else:
            new_receps = new_receps + [oid]
        world_state[picked_oid]["parentReceptacles"] = new_receps
        world_state[picked_oid]["directParentReceptacles"] = oid
        # change the state of its receptacle
        if world_state[oid]["receptacleObjectIds"] is not None:
            world_state[oid]["receptacleObjectIds"].append(picked_oid)
        else:
            world_state[oid]["receptacleObjectIds"] = [picked_oid]
        picked_oid = None

    if "Slice" in aname:
        parent_rec = world_state[oid]["parentReceptacles"]
        slice_number = 1 if "Egg" in oid else 9
        for i in range(slice_number):
            otype = oid.split("|")[0]
            if "Egg" in oid:
                sliced_oid = "%s|EggCracked_0" % (oid)
                sliced_otype = "EggCracked"
            else:
                sliced_oid = "%s|%sSliced_%d" % (oid, otype, i)
                sliced_otype = "%sSliced" % otype
            world_state[sliced_oid] = {
                "objectId": sliced_oid,
                "objectType": sliced_otype,
                "simbotObjectClass": [otype, "food"],
                "parentReceptacles": parent_rec,
                "receptacleObjectIds": None,
                "isPickedUp": False,
            }
            for recep in parent_rec:
                world_state[recep]["receptacleObjectIds"].append(sliced_oid)

    return world_state, picked_oid


def get_state_change(state_before, state_after, init_state):
    state_diff = {}
    obj_before = set(state_before.keys())
    obj_after = set(state_after.keys())

    for oid in obj_after.union(obj_before):
        state_diff[oid] = {}
        if oid in obj_before and oid in obj_after:
            props_before = get_prop_to_check(set(state_before[oid].keys()))
            props_after = get_prop_to_check(set(state_after[oid].keys()))
            for prop in props_after.union(props_before):
                if prop in props_before and prop in props_after:
                    val_before = state_before[oid][prop]
                    val_after = state_after[oid][prop]
                    if isinstance(val_before, list) and isinstance(val_after, list):
                        if set(val_before) != set(val_after):
                            state_diff[oid][prop] = val_after
                    else:
                        if val_before != val_after:
                            state_diff[oid][prop] = val_after
                elif prop in props_after:
                    val_after = state_after[oid][prop]
                    state_diff[oid][prop] = val_after
                else:
                    val_before = state_before[oid][prop]
                    if oid in init_state and prop in init_state[oid]:
                        state_diff[oid][prop] = init_state[oid][prop]
                    elif val_before in [True, False, 0, 1]:
                        state_diff[oid][prop] = int(not val_before)
                    else:
                        # pass
                        state_diff[oid][prop] = val_before

                # if 'isPickedUp' in props_before and \
                #    state_before[oid].get('simbotLastParentReceptacle') in ['Sink' in props_before and :

        elif oid in obj_after:
            props_after = set(state_after[oid].keys()).intersection(PROPS_TO_CHECK)
            if "sliceable" in state_after[oid]:
                props_after.add(
                    "sliceable"
                )  # just to mark the appearance of sliced/cracked pieces
            for prop in props_after:
                state_diff[oid][prop] = state_after[oid][prop]
            # print('AFTER')
            # print(oid)
            # pprint(state_after[oid])
        elif oid in obj_before:
            props_before = set(state_before[oid].keys()).intersection(PROPS_TO_CHECK)
            for prop in props_before:
                state_diff[oid][prop] = init_state[oid][prop]

        if not state_diff[oid]:
            del state_diff[oid]

    return state_diff


def check_goal_state(goal_tree, world_state, verbose=False):
    new_complete_nodes = []
    oid_to_alias = {}
    object_candidate_by_type = {}
    for node in PostOrderIter(goal_tree):  # should use post order iteration
        # Completion of subgoals (primitive predicate constraints)
        # print(node.name)
        if verbose:
            print("----------- node name:", node.name, "----------------")
        original_complete_num = len(node.data.get("goal_instances", set()))
        if node.is_leaf:  # and 'extra' not in node.data:
            node.data["goal_instances"] = set()
            otype, prop, val = node.name
            if otype not in object_candidate_by_type:
                object_candidate_by_type[otype] = {}
                for cand_oid, cand_obj in world_state.items():
                    cand_otype = ai2.get_obj_type_from_oid(cand_oid)
                    if ai2.check_object_type_match(cand_otype, otype):
                        object_candidate_by_type[otype].update({cand_oid: cand_obj})
                # print('-'*50)
                # print(otype, [vv for k, v in object_candidate_by_type.items() for vv in v])
                # if otype == 'BreadSliced':
                #     print([(oid, oprop.get('isSliced'), oprop.get('isCooked')) for oid, oprop in object_candidate_by_type[otype].items()])
                # if otype == 'Mug':
                #     print([(oid, oprop.get('simbotIsFilledWithCoffee')) for oid, oprop in object_candidate_by_type[otype].items()])

            if prop != "parentReceptacles":
                for cand_oid, cand_obj in object_candidate_by_type[otype].items():
                    if cand_obj.get(prop) == val:
                        node.data["goal_instances"].add(cand_oid)
                if len(node.data["goal_instances"]) > original_complete_num:
                    new_complete_nodes.append(node)
                node.data["complete"] = (
                    len(node.data["goal_instances"]) >= node.data["count"]
                )
            else:
                # For a spatial constraint subgoal ('obj1', 'parentReceptacles', 'obj2'),
                # check whether obj1 and obj2 meet all the properties specified by other subgoals
                # E.g. if the subgoal is to put a toast (cooked slice of bread) on a clean plate,
                # check whether there exists a clean plate that holds a cooked slice of bread.
                # The goal_instance for such subgoal nodes should be the tail object (e.g. the plate)
                if verbose:
                    print(
                        "original heads:", node.data.get("goal_instances_head", set())
                    )
                original_head_num = len(node.data.get("goal_instances_head", set()))
                node.data["goal_instances_head"] = set()
                complete = False
                if not node.data.get("tail_name"):
                    if val not in object_candidate_by_type:
                        object_candidate_by_type[val] = {}
                        for cand_oid, cand_obj in world_state.items():
                            cand_otype = ai2.get_obj_type_from_oid(cand_oid)
                            if ai2.check_object_type_match(cand_otype, val):
                                object_candidate_by_type[val].update(
                                    {cand_oid: cand_obj}
                                )
                        # if 'Sink' in val:
                        #     print('\n\n', '-'*50)
                        #     pprint([vv for v in object_candidate_by_type.values() for vv in v.keys()])
                        #     print( '-'*50, '\n\n')
                    cand_object_ids = set(object_candidate_by_type[val].keys())
                    # pprint([o for o in cand_object_ids.keys()])
                else:
                    tail_task = [
                        n for n in node.siblings if n.name == node.data["tail_name"]
                    ]
                    assert len(tail_task) == 1
                    tail_task = tail_task[0]
                    cand_object_ids = tail_task.data["goal_instances"]
                desired_recep_num = node.parent.data["count"]

                if not node.data.get("head_name"):
                    head_objects_to_check = set(object_candidate_by_type[otype].keys())
                else:
                    head_task = [
                        n for n in node.siblings if n.name == node.data["head_name"]
                    ]
                    assert len(head_task) == 1
                    head_task = head_task[0]
                    head_objects_to_check = head_task.data["goal_instances"]
                desired_obj_num_per_recep = node.data[
                    "count"
                ]  # int(node.data['count'] / desired_recep_num)

                if verbose:
                    print("desired_obj_num_per_recep", desired_obj_num_per_recep)
                    print("desired_recep_num", desired_recep_num)

                for cand_oid in cand_object_ids:
                    cand_obj = world_state[cand_oid]
                    simbot_recep = cand_obj.get("simbotIsReceptacleOf", [])
                    ithor_recep = cand_obj.get("receptacleObjectIds", [])
                    ithor_recep = [] if ithor_recep == None else ithor_recep
                    cand_obj_contains = set(simbot_recep).union(set(ithor_recep))
                    head_obj_be_contained = cand_obj_contains.intersection(
                        head_objects_to_check
                    )
                    if verbose:
                        print("ithor_recep", ithor_recep)
                        print("simbot_recep", simbot_recep)
                        print("cand_obj_contains", cand_obj_contains)
                        print("head_obj_be_contained", head_obj_be_contained)
                    # if len(head_obj_contained)>0:
                    #     node.data['goal_instances'].add(cand_oid)
                    if len(head_obj_be_contained) >= desired_obj_num_per_recep:
                        node.data["goal_instances"].add(cand_oid)
                        complete = True

                if verbose:
                    print("head_objects_to_check")
                    pprint(head_objects_to_check)
                    print("cand_object_ids")
                    pprint(cand_object_ids)

                for cand_oid in head_objects_to_check:
                    cand_obj = world_state[cand_oid]
                    simbot_parent = cand_obj.get("simbotLastParentReceptacle", None)
                    simbot_parent = (
                        set() if simbot_parent == None else set([simbot_parent])
                    )
                    ithor_parent = cand_obj.get("parentReceptacles", [])
                    ithor_parent = set() if ithor_parent == None else set(ithor_parent)
                    cand_obj_is_on = (
                        set(simbot_parent)
                        .union(ithor_parent)
                        .intersection(cand_object_ids)
                    )
                    if verbose:
                        print("cand_id:", cand_oid, "\ncand_obj_is_on:")
                        pprint(cand_obj_is_on)

                    if len(cand_obj_is_on) > 0:
                        node.data["goal_instances_head"].add(cand_oid)
                        if verbose:
                            print("adding head:", cand_oid)

                if verbose:
                    print("current heads:", node.data["goal_instances_head"])
                    print("original_complete_num", original_complete_num)
                    print("current goal_instances", node.data["goal_instances"])

                if (
                    len(node.data["goal_instances"]) > original_complete_num
                    or len(node.data["goal_instances_head"]) > original_head_num
                ):
                    new_complete_nodes.append(node)
                    if verbose:
                        print("new complete node!")
                        print("goal_instances:", node.data["goal_instances"])
                        print("goal_instances_head:", node.data["goal_instances_head"])
                node.data["complete"] = complete

        # Completion of tasks/subtasks
        # Note: different constraints for the same object type should be satisfied simultaneously
        # for the same object instance. Check if there exists such an object.
        if not node.is_leaf:
            node.data["goal_instances"] = set()
            child_of_interest = [n for n in node.children if "extra" not in n.data]
            child_type = [n.is_leaf for n in node.children if "extra" not in n.data]
            # print('-'*50)
            # print(node.name)
            # print([n.name for n in child_of_interest])
            if all(child_type):
                # Goal state of "primitive" subtasks (tasks whose goal conditions are all subgoals)
                # are justified by whether there exist enough instances that meet the joint goal conditions.
                # The instance that meets the joint subgoal can be obtained by get the intersection of
                # instances that meet each individual subgoal. Completion of such primitive subtasks should
                # be able to ground to specific object instances just like the completion of subgoals.
                goal_instances = child_of_interest[0].data["goal_instances"]
                # print(goal_instances)
                for child in child_of_interest[1:]:
                    # print(child.data['goal_instances'])
                    goal_instances = goal_instances.intersection(
                        child.data["goal_instances"]
                    )

                for oid in goal_instances:
                    oid_to_alias[oid] = node.data["instance_name"]

                # print(goal_instances)
                node.data["goal_instances"] = goal_instances
                node.data["complete"] = len(goal_instances) >= node.data["count"]
                if len(node.data["goal_instances"]) > original_complete_num:
                    new_complete_nodes.append(node)
                # print(node.data['complete'])
            else:
                # High-level tasks whose completion depends on the completion of other tasks. Such task should
                # only have a "count" of 1 in TEACh. The completion of such tasks depend on whether all its
                # children tasks/subgoals are completed or not. The goal instances should be the goal instance
                # defined in the spatial relation subgoal: e.g., put salad on the dining table indicates that
                # the goal instance of "Make Breakfast" is the DiningTable that holds the "salad" (e.g. plate
                # of sliced tomatoes and lettuces)

                complete = True
                for child in child_of_interest:
                    if not child.data["complete"]:
                        complete = False

                node.data["complete"] = complete

                rel_nodes = [
                    n for n in child_of_interest if n.name[1] == "parentReceptacles"
                ]
                assert (
                    len(rel_nodes) > 0
                ), f"task: {goal_tree.name} is compositional but has no relation subgoal!"
                for rel_node in rel_nodes:
                    node.data["goal_instances"] = node.data["goal_instances"].union(
                        rel_node.data["goal_instances"]
                    )
                for oid in node.data["goal_instances"]:
                    oid_to_alias[oid] = node.data["instance_name"]

                if len(node.data["goal_instances"]) > original_complete_num:
                    new_complete_nodes.append(node)

    return new_complete_nodes, oid_to_alias


def annotate_intentions_for_each_step(
    high_actions, intentions, goal_tree_aug, oid_to_alias_each_step, game_id=None
):
    # In check_goal_state we can only get the label whenever a subgoal/task is completed.
    # To obtain the intention label of ongoing subgoals/tasks, we first assign intention of steps
    # previous to a task completion as this task goal. E.g., when the task A is completed at step i
    # and task B is completed at step j, we assign the intention of step i+1,...,j as completing task B.
    # For subgoals, we manully set several rules to refine the annotation:
    # (1) if obejct Y is opened during picking up an object X and closing Y happens right after X is
    # picked up, closing Y is also considered as part of the "Get X" subgoal.
    # (2) if object X is toggled on in a subgoal (e.g. cook/clean Y), toggled it off is considered as
    # part of the subgoal although it may happens after the subgoal is completed. For example, the potato
    # will be cooked right after turning on the microwave and the subgoal is completed at this step,
    # turning down the microwave is also set as part of the cooking subgoal.
    # (3) if the subgoal is to "Place X on Y", and there are actions trying to clear Y before picking
    # up X and place it, then the intentions of such actions are not for "Get X" but to "Place X on Y".
    # Also the intention of the pick up action is assigned as "Place X on Y".

    def add_determinator(intent, count):
        # intent is in the form of "nnn: xxxxxxx"
        intent_list = list(intent)
        if not intent_list[5].isdigit():
            intent_list.insert(5, "%d " % count)
        else:
            intent_list[5] = str(count)
        return "".join(intent_list)

    def assign_sg(new_sg, goal_tree_aug, refined_intentions, tracked_intents):
        parents = findall(goal_tree_aug, filter_=lambda node: node.name == new_sg)
        if parents and len(parents) == 1:
            for n in parents[0].path:
                l = n.height
                intent = n.name
                if type(n.name) is str and n.name != "SKIP" and n.data["count"] != 1:
                    intent = add_determinator(intent, n.data["count"])
                refined_intentions[l] = tracked_intents[l] = intent
        else:
            refined_intentions[0] = tracked_intents[0] = new_sg

    level_num = goal_tree_aug.height + 1
    tracked_intents = [None for i in range(level_num)]
    processed_act = []
    new_sg_label = []
    for idx in range(len(high_actions) - 1, -1, -1):
        act = high_actions[idx]
        if act[1] == "USR" or act[2] == "dial":
            act.append([None for i in range(level_num)])
            new_sg_label.insert(0, None)
        else:
            refined_intentions = [None for i in range(level_num)]
            raw_intentions = intentions[idx]
            for level in range(level_num - 1, -1, -1):
                raw_intent_nodes = raw_intentions[level]
                # pprint(raw_intent_nodes)
                if len(raw_intent_nodes) == 1:
                    intent = raw_intent_nodes[0].name
                    if (
                        type(intent) is str
                        and intent != "SKIP"
                        and raw_intent_nodes[0].data["count"] != 1
                    ):
                        intent = add_determinator(
                            intent, raw_intent_nodes[0].data["count"]
                        )

                    refined_intentions[level] = intent
                    tracked_intents[level] = intent
                    if level == 0:
                        new_sg_label.insert(0, 1)
                elif raw_intent_nodes:  # disambiguation
                    raw_intent_names = [n.name for n in raw_intent_nodes]
                    if len(set(raw_intent_names)) == 1:
                        # if all the intents have the same name: no need to distinguish them
                        intent = raw_intent_names[0]
                        if level == 0:
                            new_sg_label.insert(0, 1)
                    else:
                        cand_intent_nodes = []
                        # first try: only keeps the key goal conditions
                        for raw_intent in raw_intent_nodes:
                            if "extra" not in raw_intent.data:
                                cand_intent_nodes.append(raw_intent)

                        if len(cand_intent_nodes) != 1:
                            # second try: only keeps those consistent with higher-level intentions
                            cand_intent_nodes = []
                            for raw_intent in raw_intent_nodes:
                                higher_level_intent_names = []
                                for hl_node in raw_intent.path[:-1]:
                                    if (
                                        type(hl_node.name) is str
                                        and hl_node.data["count"] != 1
                                    ):
                                        higher_level_intent_names.append(
                                            add_determinator(
                                                hl_node.name, hl_node.data["count"]
                                            )
                                        )
                                    else:
                                        higher_level_intent_names.append(hl_node.name)
                                if all(
                                    [
                                        n in refined_intentions
                                        for n in higher_level_intent_names
                                    ]
                                ):
                                    cand_intent_nodes.append(raw_intent)

                        if not cand_intent_nodes:
                            # if all the nodes are not matched with higher level intentions,
                            # we consider this step as an intermediate step for the subsequent subgoal
                            intent = tracked_intents[level]
                            if level == 0:
                                new_sg_label.insert(0, 0)
                        else:
                            # try:
                            #     assert len(cand_intent_nodes) == 1, (idx, refined_intentions, [[n.name for n in nn.path] for nn in cand_intent_nodes])
                            # except:
                            #     print(idx)
                            #     print('refined_intentions', refined_intentions)
                            #     print('cand_intent_nodes"')
                            #     pprint([[n.name for n in nn.path] for nn in cand_intent_nodes])
                            #     goal_tree_fn = "/home/zhangyic/project/EAI/ehsd_dev/analysis_dev/%s.png"%(game_fn)
                            #     goal_tree_aug.draw_tree(goal_tree_fn)
                            #     quit()
                            try:
                                # if still have multiple matched intentions, select the most "local" one
                                # E.g. ["A->B->C-D", "A->D"] : select "A->B->C-D"
                                cand_intent_nodes.sort(
                                    key=lambda x: len(x.path), reverse=True
                                )
                                intent = cand_intent_nodes[0].name
                                if (
                                    intent is str
                                    and cand_intent_nodes[0].data["count"] != 1
                                ):
                                    intent = add_determinator(
                                        intent, cand_intent_nodes[0].data["count"]
                                    )
                            except:
                                print(idx, level, game_id)  # game_id
                                print("refined_intentions", refined_intentions)
                                print("raw intents")
                                pprint(
                                    [
                                        [n.name for n in nn.path]
                                        for nn in raw_intent_nodes
                                    ]
                                )
                                print('cand_intent_nodes"')
                                pprint(
                                    [
                                        [n.name for n in nn.path]
                                        for nn in cand_intent_nodes
                                    ]
                                )
                                quit()

                            if level == 0:
                                new_sg_label.insert(0, 1)

                    refined_intentions[level] = intent
                    tracked_intents[level] = intent

                else:
                    # if there is no subgoal completed, we consider the action as an intermediate
                    # step of accomplishing the subsequent subgoal.
                    refined_intentions[level] = tracked_intents[level]
                    if level == 0:
                        new_sg_label.insert(0, 0)

            if refined_intentions[0] is not None:
                if (
                    "Place" in act[3]
                    and refined_intentions[0][1] == "parentReceptacles"
                ):
                    picked_oid = act[4].split(" ")[0]
                    if picked_oid in oid_to_alias_each_step[idx]:
                        ori_sg = refined_intentions[0]
                        new_sg = (
                            oid_to_alias_each_step[idx][picked_oid],
                            ori_sg[1],
                            ori_sg[2],
                        )
                        refined_intentions[0] = tracked_intents[0] = new_sg

                if "Pickup" in act[3] and refined_intentions[0][1] == "isPickedUp":
                    picking_oid = act[4].split(" ")[0]
                    if picking_oid in oid_to_alias_each_step[idx]:
                        ori_sg = refined_intentions[0]
                        new_sg = (
                            oid_to_alias_each_step[idx][picking_oid],
                            ori_sg[1],
                            ori_sg[2],
                        )
                        refined_intentions[0] = tracked_intents[0] = new_sg

                if len(processed_act) > 1:
                    next_act = high_actions[processed_act[0]]
                    next_next_act = high_actions[processed_act[1]]
                    if act[3] == "ToggleOn":
                        sg = refined_intentions[0]
                        if "Faucet" in act[4] and (
                            sg[1] != "isDirty" and "Fill" not in sg[1]
                        ):
                            # sometimes the player mistakenly clean something that is not dirty
                            # we manually correct the intentions of those actions
                            new_sg = (sg[0], "isDirty", 0)
                            assign_sg(
                                new_sg,
                                goal_tree_aug,
                                refined_intentions,
                                tracked_intents,
                            )
                            # refined_intentions[0] = tracked_intents[0] = new_sg
                        if (
                            "CoffeeMachine" in act[4]
                            and sg[1] != "simbotIsFilledWithCoffee"
                        ):
                            #  we manually correct the intentions of making coffee
                            new_sg = ("Mug", "simbotIsFilledWithCoffee", 1)
                            assign_sg(
                                new_sg,
                                goal_tree_aug,
                                refined_intentions,
                                tracked_intents,
                            )
                            # refined_intentions[0] = tracked_intents[0] = new_sg
                        if "Toaster" in act[4] and sg[1] != "isCooked":
                            #  we manually correct the intentions of making toast
                            new_sg = ("BreadSliced", "isCooked", 1)
                            assign_sg(
                                new_sg,
                                goal_tree_aug,
                                refined_intentions,
                                tracked_intents,
                            )
                            # refined_intentions[0] = tracked_intents[0] = new_sg
                        if next_act[3] == "ToggleOff":
                            # add ToggleOff as part of the intention of ToggleOn
                            next_act[7] = copy.deepcopy(refined_intentions)

                    elif act[3] == "Pickup" and next_act[3] == "Close":
                        next_act[7] = copy.deepcopy(refined_intentions)
                    elif act[3] == "Slice":
                        # place knife as part of slicing
                        if next_act[3] == "Place":
                            next_act[7] = copy.deepcopy(refined_intentions)
                        if next_act[3] == "Goto" and next_next_act[3] == "Place":
                            next_act[7] = copy.deepcopy(refined_intentions)
                            next_next_act[7] = copy.deepcopy(refined_intentions)

                    # elif (act[3] == 'Pickup' and 'Sink' in act[4]) and \
                    #         (next_act[3] == 'Place' and 'Sink' not in next_act[4]) and \
                    #         ('Cleaned' not in refined_intentions[0][0]):
                    #     refined_intentions[0] = ('Sink', 'isGettingClear', 1)
                    #     # higher-level intention should be 'Clean X', 'Coffee', 'Boil X', 'Water Plant'
                    #     if next_next_act[7][1] is not None and \
                    #             any([x in next_next_act[7][1] for x in ['Clean', 'Coffee', 'Boil', 'Water']]):
                    #         refined_intentions[1:] = copy.deepcopy(next_next_act[7][1:])
                    #         tracked_intents = copy.deepcopy(refined_intentions)
                    #     next_act[7] = copy.deepcopy(refined_intentions)

                    elif act[3] == "Pickup":
                        # we only add the "Get" intention for "Goto + Pick" combo
                        # otherwise, we consider the pick up action as a pre-step of the future intention"
                        for temp_idx in range(idx - 1, -1, -1):
                            temp_act = high_actions[temp_idx]
                            if temp_act[1] == "USR" or temp_act[2] == "dial":
                                continue
                            if temp_act[3] in ["Open", "Place", "ToggleOff"]:
                                continue
                            if temp_act[3] == "Goto":
                                break
                            else:
                                refined_intentions = copy.deepcopy(next_act[7])
                                new_sg_label[0] = 0
                                for level, intent in enumerate(refined_intentions):
                                    if intent is not None:
                                        tracked_intents[level] = intent

            processed_act = [idx] + processed_act

            act.append(refined_intentions)
            # print(idx, refined_intentions)

    for idx in processed_act:
        for level, sg in enumerate(high_actions[idx][7]):
            if sg == "SKIP":
                high_actions[idx][7][level] = None

    return new_sg_label


def get_initial_state(init_world_state, oid_to_otype_id, goal_conditions):
    # Process initial state of objects relevant to the task goal
    # Objects we are interested in:
    # (1) objects directly relevant to the completion of task (i.e., in goal condition)
    # (2) objects required to reach the goal, such as tool (knife, coffeemachine) or location
    # state of such objects we are interested in:
    # (1) state in the goal condition
    # (2) custom state: isOccupied, isInsideClosed,

    # for ometa in init_state_meta:
    #     oid = ometa['objectId']
    #     otype = ometa['objectType']
    #     oid_to_otype_id, otype_cnt = register_object(oid, oid_to_otype_id, otype_cnt)

    # reverse mapping from object type to object ids
    # e.g. Mug -> [Mug|+1.00|+1.00|+1.00, Mug|-1.00|-1.00|-1.00]
    otype_to_oid_list = {}
    for oid in oid_to_otype_id:
        oid_split = oid.split("|")
        if len(oid_split) == 5:
            otype = oid_split[-1].split("_")[0]
        else:
            otype = oid_split[0]
        if otype not in otype_to_oid_list:
            otype_to_oid_list[otype] = []

        otype_to_oid_list[otype].append(oid)

    for otype, oid_list in otype_to_oid_list.items():
        oid_list.sort(key=lambda x: oid_to_otype_id[x])

    # add relevant objects and states not specified in goal. Include:
    # (1) ON/OFF state of tooglable tools:
    #     Coffeemachine, Toaster, Faucet, Microwave, StoveKnob
    # (2) occupation state of functional receptcles:
    #     Sink, SinkBasin, Pan, Pot, StoveBurner, Microwave, Coffeemachine
    # (3) slice tool: Knife, ButterKnife

    def add_os(o_s_dict, objectType, attr=None, attr_value=None):
        if objectType in o_s_dict and attr is not None:
            o_s_dict[objectType].update({attr: attr_value})
        elif attr is not None:
            o_s_dict[objectType] = {attr: attr_value}
        else:
            o_s_dict[objectType] = {}

        return o_s_dict

    rel_obj_state = copy.deepcopy(goal_conditions)

    for otype, states in goal_conditions.items():
        if otype == "Mug" and "simbotIsFilledWithCoffee" in states:
            rel_obj_state = add_os(rel_obj_state, "CoffeeMachine", "isToggled", 0)
            rel_obj_state = add_os(
                rel_obj_state, "CoffeeMachine", "receptacleObjectIds", []
            )
        if "isDirty" in states:
            rel_obj_state = add_os(rel_obj_state, "Faucet", "isToggled", 0)
            rel_obj_state = add_os(rel_obj_state, "Sink", "receptacleObjectIds", [])
            rel_obj_state = add_os(
                rel_obj_state, "SinkBasin", "receptacleObjectIds", []
            )
        if "Sliced" in otype:
            rel_obj_state = add_os(rel_obj_state, "Knife")
            rel_obj_state = add_os(rel_obj_state, "ButterKnife")
        if "Sink" in goal_conditions:
            rel_obj_state = add_os(rel_obj_state, "Sink", "receptacleObjectIds", [])
            rel_obj_state = add_os(
                rel_obj_state, "SinkBasin", "receptacleObjectIds", []
            )
        if "simbotIsBoiled" in states:
            rel_obj_state = add_os(
                rel_obj_state, "StoveBurner", "receptacleObjectIds", []
            )
            rel_obj_state = add_os(rel_obj_state, "StoveKnob", "isToggled", 0)
            rel_obj_state = add_os(rel_obj_state, "Faucet", "isToggled", 0)
            rel_obj_state = add_os(rel_obj_state, "Pot", "isFilledWithLiquid", 0)
        if "simbotIsCooked" in states or "isCooked" in states:
            rel_obj_state = add_os(rel_obj_state, "StoveKnob", "isToggled", 0)
            rel_obj_state = add_os(
                rel_obj_state, "StoveBurner", "receptacleObjectIds", []
            )
            rel_obj_state = add_os(rel_obj_state, "Microwave", "isToggled", 0)
            rel_obj_state = add_os(
                rel_obj_state, "Microwave", "receptacleObjectIds", []
            )
            if otype in ["Bread", "BreadSliced"]:
                rel_obj_state = add_os(rel_obj_state, "Toaster", "isToggled", 0)
        if otype in otype_to_oid_list:
            oid = otype_to_oid_list[otype][0]
            if oid in init_world_state:
                ometa = init_world_state[oid]
                if ometa["canFillWithLiquid"]:
                    rel_obj_state = add_os(
                        rel_obj_state, otype, "isFilledWithLiquid", []
                    )

    # get the intial state for relevant objects in the environment
    init_rel_obj_states = {}
    for rel_otype, rel_state_values in rel_obj_state.items():
        if rel_otype in SEMANTIC_CLS_TO_OBJECTS:
            # TODO: add custom object classes
            continue
        # We only consider recording objects appears in the trajectory
        # except for two special cases:
        # (1) Sink and Sinkbasin: AI2THOR kind of messing them up
        # (2) We manully add the corresponding StoveKnobs to StoveBurners
        if rel_otype not in otype_to_oid_list:
            if rel_otype == "Sink" and "SinkBasin" in otype_to_oid_list:
                rel_otype = "SinkBasin"
            elif rel_otype == "StoveKnob" and "StoveBurner" in otype_to_oid_list:
                init_rel_obj_states[rel_otype] = []
                for burner in otype_to_oid_list["StoveBurner"]:
                    burner_name = init_world_state[burner]["name"]
                    if burner_name in BURNER_TO_KNOB:
                        knob_name = BURNER_TO_KNOB[burner_name]
                        for ometa in init_world_state.values():
                            if ometa["name"] == knob_name:
                                knob = ometa["objectId"]
                                break
                        s = "isToggled"
                        obj_state = {s: init_world_state[knob][s]}
                        init_rel_obj_states[rel_otype].append(obj_state)
                continue
            else:
                continue
        init_rel_obj_states[rel_otype] = []
        for oid in otype_to_oid_list[rel_otype]:
            if oid in init_world_state:
                ometa = init_world_state[oid]
                obj_state = {}
                for s in rel_state_values:
                    if s in ["count", "receptacle", "parentReceptacles"]:
                        continue
                    elif (
                        s == "receptacleObjectIds"
                        and init_world_state[oid][s] is not None
                    ):
                        obj_state[s] = []
                        for hold_oid in init_world_state[oid][s]:
                            oname = ai2.get_obj_name(hold_oid, oid_to_otype_id)
                            obj_state[s].append(oname)
                    else:
                        if s in init_world_state[oid]:
                            obj_state[s] = init_world_state[oid][s]
                        else:
                            obj_state[s] = 0

                o_recep = ai2.get_direct_parent_receptacle(oid, init_world_state)[0]
                obj_state["location"] = ai2.get_obj_name(o_recep, oid_to_otype_id)
                if ometa["pickupable"]:
                    container = ai2.check_inside_closed(oid, init_world_state)
                    obj_state["insideClosed"] = [
                        ai2.get_obj_name(o, oid_to_otype_id) for o in container
                    ]
                # print('obj_state', obj_state)
                init_rel_obj_states[rel_otype].append(obj_state)

    return init_rel_obj_states


def preprocess(job):
    data_path, save_path, game_id, split, job_idx, nlp, edh_files = job

    # if idx % 100 == 0 and idx != 0:
    # print('Processing: %d' % job_idx)

    game_fn_path = os.path.join(data_path, "all_game_files", "%s.game.json" % game_id)
    with open(game_fn_path, "r", encoding="utf-8") as f:
        game = json.load(f)

    definitions_actions = game["definitions"]["actions"]
    action_id_to_type = {}
    for a in definitions_actions:
        action_id_to_type[a["action_id"]] = a["action_name"]

    assert len(game["tasks"]) == 1
    task = game["tasks"][0]
    task_name = str(task["task_id"]) + ": " + task["task_name"]
    task_param = ", ".join(task["task_params"])
    grounded_task = task_name + " (%s)" % task_param
    episode = task["episodes"][0]
    assert episode["episode_id"] == game_id
    # split = game_id_to_split.get(game_id, 'test')
    init_state_meta = episode["initial_state"]["objects"]
    world_state = ai2.meta_to_world_state(init_state_meta)
    world_state_simbot = copy.deepcopy(world_state)
    init_world_state = copy.deepcopy(world_state)

    debug_games = [
        "2fbb46170662d917_06a8",
        "0fb77c7054bc7a0c_c243",
        "47cdabd6da727ed1_1151",
        "0a3ed01387e5ec38_4214",
    ]
    # if game_id not in debug_games:
    #     continue
    # else:
    #     split = 'train'

    # Typically one episode is recored in one game file,
    # while sometimes the episodes data stored twice
    try:
        assert len(task["episodes"]) == 1
    except:
        assert task["episodes"][0] == task["episodes"][1]

    # # TODO: REMOVE THIS PART OUT OF THIS FUNCTION
    # # whether the game file has the corresponding tfd/edh instances
    # has_tfd = False
    # if game_id in game_id_to_tfd_fn:
    #     tfd_fn = game_id_to_tfd_fn[game_id]
    #     with open(os.path.join(tfd_path, tfd_fn), 'r', encoding='utf-8') as f:
    #         tfd = json.load(f)
    #     has_tfd = True
    #     # assert 'dialog_history_cleaned' in tfd
    #         # print(tfd_fn)
    # has_edh = False
    # if game_id in game_id_to_edh_fn:
    #     edh_fn = game_id_to_edh_fn[game_id][-1]
    #     with open(os.path.join(edh_path, edh_fn), 'r', encoding='utf-8') as f:
    #         edh = json.load(f)
    #     has_edh = True
    #         # print(edh_fn)

    clean_dialog = None

    tfd_fn = os.path.join(data_path, "tfd_instances", split, "%s.tfd.json" % game_id)
    if os.path.exists(tfd_fn):
        with open(tfd_fn, "r", encoding="utf-8") as f:
            clean_dialog = json.load(f)["dialog_history_cleaned"]
    # extract grounded task goals
    goal_conditions, goal_tree = get_task_goals(
        task, OrderedDict(), None, init_state_meta
    )
    # pprint(goal_conditions)
    # print('Original Tree: ')
    # for pre, _, node in RenderTree(goal_tree):
    #     print("%s%s: %s" % (pre, node.name, str(node.data)))

    # print('Augmented Tree: ')
    goal_tree_aug = add_extra_goal_conditions(goal_tree)
    # for pre, _, node in RenderTree(goal_tree_aug):
    #     print("%s%s: %s" % (pre, node.name, str(node.data)))

    # print('_'*80)
    # print(game_id)
    check_goal_state(goal_tree_aug, world_state)

    # # if 'c9fcfe99c353ceda_8cb6' in game_fn:
    # for pre, _, node in RenderTree(goal_tree_aug):
    #     print("%s%s: %s" % (pre, node.name, str(node.data)))
    # tree_dict = goal_tree.to_dict()
    # pprint(tree_dict)
    # pickle.dump(goal_tree, open('test.pickle', 'wb'))
    # tree =  TreeNode.from_dict(tree_dict)
    # goal_tree_fn = "analysis/task_goals/%s.png"%(game_fn)
    # goal_tree.draw_tree(goal_tree_fn)
    # goal_tree_fn = "analysis/task_goals/%s_aug.png"%(game_fn)
    # goal_tree_aug.draw_tree(goal_tree_fn)

    # get ready for preprocessing
    # if task_name not in data_task_order:
    #     data_task_order[task_name] = {}
    #     data_task_order[task_name][task_param] = []
    # elif task_param not in data_task_order[task_name]:
    #     data_task_order[task_name][task_param] = []

    traj = {
        "game_id": game_id,
        "split": split,
        "task_name": task_name,
        "task_param": task_param,
    }
    low_actions = []
    high_actions = []
    intentions = []
    oid_to_alias_each_step = []
    sdiff_each_step = []

    """ 
    event in the raw data is recorded as a low action
    each low action is recorded as a tuple with 10 elements:
    (1) low action index;
    (2) corresponding high action index;
    (3) role (user/bot)
    (4) success (0/1);
    (5) action type (dial/sys/navi/mani);
    (6) action name;
    (7) arguments
    (8) time_start;

    low actions are turned into readable high action by:
    (1) merging low level navigation actions into a single GOTO action
    (2) removing failed actions
    each high action is recorded as a dict with 6 elements:
    (1) high action index;
    (2) role (user/bot)
    (3) action type (dial/sys/navi/mani);
    (4) action name
    (5) arguments;
    (6) readable action string
    (7) time_start
    (8) ongoing tasks (list of subgoal, higher level subtask goals etc);
    (9) intention event (intentions of beginning or finishing a subgoal);
    """

    goal_setup = False
    picked_oid = None
    otype_cnt = {}
    oid_to_otype_id = {}
    rel_obj_state = {}
    prev_succ_bot_non_dial_action_event = {}
    prev_mani_action_event = {"location": None}
    agent_posture = [0, 0]

    state_diffs = None
    statediff_file = os.path.join(data_path, f"state_diffs/{game_id}.json")
    if split != "test":
        assert os.path.exists(statediff_file)
        with open(statediff_file, "r") as f:
            state_diffs = json.load(f)

    dial_idx = 0
    for eidx, action_event in enumerate(episode["interactions"]):
        role = "BOT" if action_event["agent_id"] == 1 else "USR"
        action_id = action_event["action_id"]
        aname = action_id_to_type[action_event["action_id"]]
        success = action_event["success"]
        time = action_event["time_start"]

        # load world state of the current time step and next step to obtain state change

        # split = 'test'
        if state_diffs is not None and role == "BOT" and "oid" in action_event:
            state_before = state_diffs[str(time)]
            if eidx != len(episode["interactions"]) - 1:
                time_next = episode["interactions"][eidx + 1]["time_start"]
                state_after = state_diffs[str(time_next)]
                sdiff = get_state_change(state_before, state_after, init_world_state)
            else:
                try:
                    state_after = state_diffs["end"]
                    sdiff = get_state_change(
                        state_before, state_after, init_world_state
                    )
                except:
                    print(f"{game_id} does not has statediff.end.json")
                    sdiff = {}
        else:
            sdiff = {}

        if role == "USR":
            if "utterance" in action_event:
                atype = "dial"
                utter = (
                    clean_dialog[dial_idx][1]
                    if clean_dialog
                    else action_event["utterance"]
                )
                arg = " ".join([t.text for t in nlp(utter.lower())])
                arg_r = utter.capitalize()
                dial_idx += 1

                # we assume the user speaks
                goal_setup = True
            elif "query" in action_event:
                atype = "sys"
                arg = action_event["query"]
                arg_r = arg
            else:
                atype = "navi"
                arg = None
                arg_r = None

            lidx, hidx = len(low_actions), len(high_actions)
            low_action = [lidx, hidx, role, success, atype, aname, arg, time, None]
            low_actions.append(low_action)

            if atype == "dial":
                high_action = [len(high_actions), role, atype, aname, arg, arg_r, time]
                high_actions.append(high_action)
                intentions.append([[] for i in range(goal_tree_aug.height + 1)])
                oid_to_alias_each_step.append({})
                sdiff_each_step.append(sdiff)

        else:
            if "utterance" in action_event:
                # bot dialogue action
                atype = "dial"

                utter = (
                    clean_dialog[dial_idx][1]
                    if clean_dialog
                    else action_event["utterance"]
                )
                arg = " ".join([t.text for t in nlp(utter.lower())])
                arg_r = utter.capitalize()
                curr_pos = copy.deepcopy(agent_posture)
                dial_idx += 1

                lidx, hidx = len(low_actions), len(high_actions)
                low_action = [
                    lidx,
                    hidx,
                    role,
                    success,
                    atype,
                    aname,
                    arg,
                    time,
                    curr_pos,
                ]
                low_actions.append(low_action)

                high_action = [len(high_actions), role, atype, aname, arg, arg_r, time]
                high_actions.append(high_action)
                intentions.append([[] for i in range(goal_tree_aug.height + 1)])
                oid_to_alias_each_step.append({})
                sdiff_each_step.append(sdiff)

            else:
                # bot interaction (navigation/manipulation) action

                # object argument
                oid = action_event["oid"] if "oid" in action_event else None
                if oid is not None and "Bread_0" in oid:  # fix a bug
                    oid = oid.replace("Bread_0", "BreadSliced_0")
                oid_to_otype_id, otype_cnt = register_object(
                    oid, oid_to_otype_id, otype_cnt
                )
                curr_pos = copy.deepcopy(agent_posture)

                if success == 0:  # not goal_setup or
                    # before the task goal is told, or the action is failed

                    # crowdsource workers may play with some random actions for fun before
                    # the commander even setup the task goal. Interactions before the goal
                    # is set up are considered non-sense and not recorded as high actions.

                    # NOTE (1/5/2022): Due to the setup of EDH, we have to include the actions
                    # before the goal is setup, although this does not make sense.

                    # crowdsource workers may input invalid actions that are failed to be
                    # executed. Such failed interactions are not recorded as high actions.
                    atype = "mani" if action_id in MANI_ACTIONS else "navi"
                    arg = action_event["oid"] if "oid" in action_event else None
                    low_action = [
                        len(low_actions),
                        None,
                        role,
                        success,
                        atype,
                        aname,
                        arg,
                        time,
                        curr_pos,
                    ]
                    low_actions.append(low_action)
                else:
                    # successful interactions after the goal is set up
                    if "oid" not in action_event:
                        # navigation
                        atype = "navi"

                        lidx, hidx = len(low_actions), len(high_actions)
                        low_action = [
                            lidx,
                            hidx,
                            role,
                            success,
                            atype,
                            aname,
                            None,
                            time,
                            curr_pos,
                        ]
                        low_actions.append(low_action)

                        if aname == "Turn Right" and success:
                            agent_posture[0] -= 90
                        elif aname == "Turn Left" and success:
                            agent_posture[0] += 90
                        elif aname == "Look Up" and success:
                            agent_posture[1] -= 30
                        elif aname == "Look Down" and success:
                            agent_posture[1] += 30
                    else:
                        # manipulation
                        atype = "mani"
                        oids = ai2.get_obj_name(oid, oid_to_otype_id)

                        # get the location of the current manipulative action
                        o_loc = ai2.get_landmark_receptacle(oid, world_state)
                        if o_loc is None or "Floor" in o_loc:
                            o_loc = oid
                        oid_to_otype_id, otype_cnt = register_object(
                            o_loc, oid_to_otype_id, otype_cnt
                        )

                        if "Pickup" in aname and oid is not None:
                            # try:
                            if "directParentReceptacles" in world_state[oid]:
                                o_recep = world_state[oid]["directParentReceptacles"]
                            else:
                                o_recep = ai2.get_direct_parent_receptacle(
                                    oid, world_state
                                )[0]
                            oid_to_otype_id, otype_cnt = register_object(
                                o_recep, oid_to_otype_id, otype_cnt
                            )
                            o_receps = ai2.get_obj_name(o_recep, oid_to_otype_id)
                            arg = "%s %s" % (oid, o_recep)
                            arg_r = "%s FROM %s" % (oids, o_receps)
                        elif "Place" in aname and oid is not None:
                            picked_oids = ai2.get_obj_name(picked_oid, oid_to_otype_id)
                            arg = "%s %s" % (picked_oid, oid)
                            arg_r = "%s TO %s" % (picked_oids, oids)
                        elif "Pour" in aname and oid is not None:
                            picked_oids = ai2.get_obj_name(picked_oid, oid_to_otype_id)
                            arg = "%s %s" % (picked_oid, oid)
                            arg_r = "%s TO %s" % (picked_oids, oids)
                        else:
                            arg = "%s" % (oid)
                            arg_r = "%s" % (oids)

                        # we need to add a merged high-level navigation action before each manipulation action when
                        # (1) there is navigation before the manipulative action and
                        # (2) the location of the current manipulative changes
                        if (
                            "pose_delta" in prev_succ_bot_non_dial_action_event
                            and o_loc != prev_mani_action_event["location"]
                        ):

                            """
                            A high-level navigation has the form of "Goto Desitination FOR TargetObject"
                            TargetObject is object to be interacted with in the current manipulative action
                            Desitination is the parent receptacle of the object in the subsequenct
                            manipulative action as the target, while also record the targeted object
                            to be found in the desination. E.g., to pick up a knife, we need to navigate
                            to the countertop where the knife is placed.
                            """

                            oid_navi = None
                            if aname not in ["Open", "Close"]:
                                oid_navi = oid
                            else:
                                for action_event_furture in episode["interactions"][
                                    eidx + 1 :
                                ]:
                                    aname_future = action_id_to_type[
                                        action_event_furture["action_id"]
                                    ]
                                    success_future = action_event_furture["success"]
                                    if (
                                        success_future
                                        and aname_future not in ["Open", "Close"]
                                        and "oid" in action_event_furture
                                    ):
                                        oid_navi = action_event_furture["oid"]
                                        oid_to_otype_id, otype_cnt = register_object(
                                            oid_navi, oid_to_otype_id, otype_cnt
                                        )
                                        break

                            if oid_navi is None:
                                oid_navi = oid

                            locs_navi = ai2.get_obj_name(o_loc, oid_to_otype_id)
                            oids_navi = ai2.get_obj_name(oid_navi, oid_to_otype_id)

                            narg = "%s %s" % (o_loc, oid_navi)
                            narg_r = "%s FOR %s" % (locs_navi, oids_navi)

                            # align the low action index with the high level navigation action
                            navi_start_time = None
                            ptr = len(low_actions) - 1
                            while ptr >= 0:
                                low_a_temp = low_actions[ptr]
                                if (
                                    low_a_temp[2] == "BOT"
                                    and low_a_temp[4] == "navi"
                                    and low_a_temp[3] == 1
                                ):
                                    low_actions[ptr][1] = len(high_actions)
                                    navi_start_time = low_actions[ptr][7]
                                elif (
                                    low_a_temp[2] == "BOT"
                                    and low_a_temp[4] == "mani"
                                    and low_a_temp[3] == 1
                                ):
                                    break
                                ptr -= 1

                            high_action = [
                                len(high_actions),
                                role,
                                "navi",
                                "Goto",
                                narg,
                                narg_r,
                                navi_start_time,
                            ]
                            high_actions.append(high_action)
                            intentions.append(
                                [[] for i in range(goal_tree_aug.height + 1)]
                            )
                            oid_to_alias_each_step.append({})
                            sdiff_each_step.append({})
                            # print('BOT  Goto(%s -> %s)'%(o_loc, oid))

                            # ptr = len(low_actions) - 1
                            # if low_actions[ptr][2] == 'USR' and low_actions[ptr][5] == high_actions[-2][3]:
                            #     # low_actions[ptr][1] += 1
                            #     ptr -= 1

                        lidx, hidx = len(low_actions), len(high_actions)
                        low_action = [
                            lidx,
                            hidx,
                            role,
                            success,
                            atype,
                            aname,
                            arg,
                            time,
                            curr_pos,
                        ]
                        low_actions.append(low_action)

                        high_action = [
                            len(high_actions),
                            role,
                            atype,
                            aname,
                            arg,
                            arg_r,
                            time,
                        ]
                        high_actions.append(high_action)

                        # print('[[%d]]'%hidx, '%.1f'%time, aname+' '+arg_r, '-----')
                        # update simbot simulator world state
                        intent = [[] for i in range(goal_tree_aug.height + 1)]
                        oid_to_alias = {}
                        if sdiff:
                            for sdiff_oid in sdiff:
                                if sdiff_oid in state_after:
                                    state_after[sdiff_oid].update(sdiff[sdiff_oid])
                                else:
                                    state_after[sdiff_oid] = sdiff[sdiff_oid]
                                if sdiff_oid in world_state_simbot:
                                    world_state_simbot[sdiff_oid].update(
                                        state_after[sdiff_oid]
                                    )
                                else:
                                    world_state_simbot[sdiff_oid] = state_after[
                                        sdiff_oid
                                    ]

                            # if verbose:
                            #     print('\n\n', '-'*30, len(high_actions), '-'*30)
                            #     print(high_action)
                            new_complete_nodes, oid_to_alias = check_goal_state(
                                goal_tree_aug, world_state_simbot, verbose=False
                            )

                            # if len(high_actions) == 15:
                            #     print('14: ----------------------------')
                            #     print(high_action)
                            #     for pre, _, node in RenderTree(goal_tree_aug):
                            #         print("%s%s: %s" % (pre, node.name, str(node.data)))

                            # if len(high_actions) == 47:
                            #     print('46: ----------------------------')
                            #     print(high_action)
                            #     for pre, _, node in RenderTree(goal_tree_aug):
                            #         print("%s%s: %s" % (pre, node.name, str(node.data)))

                            # if len(high_actions) == 29:
                            #     print('28: ----------------------------')
                            #     print(high_action)
                            #     for pre, _, node in RenderTree(goal_tree_aug):
                            #         print("%s%s: %s" % (pre, node.name, str(node.data)))

                            #     print(high_action)
                            #     pprint([n.name for n in new_complete_nodes])
                            #     pprint([n.data['goal_instances'] for n in new_complete_nodes])
                            #     for pre, _, node in RenderTree(goal_tree_aug):
                            #         print("%s%s: %s" % (pre, node.name, str(node.data)))

                            # if len(high_actions) == 31:
                            #     print('30: ----------------------------')
                            #     print(high_action)
                            #     for pre, _, node in RenderTree(goal_tree_aug):
                            #         print("%s%s: %s" % (pre, node.name, str(node.data)))

                            #     print(high_action)
                            #     pprint([n.name for n in new_complete_nodes])
                            #     pprint([n.data['goal_instances'] for n in new_complete_nodes])
                            #     for pre, _, node in RenderTree(goal_tree_aug):
                            #         print("%s%s: %s" % (pre, node.name, str(node.data)))

                            # if len(high_actions) == 50:
                            #     print('49: ----------------------------')
                            #     print(high_action)
                            #     for pre, _, node in RenderTree(goal_tree_aug):
                            #         print("%s%s: %s" % (pre, node.name, str(node.data)))

                            #     print(high_action)
                            #     pprint([n.name for n in new_complete_nodes])
                            #     pprint([n.data['goal_instances'] for n in new_complete_nodes])
                            #     for pre, _, node in RenderTree(goal_tree_aug):
                            #         print("%s%s: %s" % (pre, node.name, str(node.data)))

                            # if len(high_actions) == 128:
                            #     print('127: ----------------------------')
                            #     print(high_action)
                            #     pprint([n.name for n in new_complete_nodes])
                            #     pprint([n.data['goal_instances'] for n in new_complete_nodes])
                            #     for pre, _, node in RenderTree(goal_tree_aug):
                            #         print("%s%s: %s" % (pre, node.name, str(node.data)))

                            # if oid_to_alias:
                            #     print(len(high_actions)-1)
                            #     pprint(oid_to_alias)

                            for node in new_complete_nodes:
                                if node.name[1] == "sliceable":
                                    continue
                                intent[node.height].append(node)

                            for level, temp_intent in enumerate(intent):
                                if len(temp_intent) == 1 and not any(
                                    intent[level + 1 :]
                                ):
                                    for n in temp_intent[0].path[:-1]:
                                        intent[n.height].append(n)

                                    # we need to manually add skip nodes as task levels may not be aligned
                                    # E.g., subgoal-> make coffee-> SKIP     -> breakfast,
                                    #       subgoal-> toast      -> sandwich -> breakfast
                                    for i in range(len(intent)):
                                        if not intent[i]:
                                            intent[i] = [TreeNode("SKIP")]

                                    break

                            # manully add intentions for placing a target object to somewhere else than the goal place
                            if not intent[0]:
                                if "Place" in aname and picked_oid in oid_to_alias:
                                    parent_name = oid_to_alias[picked_oid]
                                    picked_otype = ai2.get_obj_type_from_oid(picked_oid)
                                    recep_otype = ai2.get_obj_type_from_oid(oid)
                                    if recep_otype != "Floor":
                                        node_name = (
                                            picked_otype,
                                            "parentReceptacles",
                                            recep_otype,
                                        )

                                        # parents = findall(goal_tree_aug,
                                        #                   filter_=lambda node: node.data.get('instance_name') == parent_name)
                                        parents = findall(
                                            goal_tree_aug,
                                            filter_=lambda node: type(node.name)
                                            is tuple
                                            and node.name[:2] == node_name[:2],
                                        )
                                        # if parents:
                                        #     print('FIND PARENTS!', [p.name for p in parents], len(high_actions))

                                        # parent = parents[0]

                                        intent[0] = [TreeNode(name=node_name)]

                                        for parent in parents:
                                            for p in parent.path[:-1]:
                                                intent[p.height] += [p]
                                        # if parents:
                                        #     print(intent)
                                        for i in range(len(intent)):
                                            if not intent[i]:
                                                intent[i] = [TreeNode("SKIP")]
                            # if new_complete_nodes:
                            #     pprint([n.name for n in new_complete_nodes])
                        intentions.append(intent)
                        oid_to_alias_each_step.append(oid_to_alias)
                        sdiff_each_step.append(sdiff)
                        # print(prefix + aname)

                        prev_mani_action_event = action_event
                        prev_mani_action_event["location"] = o_loc

                if success == 1:
                    # We need to manully simulate the change of object locations
                    world_state, picked_oid = custom_world_state_update(
                        world_state, aname, oid, picked_oid
                    )

            if success == 1 and atype != "dial":
                prev_succ_bot_non_dial_action_event = action_event
                prev_succ_bot_non_dial_action_event["aname"] = aname

    # print('-'*50)
    # print('AT THE END OF THE PROCESSING (BEFORE INTENTION ANNOTATION): ')
    # for pre, _, node in RenderTree(goal_tree_aug):
    #     print("%s%s: %s" % (pre, node.name, str(node.data)))

    # print('-' * 50)
    # for hidx, h in enumerate(high_actions):
    #     hidx, role, atype, aname, arg, arg_r, time = h
    #     print('[%03d %s %s] %s %s' % (hidx, role, atype[0].upper(), aname, arg_r))
    #     pprint([[vvv.name for vv in v for vvv in vv.path[::-1]] for v in intentions[hidx]])
    # goal_tree_fn = "/home/zhangyic/data/teach-dataset/debug/%s.png" % (game_id)
    # goal_tree_aug.draw_tree(goal_tree_fn)

    new_sg_label = annotate_intentions_for_each_step(
        high_actions, intentions, goal_tree_aug, oid_to_alias_each_step, game_id=game_id
    )

    # for intent in high_actions[]

    # print('-' * 50)
    # print('AFTER INTENTION REFINEMENT: ')
    # for hidx, h in enumerate(high_actions):
    #     hidx, role, atype, aname, arg, arg_r, time, sg = h
    #     print('[%03d %s %s] %s %s' % (hidx, role, atype[0].upper(), aname, arg_r))
    #     pprint(sg)

    def language_like(intent):
        if type(intent) is tuple:
            if intent[1] == "parentReceptacles":
                s_str = "Place %s To %s" % (intent[0], intent[2])
            else:
                s_str = CONDITION_TO_SUBGOAL[(intent[1], intent[2])] + " " + intent[0]
        else:
            assert type(intent) is str, "invalid intent type: %s" % type(intent)
            s_str = intent[5:]
            if (
                "Breakfast" in s_str
                or "Toast" in s_str
                or "Coffee" in s_str
                or "Slice" in s_str
                or "Sandwich" in s_str
                or "Salad" in s_str
            ):
                s_str = "Make " + s_str
        return s_str

    level_num = goal_tree_aug.height + 1
    high_intentions = [[] for i in range(level_num - 1)]
    prev_sg, prev_idx = None, None
    for hidx, high_action in enumerate(high_actions):
        event = [[], []]
        intentions = high_action[7]

        for level, high_intent in enumerate(intentions[1:]):
            high_intentions[level].append(high_intent)

        curr_sg = intentions[0]
        if curr_sg is not None:
            if curr_sg != prev_sg:
                event[0].append(language_like(curr_sg))
                if prev_sg is not None:
                    high_actions[prev_idx][8][1].append(language_like(prev_sg))
            prev_sg, prev_idx = curr_sg, hidx

        high_action.append(event)

    if prev_sg is not None:
        high_actions[prev_idx][8][1].append(language_like(prev_sg))

    for high_intent_lists in high_intentions:
        for high_intent in set(high_intent_lists):
            if high_intent is None:
                continue
            beg = high_intent_lists.index(high_intent)
            end = high_intent_lists[::-1].index(high_intent)
            high_intent_str = language_like(high_intent)
            high_actions[beg][8][0].insert(0, high_intent_str)
            high_actions[len(high_actions) - end - 1][8][1].append(high_intent_str)

    for high_action in high_actions:
        event = high_action[8]
        for i in [0, 1]:
            intent_str = " ; ".join([e for e in event[i]])
            intent_str = (
                intent_str.replace("CounterTop", "countertop")
                .replace("CD", "cd")
                .replace("TV", "tv")
            )
            intent_str = " ".join(
                re.sub(r"([A-Z])", r" \1", intent_str).split()
            ).lower()
            event.append(intent_str)

        # new: add new subgoal event

    number_to_word = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
    }

    def check_goal_mention_in_dialog(task_full_name, dialog_context):
        task_id, task_name = task_full_name.split(": ")
        dialog_context = dialog_context.lower()
        if task_name[0].isdigit():
            number = task_name[0]
            number_w = number_to_word[number]
            kw_or = [[number, number_w]]
            if number == "2":
                kw_or[0].extend(["another", "one more"])
        else:
            kw_or = []

        kw = []
        if task_id == "101":
            kw = ["toast"]
        elif task_id == "102":
            kw = ["coffee"]
        elif task_id == "103":
            kw = ["clean"]
        elif task_id == "104":
            m = re.match(r"[\d]?[\s]?Sliced ([\w\s]+)", task_name)
            (obj,) = m.groups()
            kw = ["slice", obj.lower()]
        elif task_id == "105":
            m = re.match(r"[\d]?[\s]?Cooked Slice Of ([\w\s]+)", task_name)
            (obj,) = m.groups()
            kw = ["slice", "cook", obj.lower()]
        elif task_id == "106":
            kw = ["toast", "plate"]
        elif task_id == "107":
            m = re.match(r"\d+ Cooked Slices Of ([\w\s]+) In ([\w\s]+)", task_name)
            obj, recep = m.groups()
            kw = ["cook", obj.lower(), recep.lower()]
            kw_or.append(["slice", "piece"])
        elif task_id == "108":
            m = re.match(r"\d+ Slices Of ([\w\s]+) In ([\w\s]+)", task_name)
            obj, recep = m.groups()
            kw = [obj.lower(), recep.lower()]
            kw_or.append(["slice", "piece"])
        elif task_id == "110":
            kw = ["put", "all"]
        elif task_id == "111":
            kw = ["put", "all"]
        elif task_id == "112":
            kw = ["boil"]
        elif task_id == "114":
            kw = ["water"]
        elif task_id == "115":
            kw = ["clean", "all"]
        elif task_id == "301":
            kw = ["breakfast"]
        elif task_id == "302":
            kw = ["sandwich"]
        elif task_id == "303":
            kw = ["salad"]

        cond1 = all([k in dialog_context for k in kw])
        cond2 = all([any([k in dialog_context for k in kk]) for kk in kw_or])
        # print('cond1', cond1, 'cond2', cond2)
        if cond1 and cond2:
            return True
        return False

    dialogs = []
    tg_candidates = {}
    for hidx, high_action in enumerate(high_actions):
        hidx, role, atype, aname, arg, arg_r, time, intent, event = high_action
        tg = [g for g in intent[1:] if g not in ["SKIP", [], None]]
        for g in tg:
            tg_candidates[g] = 0  # 0: not mentioned, 1: mentioned

    # add naive subgoals
    dialogs = []
    for hidx, high_action in enumerate(high_actions):

        hidx, role, atype, aname, arg, arg_r, time, intent, event = high_action

        sg = intent[0]

        mentioned_tgs = []
        if atype == "dial":
            dialogs.append(arg)
            dialog_context_sofar = " ".join(dialogs)
            for task_goal, mentioned in tg_candidates.items():
                if not mentioned:
                    if check_goal_mention_in_dialog(task_goal, dialog_context_sofar):
                        tg_candidates[task_goal] = 1
                        mentioned_tgs.append(task_goal)

        if new_sg_label[hidx]:
            event.append(sg)
        else:
            event.append(None)

        event.append(mentioned_tgs)

        # print('[%03d %s %s] %s %s' % (hidx, role, atype[0].upper(), aname, arg_r))
        # print('            ', sg, '   ', new_sg_label[hidx])
        # print('            ', tg)
        # print('            ', event[4])
        # print('            ', event[5])
        # print(sdiff_each_step[hidx])
    # turn to language

    # print('-' * 50)
    # print('AFTER INTENTION REFINEMENT: ')
    # for hidx, h in enumerate(high_actions):
    #     hidx, role, atype, aname, arg, arg_r, sdiff, sg, event = h
    #     print('[%03d %s %s] %s %s' % (hidx, role, atype[0].upper(), aname, arg_r))
    #     pprint(sg)
    #     pprint(event)

    # get the initial for relevant objects (for better data interpretability only)
    init_rel_obj_states = get_initial_state(
        init_world_state, oid_to_otype_id, goal_conditions
    )

    # print(game_fn)
    # print(grounded_task)
    # pprint(otype_to_oid_list)
    # pprint(goal_conditions)
    # pprint(init_rel_obj_states)
    # quit()

    # record an episode
    traj["length"] = "step: %d, high_step: %d" % (len(low_actions), len(high_actions))
    traj["goal_state"] = goal_conditions
    traj["init_state"] = init_rel_obj_states
    traj["low_actions"] = low_actions
    traj["high_actions"] = high_actions

    time_to_lidx = {k[7]: k[0] for k in low_actions}
    time_to_hidx = {k[7]: k[1] for k in low_actions}
    edh_info = []
    if edh_files is not None:
        for edh_f in edh_files:
            with open(os.path.join(data_path, "edh_instances", edh_f)) as f:
                edh = json.load(f)

            # Sometimes the truncation point of an edh instance is in the middle of a navigation action.
            # And if someoen says something during that navigation process, it will be recorded in without
            # special handling. The following code tries to address this issue.

            pred_end_high_idx = None
            for a in edh["driver_actions_future"][::-1]:
                # find the last non-navigation action
                if a["obj_interaction_action"]:
                    if time_to_hidx[a["time_start"]] is not None:
                        pred_end_high_idx = time_to_hidx[a["time_start"]] + 1
                        # print(edh['instance_id'], pred_end_high_idx,
                        #       time_to_hidx[edh['driver_actions_future'][-1]['time_start']])
                    break

            if pred_end_high_idx is None:
                # Since the actions before the first user utterances are not recored as high actions,
                # we have to use the navigation destination frame as the ending point.
                # Should not happen after removing the goal setup trick.
                pred_end_high_idx = time_to_hidx[
                    edh["driver_actions_future"][-1]["time_start"]
                ]
                print("pred_end_high_idx is None:", edh_f)

            edh_info.append(
                {
                    "file_name": edh_f.split(".")[-2],
                    "pred_start_low_idx": edh["pred_start_idx"],
                    "pred_end_low_idx": time_to_lidx[
                        edh["driver_actions_future"][-1]["time_start"]
                    ],
                    "pred_start_high_idx": time_to_hidx[
                        edh["driver_actions_future"][0]["time_start"]
                    ],
                    # Minus 1 because the prediction has to condition on the last token in the history
                    "pred_end_high_idx": pred_end_high_idx,
                }
            )
    traj["edh_info"] = edh_info

    with open(
        os.path.join(save_path, "processed_gamefiles", "%s.json" % game_id), "w"
    ) as f:
        json.dump(traj, f, sort_keys=False, indent=4)

    return traj


def build_vocabs(data, args, truncate_freq=3):
    pred2lang_file = os.path.join(args.ithor_assets_dir, "predicate_to_language.json")
    obj2id_file = os.path.join(args.ithor_assets_dir, "object_to_id.json")
    with open(os.path.join(pred2lang_file), "r") as f:
        pred2lang = json.load(f)

    dial_tk_freq, intent_tk_freq = Counter(), Counter()
    for traj in data.values():
        for a in traj["high_actions"]:
            if a[2] == "dial":
                dial_tk_freq.update(a[4].split())
                dial_tk_freq.update(["speak"])
            else:
                predicates = [a[3]] + [
                    ai2.get_obj_type_from_oid(o) for o in a[4].split()
                ]
                for p in predicates:
                    dial_tk_freq.update(pred2lang[p].split())
            dial_tk_freq.update(a[8][2].split())
            intent_tk_freq.update(a[8][2].split())
    save_path = os.path.join(args.out_data_dir, "vocab")

    dial_freq = {}
    input_vocab_w2id = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[USR]": 2,
        "[BOT]": 3,
        "[EOS_USR]": 4,
        "[EOS_BOT]": 5,
        "[END_DIAL]": 6,
        "[BEG_TRAJ]": 7,
        "[BEG_DONE]": 8,
        "[BEG_TODO]": 9,
    }

    for w, cnt in dial_tk_freq.most_common():
        dial_freq[w] = cnt
        if w not in input_vocab_w2id and cnt >= truncate_freq:
            input_vocab_w2id[w] = len(input_vocab_w2id)

    intent_freq = {}
    intent_w2id = {"[PAD]": 0, "[END_DONE]": 1, "[END_TODO]": 2}
    for w, cnt in intent_tk_freq.most_common():
        intent_freq[w] = cnt
        intent_w2id[w] = len(intent_w2id)

    # Action to index vocabularies
    action_high_name2id = {"[PAD]": 0, "[BEG]": 1, "[END]": 2}
    for a in HIGH_ACTIONS:
        action_high_name2id[a] = len(action_high_name2id)

    action_navi_name2id = {"[PAD]": 0, "[BEG]": 1, "[END]": 2}
    for a in NAVI_ACTIONS:
        action_navi_name2id[a] = len(action_navi_name2id)

    action_all_name2id = {"[PAD]": 0, "[BEG]": 1, "[END]": 2}
    for a in ALL_ACTIONS:
        action_all_name2id[a] = len(action_all_name2id)

    object_name2id = {"[PAD]": 0, "[BEG]": 1, "[END]": 2, "None": 3}
    with open(obj2id_file, "r") as f:
        all_objects = json.load(f)["all"]
    for o in all_objects:
        object_name2id[o] = len(object_name2id)

    vocabs = {
        "input_vocab_word2id": input_vocab_w2id,
        "output_vocab_intent": intent_w2id,
        "output_vocab_action_high": action_high_name2id,
        "output_vocab_action_navi": action_navi_name2id,
        "output_vocab_action_all": action_all_name2id,
        "output_vocab_object": object_name2id,
        "dialog_word_freq": dial_freq,
        "intent_word_freq": intent_freq,
    }

    for k, w2id_dict in vocabs.items():
        with open(os.path.join(save_path, "%s.json" % k), "w") as f:
            json.dump(w2id_dict, f, sort_keys=False, indent=4)
    # print(action_names)
    # print(object_names)


def main(args):

    if not os.path.isdir(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    if not os.path.isdir(args.out_data_dir + "processed_gamefiles/"):
        os.makedirs(args.out_data_dir + "processed_gamefiles/")
    if not os.path.isdir(args.out_data_dir + "vocab/"):
        os.makedirs(args.out_data_dir + "vocab/")

    # tokenizer
    NLP = spacy.load("en_core_web_sm")

    # link game file to its edh/tfd instances
    game_id_to_split_file = os.path.join(args.out_data_dir, "game_id_to_split.json")
    # game_id_to_tfd_fn = os.path.join(args.out_data_dir, 'game_id_to_tfd_fn.json')
    game_id_to_edh_fn = os.path.join(args.out_data_dir, "game_id_to_edh_fn.json")
    if os.path.exists(game_id_to_split_file):
        with open(game_id_to_split_file, "r") as f:
            game_id_to_split = json.load(f)
        with open(game_id_to_edh_fn, "r") as f:
            game_id_to_edh_fn = json.load(f)
    else:
        print("Linking game id to split and edh/tfd instances")
        (
            game_id_to_split,
            game_id_to_edh_fn,
            game_id_to_tfd_fn,
        ) = link_game_file_to_edh_tfd(args)

    # preprocess
    gamefiles_dir = os.path.join(args.raw_data_dir, "all_game_files")
    all_paths = []
    for idx, game_fn in enumerate(os.listdir(gamefiles_dir)):
        game_fn_path = os.path.join(gamefiles_dir, game_fn)

        game_id = game_fn.split(".")[0]
        split = game_id_to_split.get(game_id, "test")
        if args.debug and split == "test":
            continue
        if args.debug and idx > 20:
            break
        # if '200129f403652491_54b7' not in game_fn_path:
        #     continue

        edh_files = game_id_to_edh_fn.get(game_id, None)
        all_paths.append(
            (args.raw_data_dir, args.out_data_dir, game_id, split, idx, NLP, edh_files)
        )

    print(
        "Preprocessing %d game files by %d workers ... "
        % (len(all_paths), args.num_workers)
    )
    with Pool(args.num_workers) as p:
        result = p.map_async(
            preprocess, all_paths, chunksize=len(all_paths) // args.num_workers + 1
        )
        result.wait()

    data, data_task_order = {}, {}
    for traj in result.get():
        task_name, task_param = traj["task_name"], traj["task_param"]
        if task_name not in data_task_order:
            data_task_order[task_name] = {}
            data_task_order[task_name][task_param] = []
        elif task_param not in data_task_order[task_name]:
            data_task_order[task_name][task_param] = []
        data_task_order[task_name][task_param].append(traj)
        data[traj["game_id"]] = traj
    assert len(data) == len(all_paths)

    print("Sorting data according to trajectory length ... ")
    for task_name, parameterized_tasks in data_task_order.items():
        for param in parameterized_tasks:
            data_task_order[task_name][param].sort(key=lambda x: len(x["high_actions"]))

    preprocessed_sorted_fn = "preprocessed_games_sort_by_task_length.json"
    with open(args.out_data_dir + preprocessed_sorted_fn, "w") as f:
        json.dump(data_task_order, f, sort_keys=False, indent=4)
    with open(args.out_data_dir + "preprocessed_games.json", "w") as f:
        json.dump(data, f, sort_keys=False, indent=4)

    # save one sample of each task for debugging
    partial = {}
    for task_name, parameterized_tasks in data_task_order.items():
        partial[task_name] = {}
        cnt = 0
        for param, tasks in parameterized_tasks.items():
            partial[task_name][param] = []
            for t in tasks:
                if cnt > 2:
                    break
                partial[task_name][param].append(t)
                cnt += 1

    partial_fn = preprocessed_sorted_fn.replace(".json", "_partial.json")
    with open(args.out_data_dir + partial_fn, "w") as f:
        json.dump(partial, f, sort_keys=False, indent=4)

    print("Building vocabularies ... ")
    build_vocabs(data, args)
