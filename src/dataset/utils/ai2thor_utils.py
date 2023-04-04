from pprint import pprint
from definitions.teach_object_semantic_class import SEMANTIC_CLS_TO_OBJECTS


def meta_to_world_state(objects_meta):
    return {obj["objectId"]: obj for obj in objects_meta}


def get_obj_type_from_oid(oid):
    parts = oid.split("|")
    if len(parts) == 4:
        return parts[0]
    else:
        return parts[-1].split("_")[0]


def check_object_type_match(otype1, otype2):
    if otype1 == otype2:
        return True
    elif (
        (otype1 == "SinkBasin" and otype2 == "Sink")
        or (otype1 == "Sink" and otype2 == "SinkBasin")
        or (otype1 == "BathtubBasin" and otype2 == "Bathtub")
        or (otype1 == "Bathtub" and otype2 == "BathtubBasin")
    ):
        return True
    elif otype1 in SEMANTIC_CLS_TO_OBJECTS.get(otype2, []):
        return True
    elif otype2 in SEMANTIC_CLS_TO_OBJECTS.get(otype1, []):
        return True
    else:
        return False


def check_is_receptcale_or_not(obj, recep):
    oid = obj["objectId"]
    rid = recep["objectId"]
    if "simbotLastParentReceptacle" in obj and obj["simbotLastParentReceptacle"] == rid:
        return True
    elif "simbotIsReceptacleOf" in recep and oid in recep["simbotIsReceptacleOf"]:
        return True
    elif "parentReceptacles" in obj and rid in obj["parentReceptacles"]:
        return True
    return False


def get_obj_name(objectId, oid_to_otype_id=None):
    if objectId is None:
        return None
    # oid_split = objectId.split("|")
    # otype = oid_split[0]
    otype = get_obj_type_from_oid(objectId)

    if oid_to_otype_id is None or objectId not in oid_to_otype_id:
        return otype

    otype_id = oid_to_otype_id[objectId]
    oname = otype + "_" + str(otype_id)
    # elif len(oid_split) == 5:
    #     # oname = oid_split[-1].replace(otype, '%s_%d_'%(otype, otype_id))
    #     oname = oid_split[-1].replace(otype, "%s_%d_" % (otype, otype_id))
    return oname


def sort_receptcales(receptacles, world_state):
    # oid: AI2THOR type objectId
    # world_state: a dictonary with objectIds as keys and their metadata as values
    # Return: a list of parent receptacle of the object sorted by its closeness to the object
    # E.g. in a world: bread -> plate -> fridge -> floor
    # the original parentReceptacles list for bread is unsorted: [floor, plate, fridge]
    # after sorting: [plate, fridge, floor]

    if receptacles is None:
        return None

    sorted_receptacles = []
    # for each receptacle in the parent receptacle list
    for i, candidate in enumerate(receptacles):
        if i == 0:
            sorted_receptacles.append(candidate)
        else:
            add_index = None
            for i, recep in enumerate(sorted_receptacles):
                recep_recep = world_state[recep]["parentReceptacles"]
                recep_recep = [] if recep_recep is None else recep_recep

                if candidate in recep_recep:
                    continue
                else:
                    add_index = i
                    break
            if add_index is None:
                sorted_receptacles.append(candidate)
            else:
                sorted_receptacles.insert(add_index, candidate)

    return sorted_receptacles


def get_direct_parent_receptacle(oid, world_state):
    # oid: AI2THOR type objectId
    # world_state: a dictonary with objectIds as keys and their metadata as values
    # Return: the direct parent receptacle of the object
    # E.g. if an apple is in a plate on the conutertop, its parentReceptacles will contain
    # both a "Plate" and a "Countertop". This method returns the "Plate" which is the direct
    # receptacle of the apple.
    # test case1: bread -> plate -> fridge -> floor
    #           : return plate
    # test case2: bread -> plate1 &  plate2 (support by both plates) -> countertop
    #           : return [plate1, plate2]

    direct_recep = []
    receptacles = world_state[oid]["parentReceptacles"]

    if receptacles is None:
        return [None]

    # for each receptacle in the parent receptacle list
    for i, candidate in enumerate(receptacles):

        is_direct_parent = True
        # check whether its another receptacle's parent receptacle
        for j, recep in enumerate(receptacles):
            if j == i:
                continue
            recep_recep = world_state[recep]["parentReceptacles"]
            recep_recep = [] if recep_recep is None else recep_recep
            if candidate in recep_recep:
                is_direct_parent = False
                break

        if is_direct_parent:
            direct_recep.append(candidate)

    if not direct_recep:
        return [None]
    return direct_recep


def get_landmark_receptacle(oid, world_state):
    # oid: AI2THOR type objectId
    # world_state: a dictonary with objectIds as keys and their metadata as values
    # Return: the landmark that can serve as a navigable destination of the object
    # we heuristically choose the first unpickable receptcale of the object as the
    # landmark. See the following examples to understand this.
    # test case1: mug -> coffeemachine -> countertop -> floor
    #           : return coffeemachine
    # test case2: apple -> bowl -> fridge -> floor
    #           : return fridge
    # test case3: bread -> plate1 &  plate2 (support by both plates) -> countertop
    #           : return countertop

    receptacles = world_state[oid]["parentReceptacles"]

    if receptacles is None:
        return None

    sorted_receptacles = sort_receptcales(receptacles, world_state)

    landmark = None
    for recep in sorted_receptacles:
        if not world_state[recep]["pickupable"]:
            landmark = recep
            break

    return landmark


def check_inside_closed(oid, world_state):
    # check whether the object is inside some closed container
    # oid: AI2THOR type objectId
    # world_state: a dictonary with objectIds as keys and their metadata as values
    # Return: the list of objectIds that contains the object and are closed
    receptacles = world_state[oid]["parentReceptacles"]
    if receptacles is None:
        return []

    insideClosed = []
    for recep in receptacles:
        if world_state[recep]["openable"] and not world_state[recep]["isOpen"]:
            insideClosed.append(recep)

    return insideClosed


if __name__ == "__main__":
    # test
    world_example1 = {
        "floor": {"parentReceptacles": None, "pickupable": False},
        "countertop": {"parentReceptacles": ["floor"], "pickupable": False},
        "plate": {"parentReceptacles": ["countertop"], "pickupable": True},
        "bowl": {"parentReceptacles": ["plate", "countertop"], "pickupable": True},
        "bread": {
            "parentReceptacles": ["plate", "countertop", "bowl"],
            "pickupable": True,
        },
    }

    print("-------------- world 1 ---------------")
    pprint(world_example1)
    oid = "floor"
    print(oid, "landmark:", get_landmark_receptacle(oid, world_example1))
    oid = "countertop"
    print(oid, "landmark:", get_landmark_receptacle(oid, world_example1))
    oid = "plate"
    print(oid, "landmark:", get_landmark_receptacle(oid, world_example1))
    oid = "bread"
    print(oid, "landmark:", get_landmark_receptacle(oid, world_example1))
    print(
        oid,
        "sorted:",
        sort_receptcales(world_example1[oid]["parentReceptacles"], world_example1),
    )
    print(oid, "direct:", get_direct_parent_receptacle(oid, world_example1))

    world_example2 = {
        "floor": {"parentReceptacles": None, "pickupable": False},
        "countertop": {"parentReceptacles": ["floor"], "pickupable": False},
        "coffeemachine": {
            "parentReceptacles": ["floor", "countertop"],
            "pickupable": False,
        },
        "mug": {
            "parentReceptacles": ["coffeemachine", "countertop"],
            "pickupable": True,
        },
    }
    oid = "mug"
    print("-------------- world 2 ---------------")
    pprint(world_example2)
    print(oid, "landmark:", get_landmark_receptacle(oid, world_example2))
    print(oid, "direct:", get_direct_parent_receptacle(oid, world_example2))

    world_example3 = {
        "countertop": {"parentReceptacles": None, "pickupable": False},
        "plate1": {"parentReceptacles": ["countertop"], "pickupable": True},
        "plate2": {"parentReceptacles": ["countertop"], "pickupable": True},
        "bread": {
            "parentReceptacles": ["plate2", "countertop", "plate1"],
            "pickupable": True,
        },
    }

    print("-------------- world 3 ---------------")
    pprint(world_example3)
    oid = "bread"
    print(oid, "landmark:", get_landmark_receptacle(oid, world_example3))
    print(oid, "direct:", get_direct_parent_receptacle(oid, world_example3))

    world_example4 = {
        "floor": {"parentReceptacles": None, "pickupable": False},
        "bread": {"parentReceptacles": ["floor"], "pickupable": True},
    }
    print("-------------- world 4 ---------------")
    pprint(world_example4)
    oid = "bread"
    print(oid, "landmark:", get_landmark_receptacle(oid, world_example4))
    print(oid, "direct:", get_direct_parent_receptacle(oid, world_example4))

    world_example5 = {
        "stovebuner": {"parentReceptacles": None, "pickupable": False},
        "pot1": {"parentReceptacles": ["stovebuner", "pot2"], "pickupable": True},
        "pot2": {"parentReceptacles": ["stovebuner", "pot1"], "pickupable": True},
        "bread": {
            "parentReceptacles": ["pot1", "stovebuner", "pot2"],
            "pickupable": True,
        },
    }
    print("-------------- world 4 ---------------")
    pprint(world_example5)
    oid = "bread"
    print(oid, "landmark:", get_landmark_receptacle(oid, world_example5))
    print(oid, "direct:", get_direct_parent_receptacle(oid, world_example5))
