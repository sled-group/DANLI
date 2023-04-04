"""
This is primarily for the purpose of converting from subgoals to PDDL.
Some preprocessing done in between.
"""
import os
import textwrap
from typing import List, Tuple, Optional
from random import randrange
import numpy as np
from pkg_resources import compatible_platforms

from definitions.teach_object_semantic_class import SemanticClass
from definitions.teach_objects import get_object_affordance
from definitions.teach_object_state import create_default_object_state
from definitions.teach_tasks import SPECIAL_GOAL_ARG_MENTIONS
from definitions.teach_receptacle_compatibility import RECEPTACLE_COMPATIBILITY

from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.symbolic_world_repr import TeachSymbolicWorld, BigReceptacles

ManipulableKitchenUtils = SemanticClass.get_all_objs_in_semcls("ManipulableKitchenUtils")
FoodCookers = SemanticClass.get_all_objs_in_semcls("FoodCookers")
WaterTaps = SemanticClass.get_all_objs_in_semcls("WaterTaps")
WaterBottomReceptacles = SemanticClass.get_all_objs_in_semcls("WaterBottomReceptacles")
Knives = SemanticClass.get_all_objs_in_semcls("Knives")
BoilContainers = SemanticClass.get_all_objs_in_semcls("BoilContainers")
WaterContainers = SemanticClass.get_all_objs_in_semcls("WaterContainers")

class TeachPDDLProblem:
    """
    A Problem is formed from one subgoal. A subgoal consists of 
    several predicates (in the form of tuples).
    """
    
    DUMMY_OBJECT_DISTANCE = 100
    
    def __init__(self, subgoal: TeachSubgoal, symbolic_state_dict: dict):
        self.subgoal = subgoal
        self.symbolic_state_dict = symbolic_state_dict

        # initialize predicates and objects
        self.objects_str, self.init_state_str = self.state_to_pddl_init(symbolic_state_dict)

        # generate the PDDL goal
        self.goal_str = self.subgoal_to_pddl_goal(subgoal)

        # generate a problem name
        self.problem_name = subgoal.to_string() + '_{}'.format(randrange(999999))

        self.pddl_problem_str = (
            f"(define (problem {self.problem_name})\n"
            "(:domain teach)\n"
            f"{self.objects_str}"
            f"{self.init_state_str}"
            f"{self.goal_str}"
            "(:metric minimize (total-cost))\n"
            ")\n"
        )
    
    def save_problem(self, save_path):
        """
        Save the PDDL problem string to a PDDL domain file
        """
        
        # Use template engine to generate PDDL file
        # to save the results
        with open(os.path.join(save_path, "%s.pddl"%self.problem_name), "w") as fh:
            fh.write(self.pddl_problem_str)
            print(f'File {fh.name} written successfully')

        return fh.name


    @classmethod
    def state_to_pddl_init(cls, symbolic_state_dict: dict) -> str:
        """
        Translate a perceived symbolic state to a PDDL init state.
        """
        assert symbolic_state_dict, "Symbolic state dict cannot be empty!"
        
        objects_definition_str = "bot - Agent\n" # START_LOC - InteractiveObject\n"
        predicate_definition_str = ""
        distance_definition_str = ""
        

        for instance_id, instance_state in symbolic_state_dict.items():
            object_type = instance_state["objectType"]

            # add the object type to the objects definition
            objects_definition_str += "{} - {}\n".format(instance_id, object_type)

            # add special predicates
            if object_type in ["Mug", "CoffeeMachine", "Toaster", "Microwave", "StoveBurner", "BreadSliced"]:
                predicate_definition_str += "(is{} {})\n".format(object_type, instance_id)
            for aff in get_object_affordance(object_type):
                if aff in ['boilable', 'dirtyable', 'cookable', 'canFillWithLiquid']:
                    aff = aff.replace('canFillWithLiquid', 'fillable')
                    predicate_definition_str += "(is{} {})\n".format(aff.capitalize(), instance_id)
            for semslc in SemanticClass.get_all_semcls_for_obj(object_type):
                if semslc in ['WaterBottomReceptacles', 'StoveTopCookers', 'WaterTaps']:
                    predicate_definition_str += "(is{} {})\n".format(semslc, instance_id)

            # add predicates for special mentions of the object
            if object_type == 'Potato' and instance_state['simbotIsBoiled']:
                predicate_definition_str += "(isBoiledPotato {})\n".format(instance_id)
            if object_type == 'BreadSliced' and instance_state['isCooked']:
                predicate_definition_str += "(isToast {})\n".format(instance_id)
            if object_type == 'Mug' and instance_state['simbotIsFilledWithCoffee']:
                predicate_definition_str += "(isCoffee {})\n".format(instance_id)
            if object_type == 'PotatoSliced' and instance_state['isCooked']:
                predicate_definition_str += "(isCookedPotatoSlice {})\n".format(instance_id)

            contains = ""
            for oid in instance_state['receptacleObjectIds']:
                contained_i_state = symbolic_state_dict.get(oid, None)
                if contained_i_state is not None:
                    contained_type = contained_i_state['objectType']
                    contained_iscooked = contained_i_state['isCooked']
                    if contained_type  == 'PotatoSliced' and contained_iscooked:
                        contains += "Cooked%s_"%(contained_type)
                    elif contained_type  == 'BreadSliced' and contained_iscooked:
                        contains += "Toast_"
                    else:
                        contains += "%s_"%(contained_type)
            
            if object_type == 'Bowl':
                if 'AppleSliced' in contains:
                    predicate_definition_str += "(isBowlOfAppleSlices {})\n".format(instance_id)
                if 'TomatoSliced' in contains:
                    predicate_definition_str += "(isBowlOfTomatoSlices {})\n".format(instance_id)
                if 'Lettucesliced' in contains:
                    predicate_definition_str += "(isBowlOfLettuceSlices {})\n".format(instance_id)
                if 'CookedPotatoSliced' in contains:
                    predicate_definition_str += "(isBowlOfCookedPotatoSlices {})\n".format(instance_id)
            if object_type == 'Plate':
                if 'AppleSliced' in contains:
                    predicate_definition_str += "(isPlateOfAppleSlices {})\n".format(instance_id)
                if 'TomatoSliced' in contains:
                    predicate_definition_str += "(isPlateOfTomatoSlices {})\n".format(instance_id)
                if 'Lettucesliced' in contains:
                    predicate_definition_str += "(isPlateOfLettuceSlices {})\n".format(instance_id)
                if 'CookedPotatoSliced' in contains:
                    predicate_definition_str += "(isPlateOfCookedPotatoSlices {})\n".format(instance_id)
                
                if 'Toast' in contains:
                    if len(instance_state['receptacleObjectIds']) == 1:
                        predicate_definition_str += "(isPlateOfToast {})\n".format(instance_id)
                    elif any([o in contains for o in ['Lettucesliced', 'TomatoSliced']]):
                        predicate_definition_str += "(isSandwich {})\n".format(instance_id)
                elif len(instance_state['receptacleObjectIds']) > 1 and \
                    all(['Sliced' in o for o in instance_state['receptacleObjectIds']]):
                        predicate_definition_str += "(isSalad {})\n".format(instance_id)

            # add auxiliary predicates
            if object_type in SemanticClass.get_all_objs_in_semcls("OpenableReceptacles") and not instance_state['isOpen']:
                predicate_definition_str += "(isClosedOpenable {})\n".format(instance_id)
            if 'pickupable' in get_object_affordance(object_type) and instance_state['isPickedUp'] and not instance_state['parentReceptacles']:
                predicate_definition_str += "(holding {})\n".format(instance_id)
            for recep_id in instance_state['parentReceptacles']:
                recep = symbolic_state_dict.get(recep_id, None)
                if recep is not None and recep['objectType'] in WaterContainers and recep['simbotIsFilledWithWater']:
                    predicate_definition_str += "(isInWater {})\n".format(instance_id)
            
            if 'pickupable' in get_object_affordance(object_type):
                valid_receptacles = RECEPTACLE_COMPATIBILITY.get(object_type, set())
                for recep_id, recep_state in symbolic_state_dict.items():
                    if recep_state['objectType'] in valid_receptacles:
                        predicate_definition_str += "(canBePlacedTo {} {})\n".format(instance_id, recep_id)

            # add actual physical state predicates
            for predicate, value in instance_state.items():
                if predicate in [
                    'objectId', 'objectType', 'distance', 'receptacleObjectIds', 'isFilledWithLiquid', 'isPlacedOn', 'simbotIsPickedUp', 'centroid', 'sliceParent'
                ]:
                    continue
                
                if not value:
                    continue
                
                if predicate == 'parentReceptacles':
                    for oid in value:
                        predicate_definition_str += "(parentReceptacles {} {})\n".format(instance_id, oid)
                elif predicate == 'visible':
                    predicate_definition_str += "(isVisible {})\n".format(instance_id)
                elif predicate == 'interactable':
                    predicate_definition_str += "(isInteractable {})\n".format(instance_id)
                else:
                    predicate_definition_str += "({} {})\n".format(predicate, instance_id)

            dis = int(np.round(instance_state['distance']*10) + 1) if 'dummy' not in instance_id else cls.DUMMY_OBJECT_DISTANCE
            distance_definition_str += "(= (distance START_LOC {}) {})\n".format(instance_id, dis)
            distance_definition_str += "(= (distance {} START_LOC) {})\n".format(instance_id, dis)

        for instance_x, state_x in symbolic_state_dict.items():
            for instance_y, state_y in symbolic_state_dict.items():
                if instance_x == instance_y:
                    continue
                
                if 'dummy' in instance_x or 'dummy' in instance_y:
                    distance_definition_str += "(= (distance {} {}) {})\n".format(instance_x, instance_y, cls.DUMMY_OBJECT_DISTANCE)
                else:
                    distance = int(np.round(np.linalg.norm(np.array(state_x['centroid']) - np.array(state_y['centroid'])) * 10)) + 1
                    distance_definition_str += "(= (distance {} {}) {})\n".format(instance_x, instance_y, distance)

        objects_str = ("(:objects\n" + \
            "%s"%(textwrap.indent(objects_definition_str, " "*4)) + \
            ")\n"
        )
        
        init_state_str = (
            "(:init\n" + \
            textwrap.indent(predicate_definition_str, " "*4) + \
            textwrap.indent(distance_definition_str, " "*4) + \
            "    (atLocation bot START_LOC)\n"
            "    (= (total-cost) 0)\n"
            ")\n"
        )

        return objects_str, init_state_str
    
    
    @classmethod
    def subgoal_to_pddl_goal(cls, subgoal: TeachSubgoal) -> str:
        """
        Translate a TeachSubgoal to a PDDL goal string.
        """
        
        if subgoal.predicate not in ['parentReceptacles', 'isClear']:
            goal_obj_type = (
                subgoal.subject_constraint["objectType"]
                if "objectType" in subgoal.subject_constraint
                else subgoal.subject_constraint["semanticClass"]
            )

            goal_predicates = []
            if subgoal.predicate == 'isFilledWithLiquid':
                goal_predicates.append('(simbotIsFilledWithWater ?o1)')
            elif subgoal.predicate == 'isEmptied':
                goal_predicates.append('(not (simbotIsFilledWithWater ?o1))') # TODO: not isFilledWithCoffee either
            elif subgoal.predicate == 'isClean':
                goal_predicates.append('(not (isDirty ?o1))')
            elif subgoal.predicate == 'isPickedUp':
                goal_predicates.append('(holding ?o1)')
            else:
                goal_predicates.append('(%s ?o1)'%subgoal.predicate)
            
            for predicate, value in subgoal.subject_constraint.items():
                if predicate in {"objectType", "semanticClass"} or not value:
                    continue
                if predicate != 'childReceptacles':
                    goal_predicates.append('(%s ?o1)'%predicate)
                elif subgoal.predicated_subgoal_tuple is not None:
                    subj_name = subgoal.predicated_subgoal_tuple[0]
                    if subj_name in SPECIAL_GOAL_ARG_MENTIONS:
                        goal_predicates.append('(is%s ?o1)'%subj_name)
            
            for exclude_id in subgoal.exclude_instance_ids:
                goal_predicates.append('(not (= ?o1 {}))'.format(exclude_id))
            for exclude_id in subgoal.exclude_instance_ids_obj:
                goal_predicates.append('(not (= ?o2 {}))'.format(exclude_id))
            
            goal_str = (
                "(:goal\n"
                "   (and\n"
                "       (exists (?o1 - {})".format(goal_obj_type) + "\n"
                "           (and\n"
                "{}".format(textwrap.indent("\n".join(goal_predicates), " "*15)) + "\n"
                "           )\n"
                "       )\n"
                "    )\n"
                ")\n"
            )
        elif subgoal.predicate == 'isClear':
            instance_id = subgoal.subject_constraint["objectId"]
            goal_str = (
                "(:goal\n"
                "   (forall (?o1 - Pickupable)\n"
                "       (and\n"
                "           (not (parentReceptacles ?o1 {}))".format(instance_id) + "\n"
                "       )\n"
                "   )\n"
                ")\n"
            )
        
        else: #binary predicate: parentReceptacles (place)
            picked_type = (
                subgoal.subject_constraint["objectType"]
                if "objectType" in subgoal.subject_constraint
                else subgoal.subject_constraint["semanticClass"]
            )
            recep_type = (
                subgoal.object_constraint["objectType"]
                if "objectType" in subgoal.object_constraint
                else subgoal.object_constraint["semanticClass"]
            )

            goal_predicates = []
            for predicate, value in subgoal.subject_constraint.items():
                if predicate in {"objectType", "semanticClass"} or not value:
                    continue
                if predicate != 'childReceptacles':
                    goal_predicates.append('(%s ?o1)'%predicate)
                elif subgoal.predicated_subgoal_tuple is not None:
                    subj_name = subgoal.predicated_subgoal_tuple[0]
                    if subj_name in SPECIAL_GOAL_ARG_MENTIONS:
                        goal_predicates.append('(is%s ?o1)'%subj_name)
            
            for exclude_id in subgoal.exclude_instance_ids:
                goal_predicates.append('(not (= ?o1 {}))'.format(exclude_id))
            for exclude_id in subgoal.exclude_instance_ids_obj:
                goal_predicates.append('(not (= ?o2 {}))'.format(exclude_id))
            
            goal_str = (
                "(:goal\n"
                "   (and\n"
                "       (exists (?o1 - {} ?o2 - {})".format(picked_type, recep_type) + "\n"
                "           (and\n"
                "               (parentReceptacles ?o1 ?o2)\n"
                "{}".format(textwrap.indent("\n".join(goal_predicates), " "*15)) + "\n"
                "           )\n"
                "       )\n"
                "    )\n"
                ")\n"
            )
        
        return goal_str

    @staticmethod
    def scene_adjustment(
        subgoal: TeachSubgoal, symbolic_state: TeachSymbolicWorld, nlu_output: dict = {}, prune=True
    ):
        """
        Adjust scene as follows:
        1. Remove objects that are already satisfied the subgoal
        2. Create and add dummy objects believed to be in the scene.
            E.g. if the subgoal is "cook a breadsliced" and the Toaster is not observed yet, 
            then create a dummy object "Toaster" and add it to the scene in order to get a plan.
        3. Update the location of dummy objects according to the NLU parsing results (if any).
            When dummy objects are created we make any assumption about their locations, and 
            the agent has to explore and search for it. However, when the location information
            can be obtained from dialog and our NLU model has recognized them, we can use it as
            the "believed" location of these dummy objects. 
        4. Prune the scene to remove objects that are not relevant to the subgoal.
        """

        goal_instance_to_remove = set()
        dummy_object_to_add = set()


        all_instances = symbolic_state.get_all_instances()
        symbolic_state_dict = symbolic_state.get_symbolic_state_dict()

        if prune:
            symbolic_state_dict = TeachPDDLProblem.prune_symbolic_state(symbolic_state_dict, subgoal)

        # Remove objects that are already satisfied the subgoal
        for instance in all_instances:
            if TeachSubgoal.check_goal_instance(subgoal, instance, all_instances):
                if subgoal.predicate != 'parentReceptacles':
                    goal_instance_to_remove.add(instance.instance_id)
                else:
                    if not TeachSubgoal.check_instance_condition(instance, subgoal.object_constraint):
                        continue
                    for recep_child_iid in instance.state["receptacleObjectIds"]():
                        recep_child = symbolic_state.get_instance(recep_child_iid)
                        print(recep_child_iid, recep_child)
                        if recep_child is not None and TeachSubgoal.check_instance_condition(recep_child, subgoal.subject_constraint):
                            goal_instance_to_remove.add(recep_child_iid)
        
        # remove the object instances
        for iid in goal_instance_to_remove:
            if iid in symbolic_state_dict:
                symbolic_state_dict.pop(iid)
        
        # remove all relations associated with these instances
        for iid, istate in symbolic_state_dict.items():
            for rel_iid in istate['receptacleObjectIds'].copy():
                if rel_iid in goal_instance_to_remove:
                    istate['receptacleObjectIds'].remove(rel_iid)
            for rel_iid in istate['parentReceptacles'].copy():
                if rel_iid in goal_instance_to_remove:
                    istate['parentReceptacles'].remove(rel_iid)
        

        # Create and add dummy objects believed to be in the scene.
        exist_object_types = set([o["objectType"] for o in symbolic_state_dict.values()])

        goal_object_types = subgoal.get_goal_object_types()
        if not any([o in exist_object_types for o in goal_object_types]):
            if len(goal_object_types) == 1:
                otype = goal_object_types.pop()
                if 'Sliced' not in otype: 
                    dummy_object_to_add.add(otype)
            else:
                dummy_object_to_add.add(subgoal.subject_constraint['semanticClass'])

        
        if subgoal.predicate == 'isCooked':
            for obj in ManipulableKitchenUtils | FoodCookers:
                if obj not in exist_object_types:
                    dummy_object_to_add.add(obj)

        elif subgoal.predicate == 'simbotIsBoiled':
            for obj in ManipulableKitchenUtils | FoodCookers | {"Faucet", "Sink"}:
                if obj not in exist_object_types:
                    dummy_object_to_add.add(obj)
        
        elif subgoal.predicate in ['isClean', 'isFilledWithLiquid', 'isEmptied']:
            for obj in {"Faucet", "Sink", "SinkBasin"}: # WaterTaps | WaterBottomReceptacles:
                if obj not in exist_object_types:
                    dummy_object_to_add.add(obj)
        
        elif subgoal.predicate == 'isSliced':
            for obj in Knives:
                if obj not in exist_object_types:
                    dummy_object_to_add.add(obj)
        
        elif subgoal.predicate == 'simbotIsFilledWithCoffee':
            if "CoffeeMachine" not in exist_object_types:
                dummy_object_to_add.add("CoffeeMachine")
            if "Mug" not in exist_object_types:
                dummy_object_to_add.add("Mug")
        
        elif subgoal.predicate == 'parentReceptacles':
            goal_object_types = subgoal.get_goal_object_types(which='object')
            if not any([o in exist_object_types for o in goal_object_types]):
                if len(goal_object_types) == 1:
                    dummy_object_to_add.add(goal_object_types.pop())
                else:
                    dummy_object_to_add.add(subgoal.object_constraint['semanticClass'])
        
        # print("goal_instance_to_remove", goal_instance_to_remove)
        # print("dummy_object_to_add", dummy_object_to_add)

        if not nlu_output:
            # TODO: implemnet spatial relationship
            pass
        

        # add dummy object instances
        for obj_cls in dummy_object_to_add:
            iid = "%s_dummy"%obj_cls
            dummy_object = {
                'objectId': iid,
                'objectType': obj_cls,
            }
            dummy_state = create_default_object_state()
            dummy_state['visible'].set_value(False)
            dummy_state['isObserved'].set_value(False)
            for state_name in dummy_state:
                dummy_object[state_name] = dummy_state[state_name]()
            
            symbolic_state_dict[iid] = dummy_object
        
        return symbolic_state_dict

    @staticmethod
    def prune_symbolic_state(symbolic_state_dict: dict, subgoal: TeachSubgoal):
        pruned_state = dict()

        # first get all the relevant object types
        relevant_types = subgoal.get_goal_object_types()
        if subgoal.predicate == 'parentReceptacles':
            relevant_types |= subgoal.get_goal_object_types(which='object')
        elif subgoal.predicate == 'isCooked':
            if 'BreadSliced' in relevant_types:
                relevant_types.add("Toaster")
            relevant_types |= {"Microwave", "StoveBurner", "Pan", "Pot"}
        elif subgoal.predicate == 'simbotIsBoiled':
            relevant_types |= ({"Bowl", "Pot", "Microwave", "StoveBurner"} | WaterTaps | WaterBottomReceptacles)
        elif subgoal.predicate in ['isClean', 'isEmptied']:
            relevant_types |=  (WaterTaps | WaterBottomReceptacles)
        elif subgoal.predicate == 'isFilledWithLiquid':
            relevant_types |=  (WaterTaps | WaterBottomReceptacles)
            if "HousePlant" in relevant_types:
                relevant_types |= WaterContainers
        elif subgoal.predicate == 'isSliced':
            relevant_types |=  Knives
        elif subgoal.predicate == 'simbotIsFilledWithCoffee':
            relevant_types.add("CoffeeMachine")
            relevant_types.add("Mug")
        elif subgoal.predicate == 'isClear':
            relevant_types.add("CounterTop")

        # then get all the relevant instances
        rel_obj = set()
        rel_obj_receptacles = set()
        rel_obj_siblings = set()
        is_holding = None
        for iid, istate in symbolic_state_dict.items():
            if istate['objectType'] in relevant_types:
                rel_obj.add(iid)
                for recep_id in istate['parentReceptacles']:
                    rel_obj_receptacles.add(recep_id)
                    recep_state = symbolic_state_dict.get(recep_id, None)
                    if recep_state is not None:
                        for sibling_id in recep_state['receptacleObjectIds']:
                            rel_obj_siblings.add(sibling_id)
                rel_obj_siblings |= set(istate['receptacleObjectIds'])
            elif istate['isPickedUp']:
                rel_obj.add(iid)
                if not istate['parentReceptacles']:
                    is_holding = istate['objectType']

        objects_to_keep = rel_obj | rel_obj_receptacles | rel_obj_siblings 

        for obj_id, obj_state in symbolic_state_dict.items():
            if obj_id in subgoal.exclude_instance_ids or subgoal.exclude_instance_ids_obj:
                continue
            if obj_id in objects_to_keep:
                pruned_state[obj_id] = obj_state
        
        # if there is something in hand, inculde at least one valid receptacle for it to free hand
        if is_holding is not None:
            compatible_receptacles = RECEPTACLE_COMPATIBILITY.get(is_holding, set())
            if not any([o['objectType'] in compatible_receptacles for o in pruned_state.values()]):
                for obj_id, obj_state in symbolic_state_dict.items():
                    otype = obj_state['objectType']
                    if otype in BigReceptacles and obj_state['interactable']:
                        pruned_state[obj_id] = obj_state
                        break

        return pruned_state


    
