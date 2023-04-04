import copy
from typing import Tuple, Union, List, Optional, Dict

from sledmap.mapper.env.teach.teach_object_instance import ObjectInstance
from sledmap.mapper.env.teach.teach_action import TeachAction
from definitions.teach_object_semantic_class import SemanticClass, check_obj_X_is_Y
from definitions.teach_objects import ObjectClass


class TeachSubgoal:
    def __init__(
        self,
        predicate: str,
        subject_constraint: Dict,
        object_constraint: Optional[Dict] = None,
        predicated_subgoal_tuple: Optional[Tuple] = None,
        exclude_instance_ids: Optional[List] = [],
        exclude_instance_ids_obj: Optional[List] = [],
    ) -> None:
        self.predicate = predicate  # predicate
        self.subject_constraint = copy.deepcopy(subject_constraint)  # dict
        self.object_constraint = copy.deepcopy(object_constraint)

        self.goal_instance_id = None

        self.predicated_subgoal_tuple = predicated_subgoal_tuple

        # instance ids that cannot be used for the goal
        self.exclude_instance_ids = copy.deepcopy(exclude_instance_ids)
        # NOTE: for subgoal PlaceTo, this is used for subject

        # instance ids that cannot be used for the object instance of PlaceTo
        self.exclude_instance_ids_obj = copy.deepcopy(exclude_instance_ids_obj)

    def get_goal_object_types(self, which="subject"):
        constraint = (
            self.subject_constraint if which == "subject" else self.object_constraint
        )
        if "objectType" in constraint:
            return set([constraint["objectType"]])
        else:
            semcls = constraint["semanticClass"]
            return SemanticClass.get_all_objs_in_semcls(semcls)

    def to_string(self):
        if self.predicated_subgoal_tuple is not None:
            predicate = (
                self.predicated_subgoal_tuple[:2]
                if self.predicate != "parentReceptacles"
                else self.predicated_subgoal_tuple
            )
            return "_".join(predicate)
        name = self.predicate
        for cons, val in self.subject_constraint.items():
            if cons in {"objectType", "semanticClass"}:
                name += "_" + val
            elif cons == "childReceptacles":
                continue
            else:
                name += "_" + (cons if val else "not%s" % cons)
        if self.object_constraint:
            name += "_" + (
                self.object_constraint["objectType"]
                if "objectType" in self.object_constraint
                else self.object_constraint["semanticClass"]
            )
        return name

    def assign_goal_instance_id(self, goal_instance_id):
        self.goal_instance_id = goal_instance_id

    @classmethod
    def create_from_predicted_tuple(
        cls, subgoal_tuple, exclude_instance_ids=[], exclude_instance_ids_obj=[]
    ):

        predicate = subgoal_tuple[1]
        subj_type = subgoal_tuple[0]
        subj_constraint = {}
        
        # childReceptacles: list of list of object class names. In each list of object names,
        # there must be at least one object name in the receptacledObjects to meet the constraint.

        if ObjectClass.has_object(subj_type):
            subj_constraint = {"objectType": subj_type}
        elif SemanticClass.has_semcls(subj_type):
            subj_constraint = {"semanticClass": subj_type}
        elif "Coffee" == subj_type:
            subj_constraint = {
                "objectType": "Mug",
                "simbotIsFilledWithCoffee": True,
                "isDirty": False,
            }
        elif "Salad" == subj_type:
            # Note: the constraint is loose since we do not know the exact salad coponent
            subj_constraint = {
                "objectType": "Plate",
                "isDirty": False,
                "childReceptacles": [["PotatoSlicd", "LettuceSliced", "TomatoSliced"]],
            }
        elif "Sandwich" == subj_type:
            # Note: the constraint is loose since we do not know the exact salad coponent
            subj_constraint = {
                "objectType": "Plate",
                "isDirty": False,
                "childReceptacles": [
                    ["LettuceSliced", "TomatoSliced"],
                    ["BreadSliced"],
                ],
            }
        elif "Toast" == subj_type:
            subj_constraint = {"objectType": "BreadSliced", "isCooked": True}
        elif "BoiledPotato" == subj_type:
            subj_constraint = {"objectType": "Potato", "simbotIsBoiled": True}
        elif "CookedPotatoSlice" == subj_type:
            subj_constraint = {"objectType": "PotatoSliced", "isCooked": True}
        else:
            if "BowlOf" in subj_type:
                child_type = subj_type.replace("BowlOf", "")
                subj_constraint = {"objectType": "Bowl"}
            else:
                assert "PlateOf" in subj_type
                child_type = subj_type.replace("PlateOf", "")
                subj_constraint = {"objectType": "Plate"}

            child_type = child_type.replace("Slices", "Sliced")
            if "Cooked" in child_type:
                child_type = child_type.replace("Cooked", "")
                # TODO: recurisively define the constraint for the child
            if "Toast" == child_type:
                child_type = "BreadSliced"
            assert ObjectClass.has_object(child_type), child_type
            subj_constraint["childReceptacles"] = [[child_type]]

        assert "objectType" in subj_constraint or "semanticClass" in subj_constraint

        if not subj_constraint:
            raise ValueError(f"Unknown subgoal subject type: {subj_type}")

        obj_constraint = None
        if predicate == "parentReceptacles":
            obj_type = subgoal_tuple[2]
            obj_constraint = {}
            if ObjectClass.has_object(obj_type):
                obj_constraint = {"objectType": obj_type}
            elif SemanticClass.has_semcls(obj_type):
                obj_constraint = {"semanticClass": obj_type}
            else:
                raise ValueError(f"Unknown subgoal object type: {obj_type}")

        return cls(
            predicate=predicate,
            subject_constraint=subj_constraint,
            object_constraint=obj_constraint,
            predicated_subgoal_tuple=subgoal_tuple,
            exclude_instance_ids=exclude_instance_ids,
            exclude_instance_ids_obj=exclude_instance_ids_obj,
        )

    @classmethod
    def check_goal_instance(
        cls,
        subgoal: "TeachSubgoal",
        target_instance: ObjectInstance,
        all_instances: List[ObjectInstance],
    ) -> bool:
        """
        Check if the target instance satisfies the subgoal constraints.
        """
        all_instance_ids = {
            instance.instance_id: instance for instance in all_instances
        }

        target_instance_state = target_instance.state
        if target_instance_state is None:
            return False

        if subgoal.predicate != "parentReceptacles":
            # Unary
            is_valid_subject = cls.check_instance_condition(
                target_instance, subgoal.subject_constraint
            )
            if not is_valid_subject:
                return False

            if subgoal.predicate == 'isClean':
                return not target_instance.state["isDirty"]()
            elif subgoal.predicate == 'isEmptied':
                return not target_instance.state["isFilledWithLiquid"]()
            elif subgoal.predicate == 'isClear':
                return not target_instance.state["receptacleObjectIds"]()
            else:
                return target_instance.state[subgoal.predicate]()
            
        else:
            # Relation
            if "receptacleObjectIds" not in target_instance_state:
                return False
            is_valid_recep = cls.check_instance_condition(
                target_instance, subgoal.object_constraint
            )
            if not is_valid_recep:
                return False
            for recep_child_instance_id in target_instance_state[
                "receptacleObjectIds"
            ].get_values():
                if recep_child_instance_id not in all_instance_ids:
                    continue
                recep_child_instance = all_instance_ids[recep_child_instance_id]
                is_valid_child = cls.check_instance_condition(
                    recep_child_instance, subgoal.subject_constraint
                )
                if is_valid_child:
                    # recep_child_instance is truely a valid instance placed on the target instance
                    subgoal.goal_instance_id = target_instance.instance_id
                    return True
            return False

    @classmethod
    def check_instance_condition(
        cls, instance: ObjectInstance, constraint: Dict
    ) -> bool:  # TODO: add Unary, Relation class

        instance_type = instance.object_type
        instance_state = instance.state

        # first check if the type matches
        goal_type = (
            constraint["objectType"]
            if "objectType" in constraint
            else constraint["semanticClass"]
        )
        if not check_obj_X_is_Y(instance_type, goal_type):
            return False

        # then check if all the states match the constraint
        for prop, condition in constraint.items():
            if prop in {"objectType", "semanticClass"}:
                continue
            if prop == "childReceptacles":
                if "receptacleObjectIds" not in instance_state:
                    # failed if the instance state is unknow for the constraint
                    return False
                instance_recep_child_ids = instance_state[
                    "receptacleObjectIds"
                ].get_values()
                instance_recep_child_types = set(
                    [i.split("_")[0] for i in instance_recep_child_ids]
                )
                for child_types in condition:
                    # child_types: list of object types. If any one is in the instance's
                    # instance_recep_child_types, then the instance satisfies the constraint.
                    if not set(child_types) & instance_recep_child_types:
                        return False

            else:
                if prop not in instance_state:
                    # failed if the instance state is unknow for the constraint
                    return False
                if instance_state[prop].get_value() != condition:
                    return False

        # all constraints are satisfied!
        return True

    def __repr__(self):
        _repr = ""
        if "objectType" in self.subject_constraint:
            _repr += self.subject_constraint["objectType"]
        else:
            _repr += self.subject_constraint["semanticClass"]
        for prop in self.subject_constraint:
            if prop in {"objectType", "semanticClass"}:
                continue
            if prop == "childReceptacles":
                _repr += f"-holds({self.subject_constraint[prop]})"
                continue
            prop = prop.replace("isDirty", "isClean")
            _repr += "-%s" % prop

        _repr += " " + self.predicate

        if self.object_constraint:
            if "objectType" in self.object_constraint:
                _repr += " " + self.object_constraint["objectType"]
            else:
                _repr += " " + self.object_constraint["semanticClass"]

        if self.goal_instance_id:
            _repr += " Satisfied: %s" % self.goal_instance_id
        return str(_repr)
