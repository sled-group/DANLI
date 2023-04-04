from typing import Dict, List, Tuple, Union, Optional
import os
import copy
from pprint import pprint

from sledmap.mapper.models.teach.skills.navigation import NavigationSkill
from sledmap.mapper.env.teach.teach_action import TeachAction
from definitions.teach_objects import ObjectClass
from definitions.teach_object_semantic_class import SemanticClass

from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
from sledmap.mapper.models.teach.neural_symbolic_state_repr import NeuralSymbolicAgentState

from sledmap.plan.domains.teach.pddl_problem import TeachPDDLProblem
from sledmap.plan.pddl_planner import PDDLPlanner


LOWER_CASE_TO_PASCAL_CLASS_NAME = {i.name.lower(): i.name for i in ObjectClass}
LOWER_CASE_TO_PASCAL_CLASS_NAME.update({i.name.lower(): i.name for i in SemanticClass})

def create_empty_record() -> dict:
    return {
        "subgoal": "",
        'num_steps': 0,
        'num_goto_steps': 0,
        'num_search_steps': 0,
        'num_plans': 0,
        'num_failed_actions': 0,
        'completed': False,
        'goal_instance_id': None,
        'fail_reason': "",
        'plans': [],
        'exceptions': [],
        'actions': [],
    }


class PlanAndExecuteSkill(NavigationSkill):
    MAXIMUM_NUM_STEP_TOTAL = 200
    # MAXIMUM_NUM_STEP_GOTO = 32
    # MAXIMUM_NUM_STEP_SEARCH = 64

    def __init__(self, args, logger=None):
        super().__init__(logger, args.disable_search_belief)

        self.args = args
        self.fastdownward_path=self.args.fastdownward_path
        self.pddl_problem_save_dir=self.args.pddl_problem_save_dir
        self.pddl_domain_file= self.args.pddl_domain_file
        self.pddl_plan_file=os.path.join(self.pddl_problem_save_dir, "plan_%d"%self.args.agent_id)
        self.disable_replan = self.args.disable_replan
        self.enable_pruning = not self.args.disable_scene_pruning

        self.planner = PDDLPlanner(fd_path=self.fastdownward_path, plan_file=self.pddl_plan_file)

        # below should appear in reset
        self.log_func("Planning and execution skill is initialized!")
        self.planned_actions = None
        self.last_action = {'action': None, 'arg': None}
        self.already_replanned_in_current_state = False
        self.try_to_move_closer = False
        self.record = create_empty_record()

    def reset(self):
        super().reset()

        self.log_func("Planning and execution skill is reset!")
        self.planned_actions = None
        self.last_action = {'action': None, 'arg': None}
        self.already_replanned_in_current_state = False
        self.try_to_move_closer = False
        self.record = create_empty_record()
    
    def step(
        self,
        subgoal: TeachSubgoal,
        state_repr: NeuralSymbolicAgentState,
    ) -> TeachAction: 
        """ Return an action to address the subgoal given the current state. """
        self.current_status = {'events': [], 'plan': [], 'complete_msg': None, 'fail_msg': None, }
        action = self._step(subgoal, state_repr)
        if action.is_stop():
            self.log_func("Plan is completed!")
            self.record['completed'] = True
            self.record['num_steps'] = self.num_steps
            self.record['subgoal'] = subgoal.to_string()
            self.current_status['complete_msg'] = self.record['subgoal'] + ' is completed!'
            all_instances = state_repr.get_all_instances()
            for instance in state_repr.get_instances_with_state_change():
                if TeachSubgoal.check_goal_instance(subgoal, instance, all_instances):
                    self.record['goal_instance_id'] = instance.instance_id
                    subgoal.assign_goal_instance_id(instance.instance_id)
                    break
            if self.record['goal_instance_id'] is None:
                self.log_func("Subgoal is completed without knowing the goal instance!")

        elif action.is_failed():
            self.log_func("Failed: %s"%self.record['fail_reason'])
            self.record['num_steps'] = self.num_steps
            self.record['subgoal'] = subgoal.to_string()
            self.current_status['fail_msg'] =  self.record['subgoal'] + ' is failed: ' + self.record['fail_reason']
        
        action_record = action.to_dict()
        action_record['step'] = self.num_steps
        self.record['actions'].append(action_record)
        self.current_status['plan'] = self.get_curr_plan()
        self.num_steps += 1
        return action

    def _step(
        self,
        subgoal: TeachSubgoal,
        state_repr: NeuralSymbolicAgentState,
    ) -> TeachAction:

        # print("=============================> Current World State: ")
        # for instance in sorted(state_repr.symbolic_world_repr.get_all_instances(), key=lambda x: x.instance_id):
        #     print("%r"%instance)
        #     if instance.state.receptacleObjectIds() or instance.state.parentReceptacles():
        #         print(' - child:', instance.state.receptacleObjectIds(), 'parent:', instance.state.parentReceptacles())
        # pprint(list(state_repr.symbolic_world_repr.get_all_instances()))

        # terminate if reach maximum number of steps
        if self.num_steps >= self.MAXIMUM_NUM_STEP_TOTAL:
            self.record['fail_reason'] = "Reach step limit: %d" % self.num_steps
            return TeachAction.fail_action()
        
        if self.planned_actions is None:
            self.log_func("Initial planning")
            self.current_status['events'].append("Get the initial plan for subgoal: %s"%subgoal.to_string())
            self.planned_actions = self.plan(subgoal, state_repr)
            if self.planned_actions is None:
                self.record['fail_reason'] = "Initial plan failed"
                return TeachAction.fail_action()
            self.already_replanned_in_current_state = False
            return self.get_next_action(subgoal, state_repr)

        # exception handling 
        if state_repr.last_action_failed:
            return self.exception_handling(subgoal, state_repr) 
        
        # normal forward
        self.already_replanned_in_current_state = False
        self.try_to_move_closer = False
        return self.get_next_action(subgoal, state_repr)

        
    def get_next_action(self, subgoal: TeachSubgoal, state_repr: NeuralSymbolicAgentState) -> TeachAction:

        if self.last_action['action']  == 'Goto':
            target_instance = state_repr.get_instance_by_id(self.last_action['arg'])
            
            if target_instance is None:
                if self.already_replanned_in_current_state:
                    self.record['fail_reason'] = "Navigation target does not exist: %s"%self.last_action['arg']
                    # this should not happen actually
                    return TeachAction.fail_action()
                else:
                    self.log_func("Wrong Goto target instance %r"%self.last_action['arg'])
                    self.record['exceptions'].append({'type': 'invalid_navi_target', 'solve': 'replan', 'step': self.num_steps})
                    self.current_status['events'].append("Exception: non-existent GoTo Target: %s"%self.last_action['arg'])
                    return self.replan_and_return_the_next_action(subgoal, state_repr)
            
            action = self.navigate_to(target_instance, state_repr)
            if not action.is_stop():
                # still navigating
                self.record['num_goto_steps'] += 1
                return action
            
        if self.last_action['action']  == 'Search':
            search_type = self.last_action['arg'].split('_dummy')[0]
            if ObjectClass.has_object(search_type):
                search_types = {search_type}
            elif SemanticClass.has_semcls(search_type):
                search_types = SemanticClass.get_all_objs_in_semcls(search_type)
            else:
                raise ValueError("Unknown search type: %r"%search_type)
            action = self.search_target(search_types, state_repr)
            if not action.is_stop():
                # still searching
                if self.search_skill.state == 'FINAL_GO_FOR':
                    self.record['num_goto_steps'] += 1
                else:
                    self.record['num_search_steps'] += 1
                return action
            
        if self.last_action['action'] == "ToggleOn" and "Stove" in self.last_action['arg']:
            # To turn on a stove burner, we have to interact with a stove knob
            burner = state_repr.get_instance_by_id(self.last_action['arg'])
            if burner is None:
                if self.already_replanned_in_current_state:
                    self.record['fail_reason'] = "Invalid interaction target: %s"%self.last_action['arg']
                    return TeachAction.fail_action()
                else:
                    self.log_func("Instance %r not exists. Replan!")
                    self.record['exceptions'].append({'type': 'invalid_interact_target', 'solve': 'replan', 'step': self.num_steps})
                    self.current_status['events'].append("Exception: non-existent Toggle target: %s"%self.last_action['arg'])
                    return self.replan_and_return_the_next_action(subgoal, state_repr)
            
            if not burner.state.isToggled():
                # randomly choose an untoggled stove knob to toggle
                self.log_func("Randomly choose an untoggled stove knob to toggle")
                for i in state_repr.get_all_instances():
                    if i.object_type == 'StoveKnob' and i.state.interactable() and not i.state.isToggled():
                        detection = state_repr.get_2D_detection_of_instance(i)
                        if detection is not None:
                            self.log_func("Select %r to toggle"%i)
                            return TeachAction.create_action_with_instance(
                                        action_type="ToggleOn",
                                        instance_id_3d=i.instance_id,
                                        detection=detection,
                                    )
                # if cannot find such a stove knob: give up; 
                self.log_func("Cannot find such a stove knob")
                self.record['fail_reason'] = "Invalid interaction target: %s"%self.last_action['arg']
                # TODO: move a little bit to manipulate the knob?
                return TeachAction.fail_action()
            else:
                # the burner is observed to be toggled: move on to the next planned action
                pass
            
        # if we reach here, we are ready to move on to the next planned action
        assert self.planned_actions is not None, "Have to plan first to get an action!"
        assert len(self.planned_actions)>0, "Plan is empty: something went wrong!"

        candidate_action = self.planned_actions.pop(0)
        self.log_func('Next candidate_action: %r'%candidate_action)
        if candidate_action['action'] in ['Goto', 'Search']:
            self.last_action = candidate_action
            return self.get_next_action(subgoal, state_repr)
        elif candidate_action['action'] == "ToggleOn" and "Stove" in candidate_action['arg']:
            self.last_action = candidate_action
            return self.get_next_action(subgoal, state_repr)
        elif candidate_action['action'] == 'Stop':
            return TeachAction.stop_action()
        
        detection = state_repr.get_2D_detection_of_instance(candidate_action['arg'])
        if detection is not None:
            self.last_action = candidate_action
            return TeachAction.create_action_with_instance(
                                    action_type=candidate_action['action'],
                                    instance_id_3d=candidate_action['arg'],
                                    detection=detection,
                                )
        elif not self.already_replanned_in_current_state:
            self.log_func("Cannot find detection of instance %r"%candidate_action['arg'])
            self.record['exceptions'].append({'type': 'interact_target_not_detected', 'solve': 'replan', 'step': self.num_steps})
            self.current_status['events'].append("Exception: no grounding for: %s"%self.last_action['arg'])
            return self.replan_and_return_the_next_action(subgoal, state_repr)
        else:
            self.record['fail_reason'] = "Invalid interaction target: %s"%candidate_action['arg']
            return TeachAction.fail_action()

    

    def plan(self, subgoal: TeachSubgoal, state_repr: NeuralSymbolicAgentState, add_stop=True) -> list:
        pddl_state = TeachPDDLProblem.scene_adjustment(
            subgoal=subgoal, symbolic_state=state_repr.symbolic_world_repr, prune=self.enable_pruning
        )
        pddl_problem = TeachPDDLProblem(subgoal=subgoal, symbolic_state_dict=pddl_state)
        # print(pddl_problem.pddl_problem_str)
        pddl_problem_file = pddl_problem.save_problem(save_path=self.pddl_problem_save_dir)
        self.log_func("### Problem file saved: %s"%pddl_problem_file)
        
        raw_plan, runtime = self.planner.plan(self.pddl_domain_file, pddl_problem_file)
        self.log_func("### Planning runtime: %.3f"%runtime)
        self.record['num_plans'] += 1
        plan_record = {
            'step': self.num_steps, 
            'pddl_file': pddl_problem_file, 
            'runtime': runtime, 
            'success': raw_plan is not None, 
            'parsed_plan': []
        }

        # self.log_func("=============================> Current World State: ")
        # for instance in sorted(state_repr.symbolic_world_repr.get_all_instances(), key=lambda x: x.instance_id):
        #     self.log_func("%r"%instance)
        #     if instance.state.receptacleObjectIds() or instance.state.parentReceptacles():
        #         self.log_func(' - child: %r'%instance.state.receptacleObjectIds() + ' parent: %r'%instance.state.parentReceptacles())


        if raw_plan is None:
            self.log_func("Failed: no plan found")
            self.record['plans'].append(plan_record)
            return None

        parsed_plan = []
        for idx, action in enumerate(raw_plan):
            if action[0] == "goto" and idx != len(raw_plan) - 1 and raw_plan[idx + 1][0] == "goto":
                # merge multiple goto actions
                continue
        
            action_name = action[0]
            if action_name == "goto":
                arg = action[3]
            elif action_name in ["place", "pourtofillable", "pourtosink"]:
                arg = action[2]
            else:
                arg = action[1]
            
            # pddl planner outputs are lower case, have to transform to pascal case
            obj_cls_lower, obj_id = arg.split('_')
            obj_cls = LOWER_CASE_TO_PASCAL_CLASS_NAME[obj_cls_lower]
            arg = obj_cls + '_' + obj_id

            if 'pour' in action_name:
                action_name = "Pour"
            elif action_name == 'toggleon':
                action_name = "ToggleOn"
            elif action_name == 'toggleoff':
                action_name = "ToggleOff"
            else:
                action_name = action_name.capitalize()
            parsed_plan.append({'action': action_name, 'arg': arg})
        
        if add_stop:
            parsed_plan.append({'action': 'Stop', 'arg': None})

        
        self.log_func("### Find a plan: \n{}".format("\n".join([" - %r %r"%(a['action'], a['arg']) for a in parsed_plan])))
        plan_record['parsed_plan'] = copy.deepcopy(parsed_plan)
        self.record['plans'].append(plan_record)

        # after planning we initialize the last planned action as None
        self.last_action = {'action': None, 'arg': None}
        return parsed_plan

    def replan_and_return_the_next_action(self, subgoal: TeachSubgoal, state_repr: NeuralSymbolicAgentState) -> TeachAction:
        if self.disable_replan:
            self.record['fail_reason'] = "Try to replan but replan is disabled"
            return TeachAction.fail_action()
        self.log_func("Try to replan!")
        self.current_status['events'].append("Replanned")
        self.planned_actions = self.plan(subgoal, state_repr)
        if self.planned_actions is None:
            self.record['fail_reason'] = "Replan failed"
            return TeachAction.fail_action()
        self.already_replanned_in_current_state = True
        return self.get_next_action(subgoal, state_repr)

    def exception_handling(self, subgoal: TeachSubgoal, state_repr: NeuralSymbolicAgentState) -> TeachAction:
        self.record['num_failed_actions'] += 1  
        if self.record['num_failed_actions'] >= 5:
            self.record['fail_reason'] = "Reach failed actions limit: %d"%self.record['num_failed_actions']
            return TeachAction.fail_action()

        last_action_type = state_repr.last_action.action_type
        interact_instance_id = state_repr.last_action.instance_id
        interact_instance = state_repr.get_instance_by_id(interact_instance_id)

        # if the receptacle is occupied, try to clear it before placing
        if last_action_type == 'Place' and interact_instance.state.receptacleObjectIds():
            planner_action_place = self.last_action.copy()
            self.log_func("Try to clear the receptacle: %r"%interact_instance)
            self.current_status['events'].append("Exception: cannot place to: %s"%planner_action_place['arg'])
            oid, otype = interact_instance_id, interact_instance.object_type
            clear_sg = TeachSubgoal(
                predicate='isClear', subject_constraint={'objectId': oid, 'objectType': otype}
            )
            plan_for_clear = self.plan(clear_sg, state_repr, add_stop=False)
            if plan_for_clear is None or len(plan_for_clear) == 0:
                self.log_func("Did not find any plan or no need to clear the receptacle")
                # then we can not do anything here, move on for other exception handling strategies below
                self.last_action = planner_action_place # restore the last action
            else:
                self.log_func("Plan for receptacle clear is generated")
                self.planned_actions = plan_for_clear + [planner_action_place] +  self.planned_actions
                self.record['exceptions'].append({'type': 'receptacle_occupied', 'solve': 'clear', 'step': self.num_steps})
                self.current_status['events'].append("Plan for clearing: %s"%oid)
                return self.get_next_action(subgoal, state_repr)
        
        if state_repr.last_action.is_interaction() or self.try_to_move_closer:
            self.log_func("The last interaction is failed.")
            if not self.try_to_move_closer and interact_instance.state.visible():
                self.log_func("Try to move closer")
                self.try_to_move_closer = True
                self.planned_actions = [self.last_action] +  self.planned_actions # add the popped action back to the plan
                self.record['exceptions'].append({'type': 'not_close_enough', 'solve': 'move_forward', 'step': self.num_steps})
                self.current_status['events'].append("Try to move closer")
                return TeachAction('Forward')
            if self.already_replanned_in_current_state:
                self.log_func("Exclude [%r] from planning!"%interact_instance)
                if subgoal.predicate != "parentReceptacles":
                    subgoal.exclude_instance_ids.append(interact_instance_id)
                else:
                    subgoal.exclude_instance_ids_obj.append(interact_instance_id)
                self.record['exceptions'].append({'type': 'interaction_failed', 'solve': 'exclude_and_replan', 'step': self.num_steps})
                self.current_status['events'].append("Exclude %s and retry"%interact_instance)
                return self.replan_and_return_the_next_action(subgoal, state_repr)
            else:
                self.record['exceptions'].append({'type': 'interaction_failed', 'solve': 'replan', 'step': self.num_steps})
                return self.replan_and_return_the_next_action(subgoal, state_repr)
            
        else:
            self.log_func("The last navigation action is failed. Try to reset target.")
            self.target_instance = None
            self.record['exceptions'].append({'type': 'navi_failed', 'solve': 'reset_navi_target', 'step': self.num_steps})
            return self.get_next_action(subgoal, state_repr)
        
    def get_record(self) -> dict:
        return copy.deepcopy(self.record)
    
    def get_curr_plan(self) -> list:
        planned_actions = copy.deepcopy(self.planned_actions) if self.planned_actions is not None else []
        return [copy.deepcopy(self.last_action)] + planned_actions
        