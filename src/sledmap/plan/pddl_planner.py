import os
import sys
import io
import subprocess
import time
from pprint import pprint

class PDDLPlanner:
    def __init__(self, fd_path, plan_file='sas_plan', alias='max-astar', timeout=60):
        self.timeout = timeout
        self.search_options = {
            # Optimal
            'dijkstra': '--heuristic "h=blind(transform=adapt_costs(cost_type=NORMAL))" '
                        '--search "astar(h,cost_type=NORMAL,max_time={})"',
            'max-astar': '--heuristic "h=hmax(transform=adapt_costs(cost_type=NORMAL))"'
                         ' --search "astar(h,cost_type=NORMAL,max_time={})"',

            # Suboptimal
            'ff-astar': '--heuristic "h=ff(transform=adapt_costs(cost_type=NORMAL))" '
                        '--search "astar(h,cost_type=NORMAL,max_time={})"',
            'ff-lazy': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                       '--search "lazy_greedy([h],preferred=[h],max_time={})" ',
        }
        self.plan_file = plan_file
        self.fd_path = fd_path #os.path.join(os.getenv('FAST_DOWNWARD_PATH'), "fast-downward.py")
        self.fd_exec_params = self.search_options[alias].format(timeout)
        # self.fd_exec_params = "--alias seq-sat-lama-2011 "

    def plan(self, domain_file, problem_file, debug=False):
        start_time = time.time()
        command = "{} --overall-time-limit {} --plan-file {} --sas-file {} {} {} {} ".format(
            self.fd_path,  self.timeout*3, self.plan_file, self.plan_file+'_temp', domain_file, problem_file, self.fd_exec_params
        )
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=None, close_fds=True)
        output, error = proc.communicate()
        runtime = time.time() - start_time
        if debug:
            print(command)
            if error:
                print(error.decode('utf-8'))
            print(output.decode('utf-8'))
            print('Search runtime:', runtime)
        if "Solution found." not in output.decode('utf-8'):
            return None, runtime
        plan = self.read_sas(self.plan_file)
        return plan, runtime

    def read_sas(self, filename):
      try:
        with open(filename, 'r') as f:
            plan = f.read()
      except:
        return None
      p = plan.split('\n')[:-2]
      retplan = []
      for act in p:
        tup = act.replace(')','').replace('(','').split(' ')
        tup = tuple(tup)
        retplan.append(tup)
      return retplan


if __name__ == '__main__':
    planner = PDDLPlanner(fd_path='/home/zhangyic/project/fastdownward/fast-downward.py')
    domain_file = "/home/zhangyic/project/ehsd_dev/src/sledmap/sledmap/plan/domains/teach/teach_domain.pddl"
    # problem_file = "/data/simbot/teach-planning-test/problems/Knives_isPickedUp_514288.pddl"

    # planner = PDDLPlanner(fd_path='/home/ubuntu/downward/fast-downward.py')
    # domain_file = "/home/ubuntu/ehsd_dev/src/sledmap/sledmap/plan/domains/teach/teach_domain.pddl"

    save_profile = {
        "bb5a2ba36c4ca244_bd02.game.json": {
            # GT subgoals: 
            # 0 - ('Knives', 'isPickedUp'),
            # 1 - ('Bread', 'isSliced'),
            # 2 - ['BreadSliced', 'parentReceptacles', 'Toaster'],
            # 3 - ('BreadSliced', 'isCooked'),
            # 4 - ['Toast', 'parentReceptacles', 'DiningTable'],
            # 5 - ('Plate', 'isPickedUp'),
            # 6 - ['Plate', 'parentReceptacles', 'DiningTable'],
            # 7 - ['Toast', 'parentReceptacles', 'Plate']
            2:0,  # [('goto', 'bot', 'start_loc', 'knife_0'), ('pickup', 'knife_0')]
            4:1,  # [('slice', 'bread_0', 'knife_0')]
            5:2,  # [('pickup', 'breadsliced_0'), ('search', 'toaster_dummy'), ('goto', 'bot', 'start_loc', 'toaster_dummy'), ('place', 'breadsliced_0', 'toaster_dummy')]
            21:3, # [('toggleon', 'toaster_0')]
            23:4, # [('pickup', 'breadsliced_4'), ('goto', 'bot', 'start_loc', 'diningtable_0'), ('place', 'breadsliced_4', 'diningtable_0')]
            28:5  # [('goto', 'bot', 'start_loc', 'plate_0'), ('pickup', 'plate_0')]
        },
        "8d6b8732d5394f80_81da.game.json": {
            # GT subgoals: 
            # 0 - ('Mug', 'isPickedUp'),
            # 1 - ('Mug', 'isClean'),
            # 2 - ('Mug', 'isEmptied'),
            # 3 - ('Mug', 'simbotIsFilledWithCoffee')
            2:0,  # [('search', 'mug_dummy'), ('goto', 'bot', 'start_loc', 'mug_dummy'), ('pickup', 'mug_dummy')]
            20:0, # [('goto', 'bot', 'start_loc', 'mug_0'), ('pickup', 'mug_0')]
            25:0, # [('pickup', 'mug_0')]
            26:1, # [('pickup', 'mug_0'), ('place', 'mug_0', 'sink_0'), ('toggleon', 'faucet_0')]
            27:1, # [('place', 'mug_0', 'sink_0'), ('toggleon', 'faucet_0')]
            28:2, # the mug is not filled yet. So there is nothing to be emptied and it is reasonable that we do not find any plan.
            29:2, # [('pickup', 'mug_0'), ('pourtosink', 'mug_0', 'sink_0')]
            31:2, # [('pourtosink', 'mug_0', 'sink_0')]
            32:3, # [('place', 'mug_0', 'coffeemachine_0')]  Note: the coffeemachine is alreadly turned on.
            33:3  # [('search', 'mug_dummy'), ('goto', 'bot', 'start_loc', 'mug_dummy'), ('pickup', 'mug_dummy'), ('goto', 'bot', 'mug_dummy', 'coffeemachine_0'), ('place', 'mug_dummy', 'coffeemachine_0')]
                  # (reasonable since the only mug is filled with coffee and removed, so a dummy mug is added)
        },
        "17dfd631e1080392_748e.game.json": {
            # GT subgoals: 
            # 0 - ('HousePlant', 'isFilledWithLiquid')
            5: 0, # [('goto', 'bot', 'start_loc', 'diningtable_0'), ('pickup', 'bowl_0'), ('goto', 'bot', 'diningtable_0', 'houseplant_0'), ('pourtofillable', 'bowl_0', 'houseplant_0')]
            27:0  # [('pickup', 'cup_0'), ('pourtofillable', 'cup_0', 'houseplant_0')]  already face a filled cup.
        },

        "5ab41ea72487ce0f_e5be.game.json": {
            # GT subgoals: 
            # 0 - [('Pot', 'isPickedUp'),
            # 1 - ('Pot', 'isFilledWithLiquid'),
            # 2 - ('Potato', 'isPickedUp'),
            # 3 - ['Potato', 'parentReceptacles', 'Pot'],
            # 4 - ['Pot', 'parentReceptacles', 'StoveBurner'],
            # 5 - ('Potato', 'simbotIsBoiled')]
            1:0,   # [('search', 'pot_dummy'), ('goto', 'bot', 'start_loc', 'pot_dummy'), ('pickup', 'pot_dummy')]
            80:1,  # [('pickup', 'pot_1'), ('goto', 'bot', 'start_loc', 'sink_0'), ('place', 'pot_1', 'sink_0')]
            94:1,  # [('pickup', 'bottle_0'), ('pourtofillable', 'bottle_0', 'pot_1')]
            97:1,  # [('place', 'potato_0', 'countertop_0'), ('pickup', 'bottle_0'), ('pourtofillable', 'bottle_0', 'pot_1')]
            107:3, # [('pickup', 'potato_0'), ('goto', 'bot', 'start_loc', 'pot_0'), ('place', 'potato_0', 'pot_0')]
            108:4, # [('pickup', 'potato_0'), ('place', 'potato_0', 'pot_0')]
            109:5, # [('pickup', 'pot_1'), ('place', 'potato_0', 'pot_1'), ('goto', 'bot', 'start_loc', 'stoveburner_0'), ('place', 'pot_1', 'stoveburner_0'), ('toggleon', 'stoveburner_0')]
            132:5  # [('toggleon', 'stoveburner_2')]
        },

        "5f8216a4b4ab3214_ef95.game.json":{
            # GT subgoals: 
            # 0 - [('Potato', 'isPickedUp'),
            # 1 - ('Potato', 'isSliced'),
            # 2 - ('PotatoSliced', 'isCooked'),
            # 3 - ['CookedPotatoSlice', 'parentReceptacles', 'Plate']]
            80:1,  # [('place', 'potato_0', 'countertop_1'), ('goto', 'bot', 'start_loc', 'butterknife_0'), ('pickup', 'butterknife_0'), ('goto', 'bot', 'butterknife_0', 'potato_0'), ('slice', 'potato_0', 'butterknife_0')]
            96:1,  # [('place', 'potato_0', 'countertop_1'), ('goto', 'bot', 'start_loc', 'butterknife_0'), ('pickup', 'butterknife_0'), ('goto', 'bot', 'butterknife_0', 'potato_0'), ('slice', 'potato_0', 'butterknife_0')]
            108:2, # [('pickup', 'potatosliced_0'), ('goto', 'bot', 'start_loc', 'countertop_0'), ('goto', 'bot', 'countertop_0', 'pan_0'), ('place', 'potatosliced_0', 'pan_0')]
            133:2, # [('place', 'potatosliced_5', 'pan_1')]
            134:3, # [('pickup', 'potatosliced_5'), ('goto', 'bot', 'start_loc', 'plate_0'), ('place', 'potatosliced_5', 'plate_0')]
            139:3  # [('pickup', 'potatosliced_5'), ('goto', 'bot', 'start_loc', 'plate_0'), ('place', 'potatosliced_5', 'plate_0')]
        },

        "b2815444299854b0_d30e.game.json": {
            # GT subgoals: 
            # 0 - [('Knives', 'isPickedUp'),
            # 1 - ('Bread', 'isSliced'),
            # 2 - ['BreadSliced', 'parentReceptacles', 'Toaster'],
            # 3 - ('BreadSliced', 'isCooked'),
            # 4 - ['Toast', 'parentReceptacles', 'Plate'],
            # 5 - ['BreadSliced', 'parentReceptacles', 'Toaster'],
            # 6 - ('BreadSliced', 'isCooked'),
            # 7 - ['Toast', 'parentReceptacles', 'Plate'],
            # 8 - ('Lettuce', 'isSliced'),
            # 9 - ['LettuceSliced', 'parentReceptacles', 'Plate'],
            # 10 - ['LettuceSliced', 'parentReceptacles', 'Plate']]
            25:1,  # [('slice', 'bread_1', 'knife_0')]
            37:3,  # [('toggleon', 'toaster_0'), ('place', 'breadsliced_4', 'toaster_0')] # Note: not good but it is correct..
            39:3,  # [('toggleon', 'toaster_0')]
            41:4,  # [('pickup', 'breadsliced_4'), ('place', 'breadsliced_4', 'plate_1')]
            47:6,  # [('toggleon', 'toaster_0'), ('place', 'breadsliced_2', 'toaster_0')]
            48:7,  # [('toggleon', 'toaster_0'), ('toggleoff', 'toaster_0'), ('pickup', 'breadsliced_2'), ('place', 'breadsliced_2', 'plate_1')]
            65:8,  # [('slice', 'lettuce_1', 'knife_0')]
            71:9,  # [('place', 'knife_0', 'countertop_2'), ('pickup', 'lettucesliced_0'), ('goto', 'bot', 'start_loc', 'countertop_0'), ('goto', 'bot', 'countertop_0', 'plate_1'), ('place', 'lettucesliced_0', 'plate_1')]
            90:10, # [('goto', 'bot', 'start_loc', 'lettucesliced_1'), ('pickup', 'lettucesliced_1'), ('goto', 'bot', 'lettucesliced_1', 'plate_1'), ('place', 'lettucesliced_1', 'plate_1')]
            110:10  # [('place', 'lettucesliced_1', 'plate_1')]
        },

        "67350daaa528a23b_b2f2.game.json": {
            # GT subgoals: 
            # 0 - [('Mug', 'isPickedUp'),
            # 1 - ('Mug', 'isClean'),
            # 2 - ('Mug', 'isEmptied'),
            # 3 - ['Mug', 'parentReceptacles', 'CoffeeMachine'],
            # 4 - ('Mug', 'simbotIsFilledWithCoffee'),
            # 5 - ('Mug', 'isEmptied'),
            # 6 - ('Coffee', 'isPickedUp'),
            # 7 - ['Coffee', 'parentReceptacles', 'DiningTable'],
            # 8 - ('Mug', 'simbotIsFilledWithCoffee'),
            # 9 - ['Coffee', 'parentReceptacles', 'DiningTable'],
            # 10 - ('Knives', 'isPickedUp'),
            # 11 - ('Bread', 'isSliced'),
            # 12 - ('Bread', 'isSliced'),
            # 13 - ('BreadSliced', 'isCooked'),
            # 14 - ['BreadSliced', 'parentReceptacles', 'Plate'],
            # 15 - ('BreadSliced', 'isCooked'),
            # 16 - ['BreadSliced', 'parentReceptacles', 'Plate'],
            # 17 - ['PlateOfToast', 'parentReceptacles', 'DiningTable']]
            35:3, 
            40:8, 
            201:16, 
            231:17
        },
    }

    import json
    import argparse
    from sledmap.mapper.env.teach.teach_subgoal import TeachSubgoal
    from sledmap.plan.domains.teach.pddl_problem import TeachPDDLProblem
    from definitions.teach_object_state import create_default_object_state

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file-path', 
    default="/data/simbot/teach-planning-test/b2815444299854b0_d30e/b2815444299854b0_d30e_71_9.json", 
    type=str, help="File path containing JSON")
    args = parser.parse_args()

    # with open(args.test_file_path, "r") as problem_file:
    #     problem_json = json.load(problem_file)    
    
    # sg_tuple = problem_json["subgoal"]
    # sg = TeachSubgoal.create_from_predicted_tuple(sg_tuple)
    # # sg = TeachSubgoal(predicate='isClear', subject_constraint={'objectType': 'SinkBasin', 'objectId': 'SinkBasin_0'})
    # pddl_state = problem_json['state']
    # pddl_state = TeachPDDLProblem.prune_symbolic_state(pddl_state, sg)
    
    # # for i in list(pddl_state.keys()):
    # #     if 'LettuceSliced_' in i:
    # #         del pddl_state[i]
    #     # if 'Toaster_' in i:
    #     #     pddl_state[i]['isToggled'] = True
    #     # elif 'Bowl' in i:
    #     #     pddl_state[i]['parentReceptacles'] = []
    #     #     # pddl_state[i]['interactable'] = True
    #     # elif 'HousePlant' in i:
    #     #     pddl_state[i]['interactable'] = True
    

    # # for obj_cls in {'Cabinet', 'LettuceSliced'}:
    # #     iid = "%s_dummy"%obj_cls
    # #     dummy_object = {
    # #         'objectId': iid,
    # #         'objectType': obj_cls,
    # #     }
    # #     dummy_state = create_default_object_state()
    # #     dummy_state['visible'].set_value(False)
    # #     dummy_state['isObserved'].set_value(False)
    # #     for state_name in dummy_state:
    # #         dummy_object[state_name] = dummy_state[state_name]()
        
    # #     pddl_state[iid] = dummy_object
    
    # # pddl_state['LettuceSliced_dummy']['parentReceptacles'] = ['Cabinet_dummy']

    # pddl_problem = TeachPDDLProblem(subgoal=sg, symbolic_state_dict=pddl_state)
    # print(pddl_problem.pddl_problem_str)
    # problem_file = pddl_problem.save_problem(save_path='/data/simbot/teach-planning-test/problems')
    # print(problem_file)

    problem_file = '/data/simbot/teach-eval/neural_symbolic/0829_debug/pddl_problems/Plate_parentReceptacles_CounterTop_660207.pddl'
    # problem_file = '/home/ubuntu/teach-eval/neural_symbolic/test/pddl_problems/HousePlant_isFilledWithLiquid_537745.pddl'
    
    plan, runtime = planner.plan(domain_file, problem_file, True)
    pprint(plan)
    print(plan)