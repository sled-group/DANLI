# Overview of Domain Definitions 
Here we define all the symbols used in our system to represent objects, actions, semantic classes, physical states and tasks, which provide the standard intput-output space for modules in our system. For example, the object classes defined in `ObjectClass` should be the output space of the object detector and the semantic parser (for object name recognition), as well as the space of defined `objectType` predicates in the PDDL domain. 

Note: The symbol set is domain-specific for the TEACh dataset which is based on AI2THOR 3.1.0. 

## Meta Data 
Metadata files adopted from the AI2THOR official document and the TEACh repo. They are used as the source to generate the domain definitions for our system.  

File structure:  
```
│  default_definitions.json                 # default action definitions used for TEACh 
│  ithor_documented_object_affordance.json  # object affordance copied from the AI2THOR doc 
│  ithor_documented_object_list.json        # name of all the 125 objects copied from the AI2THOR doc 
│  ithor_knob_to_burner.json                # mapping between stove knobs to the stove burners they control
│  process_meta_data.py                     # script to extract structural information from meta data
│  __init__.py
│
├─ai2thor_resources                         # ai2thor resources from the original TEACh repo 
│  │  action_idx_to_action_name.json        # mapping between action successive indexes to action names
│  │  action_to_action_idx.json             # mapping between action ids to their successive indexes
│  │  custom_object_classes.json            # custom semantic classes used in TEACh
│  │  __init__.py
│  │
│  └─alfred_split                           # train/valid/test scene split (same to ALFRED)
│
├─config                                    # custom object models (not sure what they are used for)
│
└─task_definitions                          # the original definitions for all the tasks in TEACh
        101__toast.json
        102__coffee.json
        103__clean_x.json
        104__sliced_x.json
        105__cooked_slice_of_x.json
        ...
```

The meta data are processed in `process_meta_data.py` to extract structural information about object, affordance, semantic classes and action definitions, etc. The extracted infomation are explicitly coded in the definition modules introduced below for the sake of readablility.  

## Definitions
We define the following symbol sets for our system.

### Objects 
In `teach_objects.py` we define: 
* All the object class names and their integer ids
* Affordance of each object (whether an object is pickable, openable etc)
* APIs for object class name and index handling

We additionaly define a mapping between object ids and unique colors across different scenes in `object_id_to_color.py` for visualization of object semantic segmentations. 

### Actions 
In `teach_actions.py` we define: 
* All the action names (includin both `navigation` and `interaction` actions) and their integer ids
* Applicability for `interaction` actions (e.g. what objects are valid for action `Pickup`)
* APIs for action name and index handling

### Object Semantic Classes
In `teach_object_semantic_class.py` we define: 
* All the semantic classes and what objects are belong to each of them. Besides the original classes defined by TEACh, we introduce additionaly classes to represent funcitonal and task-relevant semantics such as tools for cooking, different types of receptcales, and structural objects. 
* APIs for semantic class handling

### Object Physical States (Under Development)
In `teach_object_state.py` we define: 
* All the physical states we are interested from the iTHOR simulator as well as some custom states introduced by TEACh. 
* APIs for state checking (TODO)

### Tasks (Under Development)
In `teach_tasks.py` we define:
* All the tasks defined for TEACh such as their names, arguments, dependancies, goal states, etc.   

### Symbol to Language (Under Development)
In `symbol_to_language.py` we define: 
* Synthetic language representations for all the symbols used above.  For example,
    * `AppleSliced`: *slice of apple*
    * `N_Cooked_Slices_Of_X_In_Y(N=1, X=Potato, Y=Bowl)`: *make a cooked potato slice and put it in a bowl*
