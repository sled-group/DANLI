# AI2THOR Meta Data

* **teach_meta_data_files**: The same as src/teach/meta_data_files which includes:
    * Ai2thor resources
        * data partition (the same as ALFRED)
        * action name to id mappings
        * custom object semantic classes
    * World configs
    * Task definitions
    * Default game definitions
* **knob_to_burner**: Mapping form knob id to burner id. Adopt from [PIGLET](https://github.com/rowanz/piglet/tree/main/data). 
* **object_dict**: Collection of all the object types in iTHOR.
* **object_to_id**: Obejct types to id mapping. Seperate pickable objects and receptacles. 
* **property_dict**: Object type to affordance mapping. 