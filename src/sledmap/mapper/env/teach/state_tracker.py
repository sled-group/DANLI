import copy
from curses.ascii import FF
import torch
import numpy as np
import math
from transforms3d import euler
import torchvision
import PIL

from definitions.teach_objects import ithor_oid_to_object_class, is_interactable
import definitions.teach_object_state as teach_obj_state 
from definitions.teach_object_semantic_class import SemanticClass

import sledmap.mapper.env.teach.segmentation_definitions as segdef
from sledmap.mapper.env.teach.teach_env_params import FRAME_SIZE
from sledmap.mapper.env.teach.teach_action import TeachAction
from sledmap.mapper.env.teach.teach_observation import TeachObservation
from sledmap.mapper.env.teach.teach_object_instance import ObjectInstanceDetection2D 

SmallHandHeldObjects = SemanticClass.get_all_objs_in_semcls("SmallHandheldObjects")
Food = SemanticClass.get_all_objs_in_semcls("Food")
Knives = SemanticClass.get_all_objs_in_semcls("Knives")
SmallPickedUps = SmallHandHeldObjects | Food | Knives

class PoseInfo():
    """
    Given all the different inputs from AI2Thor event, constructs a pose matrix and a position vector
    to add to the observation.
    """

    def __init__(self,
                 cam_horizon_deg,
                 cam_pos_enu,
                 rot_3d_enu_deg,
                 ):
        self.cam_horizon_deg = cam_horizon_deg
        self.cam_pos_enu = cam_pos_enu
        self.rot_3d_enu_deg = rot_3d_enu_deg

    def is_close(self, pi: "PoseInfo"):
        horizon_close = math.isclose(self.cam_horizon_deg, pi.cam_horizon_deg, abs_tol=1e-3, rel_tol=1e-3)
        cam_pos_close = [math.isclose(a, b) for a,b in zip(self.cam_pos_enu, pi.cam_pos_enu)]
        rot_close = [math.isclose(a, b) for a,b in zip(self.rot_3d_enu_deg, pi.rot_3d_enu_deg)]
        all_close = horizon_close and cam_pos_close[0] and cam_pos_close[1] and cam_pos_close[2] and rot_close[0] and rot_close[1] and rot_close[2]
        return all_close

    @classmethod
    def from_ai2thor_event(cls, event):
        # Unity uses a left-handed coordinate frame with X-Z axis on ground, Y axis pointing up.
        # We want to convert to a right-handed coordinate frame with X-Y axis on ground, and Z axis pointing up.
        # To do this, all you have to do is swap Y and Z axes.

        cam_horizon_deg = event.metadata['agent']['cameraHorizon']

        # Translation from world origin to camera/agent position
        cam_pos_dict_3d_unity = event.metadata['cameraPosition']
        # Remap Unity left-handed frame to ENU right-handed frame (X Y Z -> X Z -Y)
        cam_pos_enu = [cam_pos_dict_3d_unity['z'],
                       -cam_pos_dict_3d_unity['x'],
                       -cam_pos_dict_3d_unity['y']]

        # ... rotation to agent frame (x-forward, y-left, z-up)
        rot_dict_3d_unity = event.metadata['agent']['rotation']
        rot_3d_enu_deg = [rot_dict_3d_unity['x'], rot_dict_3d_unity['z'], rot_dict_3d_unity['y']]

        return PoseInfo(cam_horizon_deg=cam_horizon_deg,
                        cam_pos_enu=cam_pos_enu,
                        rot_3d_enu_deg=rot_3d_enu_deg)

    @classmethod
    def create_new_initial(cls):
        cam_horizon_deg = 30.0
        cam_pos_enu = [0.0, 0.0, -1.576]
        rot_3d_enu_deg = [0.0, 0.0, 0.0]
        return PoseInfo(cam_horizon_deg=cam_horizon_deg,
                        cam_pos_enu=cam_pos_enu,
                        rot_3d_enu_deg=rot_3d_enu_deg)

    def simulate_successful_action(self, action: TeachAction):
        MOVE_STEP = 0.25
        PITCH_STEP = 30 # TODO: TEACh setting
        YAW_STEP = 90

        if action.action_type == "Turn Left":
            self.rot_3d_enu_deg[2] = (self.rot_3d_enu_deg[2] - YAW_STEP) % 360
        elif action.action_type == "Turn Right":
            self.rot_3d_enu_deg[2] = (self.rot_3d_enu_deg[2] + YAW_STEP) % 360
        elif action.action_type == "Forward":
            # TODO: Solve this with a geometry equation instead
            if math.isclose(self.rot_3d_enu_deg[2] % 360, 270):
                self.cam_pos_enu[1] += MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 90):
                self.cam_pos_enu[1] -= MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 180):
                self.cam_pos_enu[0] -= MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 0) or math.isclose(self.rot_3d_enu_deg[2] % 360, 360):
                self.cam_pos_enu[0] += MOVE_STEP
            else:
                raise ValueError("Agent doesn't appear to be on a 90-degree grid! This is not supported")
        elif action.action_type == "Backward":
            # TODO: Solve this with a geometry equation instead
            if math.isclose(self.rot_3d_enu_deg[2] % 360, 270):
                self.cam_pos_enu[1] -= MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 90):
                self.cam_pos_enu[1] += MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 180):
                self.cam_pos_enu[0] += MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 0) or math.isclose(self.rot_3d_enu_deg[2] % 360, 360):
                self.cam_pos_enu[0] -= MOVE_STEP
            else:
                raise ValueError("Agent doesn't appear to be on a 90-degree grid! This is not supported")
        elif action.action_type == "Pan Right":
            # TODO: Solve this with a geometry equation instead
            if math.isclose(self.rot_3d_enu_deg[2] % 360, 270):
                self.cam_pos_enu[0] += MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 90):
                self.cam_pos_enu[0] -= MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 180):
                self.cam_pos_enu[1] += MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 0) or math.isclose(self.rot_3d_enu_deg[2] % 360, 360):
                self.cam_pos_enu[1] -= MOVE_STEP
            else:
                raise ValueError("Agent doesn't appear to be on a 90-degree grid! This is not supported")
        elif action.action_type == "Pan Left":
            # TODO: Solve this with a geometry equation instead
            if math.isclose(self.rot_3d_enu_deg[2] % 360, 270):
                self.cam_pos_enu[0] -= MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 90):
                self.cam_pos_enu[0] += MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 180):
                self.cam_pos_enu[1] -= MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 0) or math.isclose(self.rot_3d_enu_deg[2] % 360, 360):
                self.cam_pos_enu[1] += MOVE_STEP
            else:
                raise ValueError("Agent doesn't appear to be on a 90-degree grid! This is not supported")
        elif action.action_type == "Look Down":
            self.cam_horizon_deg = self.cam_horizon_deg + PITCH_STEP
        elif action.action_type == "Look Up":
            self.cam_horizon_deg = self.cam_horizon_deg - PITCH_STEP

    def get_agent_pos(self, device="cpu", tensorize=True):
        cam_pos = [
            -self.cam_pos_enu[0],
            -self.cam_pos_enu[1],
            -self.cam_pos_enu[2]
        ]
        if tensorize:
            cam_pos = torch.tensor(cam_pos, device=device, dtype=torch.float32)
        return cam_pos

    def get_pose_mat(self):
        cam_pos_enu = torch.tensor(self.cam_pos_enu)
        # Translation from world origin to camera/agent position
        T_world_to_agent_pos = np.array([[1, 0, 0, cam_pos_enu[0]],
                                         [0, 1, 0, cam_pos_enu[1]],
                                         [0, 0, 1, cam_pos_enu[2]],
                                         [0, 0, 0, 1]])

        # ... rotation to agent frame (x-forward, y-left, z-up)
        rot_3d_enu_rad = [math.radians(r) for r in self.rot_3d_enu_deg]
        R_agent = euler.euler2mat(rot_3d_enu_rad[0], rot_3d_enu_rad[1], rot_3d_enu_rad[2])
        T_agent_pos_to_agent = np.asarray([[R_agent[0, 0], R_agent[0, 1], R_agent[0, 2], 0],
                                           [R_agent[1, 0], R_agent[1, 1], R_agent[1, 2], 0],
                                           [R_agent[2, 0], R_agent[2, 1], R_agent[2, 2], 0],
                                           [0, 0, 0, 1]])

        # .. transform to camera-forward frame (x-right, y-down, z-forward) that ignores camera pitch
        R_agent_to_camflat = euler.euler2mat(0, math.radians(90), math.radians(-90))
        T_agent_to_camflat = np.asarray(
            [[R_agent_to_camflat[0, 0], R_agent_to_camflat[0, 1], R_agent_to_camflat[0, 2], 0],
             [R_agent_to_camflat[1, 0], R_agent_to_camflat[1, 1], R_agent_to_camflat[1, 2], 0],
             [R_agent_to_camflat[2, 0], R_agent_to_camflat[2, 1], R_agent_to_camflat[2, 2], 0],
             [0, 0, 0, 1]])

        # .. transform to camera frame (x-right, y-down, z-forward) that also incorporates camera pitch
        R_camflat_to_cam = euler.euler2mat(math.radians(self.cam_horizon_deg), 0, 0)
        T_camflat_to_cam = np.asarray([[R_camflat_to_cam[0, 0], R_camflat_to_cam[0, 1], R_camflat_to_cam[0, 2], 0],
                                       [R_camflat_to_cam[1, 0], R_camflat_to_cam[1, 1], R_camflat_to_cam[1, 2], 0],
                                       [R_camflat_to_cam[2, 0], R_camflat_to_cam[2, 1], R_camflat_to_cam[2, 2], 0],
                                       [0, 0, 0, 1]])

        # compose into a transform from world to camera
        T_world_to_cam = T_camflat_to_cam @ T_agent_to_camflat @ T_agent_pos_to_agent @ T_world_to_agent_pos
        T_world_to_cam = torch.from_numpy(T_world_to_cam).unsqueeze(0)
        return T_world_to_cam
    
    def __repr__(self):
        prt_str = 'Agent Pose: \n'
        prt_str += "pos: (%.2f %.2f %.2f)\n"%(tuple(self.get_agent_pos(tensorize=False))) 
        prt_str += "rot: (%.2f %.2f %.2f)\n"%(tuple(self.rot_3d_enu_deg)) 
        prt_str += "horizon: %d"%(self.cam_horizon_deg)
        return prt_str


class InventoryInfo():
    def __init__(self, inventory_objects):
        self.inventory_object_ids = inventory_objects

    @classmethod
    def create_empty_initial(cls):
        return InventoryInfo([])

    @classmethod
    def from_ai2thor_event(cls, event):
        # For each object in the inventory, mark the corresponding dimension in the inventory vector with a 1.0
        inventory_objects = []
        for object in event.metadata['inventoryObjects']:
            object_str = object['objectType'].split("_")[0]
            object_id = segdef.object_string_to_intid(object_str)
            inventory_objects.append(object_id)
        return InventoryInfo(inventory_objects)

    def simulate_successful_action(self, action, latest_observaton):
        if action.action_type == "Pickup":
            arg_id = action.object_type_id
            if len(self.inventory_object_ids) == 0:
                self.inventory_object_ids.append(arg_id)
        elif action.action_type == "Place":
            self.inventory_object_ids = []

    def get_inventory_vector(self, device="cpu"):
        num_objects = segdef.get_num_objects()
        inv_vector = torch.zeros([num_objects], device=device, dtype=torch.uint8)
        for object_id in self.inventory_object_ids:
            inv_vector[object_id] = 1
        return inv_vector
    
    def get_inventory_object_class(self):
        return [segdef.object_intid_to_string(i) for i in self.inventory_object_ids]

    def summarize(self):
        summary = f"Inventory with: {self.get_inventory_object_class()}"
        return summary


class StateTracker():
    """
    Converts raw RGB images and executed actions to AlfredObservation instances that eval:
    - Segmentation
    - Depth
    - Pose
    - Inventory information
    """

    def __init__(self,
                 args=None,
                 seg_model=None,
                 depth_model=None,
                 device="cuda",
                 fov=90,
                 logger=None,
                 debug=False):
        self.first_event = None
        self.latest_event = None
        self.latest_observation = None
        self.latest_action = None
        self.latest_extra_events = []

        self.pose_info = None
        self.inventory_info = None

        self.reference_seg = args.hlsm_use_gt_seg
        self.reference_obj_det = args.hlsm_use_gt_obj_det
        self.reference_depth = args.hlsm_use_gt_depth
        self.reference_pose = args.hlsm_use_gt_pose
        self.reference_inventory =args.hlsm_use_gt_inventory

        self.obj_det_conf_threshold = args.obj_det_conf_threshold
        self.state_det_conf_threshold = args.state_det_conf_threshold

        self.fov = fov
        self.device = device

        self.seg_model = seg_model
        self.depth_model = depth_model

        self.log_func = print if logger is None else logger.info if not debug else logger.debug
        self.debug = debug

    def reset(self, event=None):
        # First reset everything
        self.latest_event = event
        self.first_event = event
        self.latest_action = None
        self.latest_observation = None
        self.last_action_failed = False

        # Initialize pose and inventory
        if self.reference_pose and event is not None:
            self.pose_info = PoseInfo.from_ai2thor_event(event)
        else:
            self.pose_info = PoseInfo.create_new_initial()

        if self.reference_inventory and event is not None:
            self.inventory_info = InventoryInfo.from_ai2thor_event(event)
        else:
            self.inventory_info = InventoryInfo.create_empty_initial()

        # Make the first observation
        # self.latest_observation = self.update_state(event=event)

    def log_action(self, action: TeachAction):
        self.latest_action = action

    def get_observation(self) -> TeachObservation:
        return self.latest_observation
    
    def get_agent_pos(self, tensorize=False):
        return self.pose_info.get_agent_pos(tensorize=tensorize)
    
    def get_agent_inventory(self):
        return self.inventory_info.get_inventory_object_class()

    def update_state(self, frame=None, event=None) -> TeachObservation:

        # RGB Image:
        # Add batch dimension to each image
        if frame is not None:
            if frame.shape[0] != FRAME_SIZE:
                frame = torchvision.transforms.functional.resize(PIL.Image.fromarray(frame), FRAME_SIZE)
                frame = np.asarray(frame)
            rgb_image = torch.from_numpy(frame.copy()).permute((2, 0, 1)).unsqueeze(0).half() / 255
        else:
            rgb_image = torch.zeros((1, 3, FRAME_SIZE, FRAME_SIZE))

        #################### HLSM's failure check ###################################
        # Simple error detection from RGB image changes
        action_failed = False
        if self.latest_observation is not None:
            assert self.latest_action is not None, "Didn't log an action, but got two observations in a row?"
            rgb_diff = (rgb_image.to(self.device) - self.latest_observation.rgb_image).float().abs().mean()
            
            
            if self.latest_action.action_type in ["Slice"]:
                # slice can be very negligible, so we loose the rgb diff threshold
                if rgb_diff < 0.0001:
                    action_failed = True
            elif self.latest_action.action_type in ['ToggleOn', 'ToggleOff']:
                # not observable at all (sometimes)
                pass
            elif (
                self.latest_action.action_type in ["Pickup", "Place"] 
                and (
                    any([self.latest_action.object_type == i for i in SmallPickedUps])
                    or any([i in self.inventory_info.summarize() for i in SmallPickedUps])
                )
            ):
                if rgb_diff < 0.0003:
                    action_failed = True
            else:
                if rgb_diff < 0.001:
                    action_failed = True
                #print(f"Action: {self.latest_action} Success with RGB Diff: {rgb_diff}")
            self.last_action_failed = action_failed
            if action_failed:
                self.latest_action.mark_as_failed()

            check_result = "SUCCESS" if not action_failed else "FAILED"
            self.log_func(f"{self.latest_action}, RGB Diff: {rgb_diff:.5f} ===> %s" % check_result)
        #################################################################################

        # Use dead-reckoning to estimate the pose, and track state of the inventory
        if self.latest_action is not None and not action_failed:
            if not self.reference_pose:
                self.pose_info.simulate_successful_action(self.latest_action)
            
            if not self.reference_inventory:
                oinv = copy.deepcopy(self.inventory_info)
                self.inventory_info.simulate_successful_action(self.latest_action, self.latest_observation)
                if len(oinv.inventory_object_ids) != len(self.inventory_info.inventory_object_ids):
                    self.log_func(self.inventory_info.summarize())

        # Pose
        if self.reference_pose:
            self.pose_info = PoseInfo.from_ai2thor_event(event)

        T_world_to_cam = self.pose_info.get_pose_mat()
        cam_horizon_deg = [self.pose_info.cam_horizon_deg]
        agent_pos = self.pose_info.get_agent_pos(tensorize=True)

        # Inventory
        if self.reference_inventory:
            self.inventory_info = InventoryInfo.from_ai2thor_event(event)
        inventory_vector = self.inventory_info.get_inventory_vector()
        inventory_vector = inventory_vector.unsqueeze(0)

        if action_failed:
            self.latest_observation.inventory_vector = inventory_vector
            self.latest_observation.set_agent_pos(agent_pos)
            return 

        # Depth
        if self.reference_depth and event is not None:
            depth_image = torch.from_numpy(event.depth_frame.copy()).unsqueeze(0).unsqueeze(0)
            depth_image = torch.clamp(depth_image, min=0.0, max=5.0)
        else:
            rgb_image_c = rgb_image.clone()
            rgb_image_c = torchvision.transforms.functional.resize(rgb_image_c, 300)
            _, pred_depth = self.depth_model.predict(rgb_image_c.float().to(self.device))
            depth_image = pred_depth
            # depth_image = pred_depth.to("cpu") # TODO: Maybe skip this? We later move it to GPU anyway

        semantic_image = object_detections = None
        # Segmentation
        if self.reference_seg and event is not None:
            semantic_image = StateTracker._extract_reference_semantic_image(event)
            semantic_image = semantic_image.unsqueeze(0)
        
        if self.reference_obj_det and event is not None:
            # TODO: add batch support
            object_detections = StateTracker._extract_reference_object_detections(
                event, self.get_agent_inventory()
            )

        if semantic_image is None or object_detections is None:
            object_detections = []
            instance_list, pred_seg = self.seg_model.parse_img(frame)
            semantic_image = pred_seg.unsqueeze(0)
            for ins in instance_list:

                if ins['score'] < self.obj_det_conf_threshold or ins["mask"].sum() == 0:
                    continue

                object_state = {}
                for pred, conf in ins['states'].items():
                    if pred == 'isFilledWithWater':
                        pred = "simbotIsFilledWithWater"
                    bool_val = True if conf >= 0.5 else False
                    conf = conf if bool_val else 1 - conf
                    if conf > self.state_det_conf_threshold:
                        object_state[pred] = teach_obj_state.Unary(pred, bool_val, conf)
                
                det = ObjectInstanceDetection2D(
                    object_type=ins['class_name'], 
                    bbox_2d=ins['bbox'], 
                    conf_score=ins['score'],
                    instance_mask=ins['mask'], 
                    state=object_state
                )
                det.check_is_held(self.get_agent_inventory())
                object_detections.append(det)

                # if object_state:
                #     print(det, ":")
                #     for s in object_state:
                #         print(f" - {object_state[s]}")
                    
            
            ObjectInstanceDetection2D.parse_receptacles(object_detections)
            
            # debug
            # for o in object_detections:
            #     print(o.tmp_unique_id, o.conf_score)
            #     if o.state:
            #         print("predicted state:", o.state)
            

        observation = TeachObservation(rgb_image,
                                         depth_image,
                                         semantic_image,
                                         object_detections,
                                         inventory_vector,
                                         T_world_to_cam,
                                         self.fov,
                                         cam_horizon_deg)
        observation = observation.to(self.device)

        if False:
            self.log_func('agent pose:', str(self.pose_info))
            # self.log_func('gt agent pose:', str(PoseInfo.from_ai2thor_event(event)))
            self.log_func("inventory_info", self.inventory_info.summarize())
            # self.log_func("gt inventory_info", InventoryInfo.from_ai2thor_event(event).summarize())
            pass

        # TODO: Use pose instead:
        observation.set_agent_pos(agent_pos)
        self.latest_observation = observation

    @classmethod
    def _extract_reference_semantic_image(cls, event, device="cpu"):
        """
        The segmentation images that come from AI2Thor have unstable color<->object mappings.
        Instead, we can build up a one-hot object image from the dictionary of class masks
        """
        num_objects = segdef.get_num_objects()
        h, w = event.frame.shape[0:2]
        seg_image = torch.zeros([num_objects, h, w], dtype=torch.int16, device=device)

        inventory_obj_strs = set()
        for object in event.metadata['inventoryObjects']:
            inventory_obj_string = object['objectType'].split("_")[0]
            inventory_obj_strs.add(inventory_obj_string)

        for obj_str, class_mask in event.class_masks.items():
            obj_int = segdef.object_string_to_intid(obj_str)
            class_mask_t = torch.from_numpy(class_mask.astype(np.int16)).to(device)
            seg_image[obj_int] = torch.max(seg_image[obj_int], class_mask_t)
        return seg_image.type(torch.ByteTensor)
    
    @classmethod
    def _extract_reference_object_detections(cls, event, inventory_object_types):
        """
        Ground truth object instances in the current observation. 
        """
        instance_bboxes = event.instance_detections2D
        instance_masks = event.instance_masks
        object_meta = {i["objectId"]: i for i in event.metadata['objects']}
        
        burner_to_knob = {}
        for obj_id in object_meta:
            if "StoveKnob" in obj_id:
                burner_id = object_meta[obj_id]['controlledObjects'][0]
                burner_to_knob[burner_id] = obj_id
        
        object_detections = []
        for obj_id, bbox in instance_bboxes.items():
            obj_type = ithor_oid_to_object_class(obj_id)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if not is_interactable(obj_type) or obj_id not in object_meta or area<100:
                continue
            mask = torch.tensor(instance_masks[obj_id]).bool()

            obj_state = {}
            meta = object_meta[obj_id] 
            
            for pred in [
                "isFilledWithLiquid", 
                "simbotIsFilledWithWater", 
                # "simbotIsFilledWithCoffee", 
                "isDirty",
                "isToggled"
            ]:
                if pred in meta:
                    obj_state[pred] = teach_obj_state.Unary(pred, meta[pred], confidence=1.0)
                    if pred == 'isFilledWithLiquid':
                        obj_state['simbotIsFilledWithWater'] = teach_obj_state.Unary(
                            'simbotIsFilledWithWater', meta[pred], confidence=1.0
                        )
            
            if obj_id in burner_to_knob:
                obj_state['isToggled'] = teach_obj_state.Unary(
                    'isToggled', object_meta[burner_to_knob[obj_id]]['isToggled'], confidence=1.0
                )
            
            # for pred in ["parentReceptacles", "receptacleObjectIds"]:
            #     if pred in meta:
            #         meta_state = meta[pred] if meta[pred] is not None else [] 
            #         meta_state = {val: 1.0 for val in meta_state}
            #         obj_state[pred] = teach_obj_state.Relation(pred, meta_state)
            # TODO: resolve iTHOR object objectIds to instance id

            detection = ObjectInstanceDetection2D(
                object_type=obj_type,
                bbox_2d=bbox,
                instance_mask=mask,
                state=obj_state
            )
            detection.check_is_held(inventory_object_types)

            object_detections.append(detection)
        
        # Note: use estimated spatial relations here
        ObjectInstanceDetection2D.parse_receptacles(object_detections)
        return object_detections
    

    @classmethod
    def _extract_reference_inventory_vector(cls, event, device="cpu"):
        num_objects = segdef.get_num_objects()
        inv_vector = torch.zeros([num_objects], device=device, dtype=torch.uint8)
        # For each object in the inventory, mark the corresponding dimension in the inventory vector with a 1.0
        for object in event.metadata['inventoryObjects']:
            object_str = object['objectType'].split("_")[0]
            object_id = segdef.object_string_to_intid(object_str)
            inv_vector[object_id] = 1
        return inv_vector
