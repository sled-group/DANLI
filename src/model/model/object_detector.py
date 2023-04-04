import os, json
import cv2
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


class MaskRCNNDetector(nn.Module):
    def __init__(self, ckpt_file, object_list_file, device="cuda", conf_thresh=None):
        super().__init__()
        self.ckpt_file = ckpt_file
        self.object_list_file = object_list_file
        self.confidence_thresh = conf_thresh if conf_thresh is not None else 0.5
        self.device = device

        with open(object_list_file, "r") as f:
            self.all_objects = json.load(f)

        self.object_to_id = self.all_objects["recognizable"]
        self.class_num = len(self.object_to_id)
        self.id_to_object = {v: k for k, v in self.object_to_id.items()}
        # self.frame_size = 900

        self.config_maskrcnn()

        self.predictor = DefaultPredictor(self.cfg)

    def config_maskrcnn(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            )
        )
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.MODEL.WEIGHTS = self.ckpt_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            self.confidence_thresh
        )  # set a custom testing threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
            self.class_num
        )  # NOTE: Corresponds to #REC_OBJ
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.MODEL.DEVICE = self.device

        self.cfg = cfg

        MetadataCatalog.get("teach").set(
            thing_classes=list(self.object_to_id.keys()), evaluator_type="coco"
        )
        self.teach_meta = MetadataCatalog.get("teach")

    def get_predictions(self, frame):
        """perform object detection and instance segmentation on a single frame

        :param frame: an image (numpy array) of opencv style (BGR)
        :return: object of type "instances" defined in detectron2.
                 For more details please refer to
                 https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        """
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        instances = self.predictor(frame)["instances"]
        return instances

    def get_visibile_objects(self, instances):
        visble_obj_ids = instances.pred_classes.tolist()
        return [self.id_to_object[i] for i in visble_obj_ids]

    def get_interaction_points(self, instances, target_object_class):
        """get interaction points (is any) for a target object class in the current frame

        :param instances: object of type "instances" defined in detectron2.
        :param target_object_class: target object to interact with
        :return: _description_
        """
        # TODO: check whether the center of the bbox is inside the mask!
        # instances = self.get_predictions(frame, target_object_class)
        if target_object_class == "None":
            return []
        target_idx = self.object_to_id[target_object_class]
        instances = instances[instances.pred_classes == target_idx]
        pred_boxes = instances.pred_boxes
        pred_centers = pred_boxes.get_centers()
        conf_scores = instances.scores
        return list(zip(pred_centers.tolist(), conf_scores.tolist()))

    def visulize(self, frame, instances):
        """visualize the predictions

        :param frame: an image (numpy array) of opencv style (BGR)
        :param instances: predictor outputs
        """

        v = Visualizer(frame[:, :, ::-1], metadata=self.teach_meta)
        v_out = v.draw_instance_predictions(instances.to("cpu"))
        img_plot = v_out.get_image()[:, :, ::-1]
        img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
        # cv2.imshow("sample",im)
        plt.imshow(img_plot)
        plt.show()
