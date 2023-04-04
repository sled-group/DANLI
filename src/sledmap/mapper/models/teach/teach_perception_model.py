import imageio
from PIL import Image
from matplotlib import pyplot as plt
from typing import List

import numpy as np
import torch
from torch import nn
from sledmap.mapper.models.teach.test_mask_rcnn import MaskRCNNDetector

from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg

from clip.model import VisionTransformer
from clip import clip

from definitions.teach_objects import (
    THING_NAMES, 
    STUFF_NAMES, 
    get_object_affordance,
    get_object_receptacle_compatibility, 
    ObjectClass
)
# from sledmap.mapper.models.teach.mask_rcnn import MaskRCNNDetector
from sledmap.mapper.env.teach.teach_object_instance import ObjectInstanceDetection2D

class StateEstimator(torch.nn.Module):
    def __init__(self, enlarge_ratio=0.25):
        super().__init__()
        self.enlarge_ratio = enlarge_ratio
        self.cat_embedder = nn.Embedding(num_embeddings=142, embedding_dim=8)
        self.img_encoder = VisionTransformer(
            input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512)
        self.img_preprocess = clip._transform(224)
        self.img_proj = torch.nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.merged_ffn = torch.nn.Sequential(
            nn.Linear(128+8, 128), nn.ReLU(), nn.Linear(128, 3), nn.Sigmoid())

    def forward(self, imgs, cat_idxs):
        # given object centric images, return the prediction for each logic
        cat_feats = self.cat_embedder(cat_idxs)
        img_feats = self.imgs_encode(imgs)
        img_feats = self.img_proj(img_feats)
        merged_feats = torch.cat((img_feats, cat_feats), dim=-1)
        preds = self.merged_ffn(merged_feats)
        return preds

    def imgs_encode(self, raw_imgs):
        """
        image encoder 
        :param raw_imgs: list[np.ndarray] or list[PIL.Image.Image]
        """
        if len(raw_imgs)>0 and type(raw_imgs[0]) != Image.Image:
            raw_imgs = [Image.fromarray(raw_img) for raw_img in raw_imgs]
        standard_imgs = [self.img_preprocess(raw_img) for raw_img in raw_imgs]
        image_input = torch.tensor(np.stack(standard_imgs)).to(
            dtype=self.img_encoder.conv1.weight.dtype,
            device=self.img_encoder.conv1.weight.device
        )
        image_features = self.img_encoder(image_input).float()
        return image_features

    def train_preprocess_a_frame(self, frame, states_info):
        obj_centric_imgs = []
        for s in states_info:
            obj_centric_imgs.append(self._crop_by_bbox(frame, s))
        return obj_centric_imgs

    def _crop_by_bbox(self, frame_img, state_info):
        x, y, w, h = state_info['bbox']
        y_start = y - int(h * self.enlarge_ratio)
        y_end = y + h + int(h * self.enlarge_ratio)
        x_start = x - int(w * self.enlarge_ratio)
        x_end = x + w + int(h * self.enlarge_ratio)
        if y_start < 0:
            y_start = 0
        if y_end >= 900:
            y_end = 899
        if x_start < 0:
            x_start = 0
        if x_end >= 900:
            x_end = 899
        cropped_img = frame_img[y_start:y_end, x_start:x_end].copy()
        return cropped_img


register_coco_panoptic_separated("panoptic-training", {}, "", "", "", "", "")
STUFF_NAMES.insert(0, 'THINGS-YOU SHOULD NOT SEE THIS')
MetadataCatalog.get("panoptic-training_separated").set(
    thing_classes=THING_NAMES, stuff_classes=STUFF_NAMES)


def build_panoptic_predictor(ckpt_path, device):
    config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = ("panoptic-training_separated",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        config_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 1000
    # faster, and good enough for this toy dataset (default: 512)S
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(THING_NAMES)  # matching instance num
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(
        STUFF_NAMES) + 1  # +1 for the instance class
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.MODEL.DEVICE = device
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7
    predictor = DefaultPredictor(cfg)
    return predictor


def build_state_estimator(ckpt_path, device):
    M = StateEstimator()
    M.load_state_dict(torch.load(ckpt_path, map_location=device))
    return M


def panoptic_to_semantic(panoptic_result):
    # one hot vector, 900 x 900 resolution

    pan_seg, pan_info_list = panoptic_result['panoptic_seg']
    device = pan_seg.device
    semantic_seg = torch.zeros(142, 900, 900, dtype=torch.half, device=device)
    # print('panoptic_output:')
    for idx, this_info in enumerate(pan_info_list):
        this_id = this_info['id']
        name = None
        if this_info['isthing']:
            name = THING_NAMES[this_info['category_id']]
        else:
            name = STUFF_NAMES[this_info['category_id']]
        seagull_idx = ObjectClass[name].value
        this_region = this_id == pan_seg
        semantic_seg[seagull_idx][this_region] = True

        if name in ["Wall", "Door"]:
            semantic_seg[seagull_idx][this_region] = 0.4
        else:
           semantic_seg[seagull_idx][this_region] = 1.0
    return semantic_seg


def parse_detection_result(result):
    out = []
    for idx in range(result['scores'].shape[0]):
        instance = {}
        instance['bbox'] = result['pred_boxes'][idx].tensor.cpu(
        ).flatten().int().tolist()
        instance['score'] = float(result['scores'][idx].cpu())
        instance['class_id'] = int(result['pred_classes'][idx].cpu()) + 1
        instance['class_name'] = THING_NAMES[instance['class_id']-1]
        instance['mask'] = result['pred_masks'][idx].cpu()
        instance['states'] = {}
        out.append(instance)
    return out


def prepare_state_in(detection_result, img):
    N = len(detection_result)
    obj_frames = []
    class_ids = []
    class_names = []
    interested_states = []
    result_idxs = []
    for idx, r in enumerate(detection_result):
        name = r['class_name']
        interests = name_to_interested_states(name)
        if len(interests) == 0:
            continue
        [x1, y1, x2, y2] = r['bbox']
        if x1 >= x2 or y1 >= y2:
            continue
        result_idxs.append(idx)
        interested_states.append(interests)
        obj_frame = img[y1:y2, x1: x2].copy()
        obj_frames.append(obj_frame)
        class_ids.append(r['class_id'])
        class_names.append(r['class_name'])
    return obj_frames, class_ids, class_names, interested_states, result_idxs


def name_to_interested_states(cat_name):
    # if cat_name in ['StoveBurner', "Faucet", "ShowerHead"]:
    if cat_name in ['StoveBurner', "Faucet", "ShowerHead", "CoffeeMachine", "Toaster"]:
        return ["isToggled"]
    remain = []
    affs = get_object_affordance(cat_name)
    if "dirtyable" in affs:
        remain.append("isDirty")
    if "canFillWithLiquid" in affs:
        remain.append("isFilledWithWater")
    return remain


def preds_to_state_dict(preds, interested_states):
    states_for_objects = []
    preds = preds.cpu()
    for i, S in enumerate(interested_states):
        this_states = {}
        for s_idx, s_name in enumerate(["isDirty", 'isFilledWithWater', 'isToggled']):
            if s_name in interested_states[i]:
                this_states[s_name] = float(preds[i][s_idx])
        states_for_objects.append(this_states)
    return states_for_objects


def merge_states_to_pred(parsed_results, states_for_objs, result_idxs):
    for idx, states in zip(result_idxs, states_for_objs):
        parsed_results[idx]['states'] = states
    return parsed_results


class PerceptionModel(torch.nn.Module):
    def __init__(self, panoptic_ckpt, state_estimator_ckpt, device, maskrcnn_ckpt=None, obj_list_file=None):
        super().__init__()
        self.panoptic_predictor = build_panoptic_predictor(panoptic_ckpt, device)
        self.state_predictor = build_state_estimator(state_estimator_ckpt, device)
        self.state_predictor.eval()

        self.mask_rcnn = None
        # if maskrcnn_ckpt is not None:
        #     # self.mask_rcnn = MaskRCNNDetector(maskrcnn_ckpt, obj_list_file, device, conf_thresh=0.7)
        #     self.mask_rcnn = MaskRCNNDetector(maskrcnn_ckpt, device, conf_thresh=0.7)

        self.device = device

    def parse_instance_result(self, result, img):
        out = parse_detection_result(result)
        obj_frames, class_ids, class_names, interested_states, result_idxs = prepare_state_in(
            out, img)
        if not obj_frames:
            return out
        with torch.no_grad():
            preds = self.state_predictor(
                obj_frames, torch.tensor(class_ids, device=self.device).long()
            )
        states_for_objects = preds_to_state_dict(preds, interested_states)
        final = merge_states_to_pred(out, states_for_objects, result_idxs)
        return final

    def parse_img(self, img: np.ndarray):
        with torch.no_grad():
            panoptic_result = self.panoptic_predictor(img)
            if self.mask_rcnn is not None:
                # TODO: jiayi:complete this
                obj_detections = self.mask_rcnn.get_predictions(img)
            else:
                obj_detections = panoptic_result['instances'].get_fields()
        
        instance_detection_result = self.parse_instance_result(obj_detections, img)
        semantic_map = panoptic_to_semantic(panoptic_result)
        return instance_detection_result, semantic_map

# def result_to_sled_map_ins_list(instance_list):
#     for ins in instance_list:
#         ObjectInstanceDetection2D(object_type=ins['class_name'], bbox_2d=ins['class_name'], conf_score=ins['score'],
#                                   instance_mask=ins['mask'], state=ins['states'])
# img = imageio.imread('sample.jpg')
# plt.imshow(img)

# # %%
# P = PerceptionModel()

# # %%
# instance_list, semantic_map = P.parse_img(img)

# # %%
# instance_list[0].keys()

# # %%
