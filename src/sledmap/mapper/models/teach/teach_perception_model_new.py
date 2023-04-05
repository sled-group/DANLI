import imageio
from PIL import Image
from matplotlib import pyplot as plt
from typing import List

import numpy as np
import torch
from torch import nn

from clip.model import VisionTransformer
from clip import clip
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from numpy import asarray
import numpy as np
from mmdet.core import INSTANCE_OFFSET, bbox2result

from definitions.teach_objects import (
    THING_NAMES,
    STUFF_NAMES,
    get_object_affordance,
    get_object_receptacle_compatibility,
    ObjectClass,
)
from sledmap.mapper.env.teach.teach_object_instance import ObjectInstanceDetection2D


class StateEstimator(torch.nn.Module):
    def __init__(self, enlarge_ratio=0.25):
        super().__init__()
        self.enlarge_ratio = enlarge_ratio
        self.cat_embedder = nn.Embedding(num_embeddings=142, embedding_dim=8)
        self.img_encoder = VisionTransformer(
            input_resolution=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=12,
            output_dim=512,
        )
        self.img_preprocess = clip._transform(224)
        self.img_proj = torch.nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.merged_ffn = torch.nn.Sequential(
            nn.Linear(128 + 8, 128), nn.ReLU(), nn.Linear(128, 3), nn.Sigmoid()
        )

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
        if len(raw_imgs) > 0 and type(raw_imgs[0]) != Image.Image:
            raw_imgs = [Image.fromarray(raw_img) for raw_img in raw_imgs]
        standard_imgs = [self.img_preprocess(raw_img) for raw_img in raw_imgs]
        image_input = torch.tensor(np.stack(standard_imgs)).to(
            dtype=self.img_encoder.conv1.weight.dtype,
            device=self.img_encoder.conv1.weight.device,
        )
        image_features = self.img_encoder(image_input).float()
        return image_features

    def train_preprocess_a_frame(self, frame, states_info):
        obj_centric_imgs = []
        for s in states_info:
            obj_centric_imgs.append(self._crop_by_bbox(frame, s))
        return obj_centric_imgs

    def _crop_by_bbox(self, frame_img, state_info):
        x, y, w, h = state_info["bbox"]
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


def build_state_estimator(ckpt_path, device):
    M = StateEstimator()
    M.load_state_dict(torch.load(ckpt_path, map_location=device))
    return M


def parse_detection_result(ins_results):
    bbox_results, mask_results = ins_results
    out = []
    for cat_id in range(len(bbox_results)):
        this_cat_bboxs = bbox_results[cat_id]
        this_cat_masks = mask_results[cat_id]
        cat_name = ObjectClass.id_to_name(cat_id + 1)
        seagull_idx = cat_id + 1
        for i in range(this_cat_bboxs.shape[0]):
            instance = {}
            instance["bbox"] = this_cat_bboxs[i][:-1].astype(int).tolist()
            instance["score"] = float(this_cat_bboxs[i][-1])
            instance["mask"] = torch.from_numpy(this_cat_masks[i]).to("cpu")
            instance["class_id"] = seagull_idx
            instance["class_name"] = cat_name
            instance["states"] = {}
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
        name = r["class_name"]
        interests = name_to_interested_states(name)
        if len(interests) == 0:
            continue
        x1, y1, x2, y2 = r["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x1 >= x2 or y1 >= y2:
            continue
        result_idxs.append(idx)
        interested_states.append(interests)
        obj_frame = img[y1:y2, x1:x2].copy()
        obj_frames.append(obj_frame)
        class_ids.append(r["class_id"])
        class_names.append(r["class_name"])
    return obj_frames, class_ids, class_names, interested_states, result_idxs


def name_to_interested_states(cat_name):
    # if cat_name in ['StoveBurner', "Faucet", "ShowerHead"]:
    if cat_name in ["StoveBurner", "Faucet", "ShowerHead", "CoffeeMachine", "Toaster"]:
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
        for s_idx, s_name in enumerate(["isDirty", "isFilledWithWater", "isToggled"]):
            if s_name in interested_states[i]:
                this_states[s_name] = float(preds[i][s_idx])
        states_for_objects.append(this_states)
    return states_for_objects


def merge_states_to_pred(parsed_results, states_for_objs, result_idxs):
    for idx, states in zip(result_idxs, states_for_objs):
        parsed_results[idx]["states"] = states
    return parsed_results


class PerceptionModel(torch.nn.Module):
    def __init__(self, panoptic_config, panoptic_ckpt, state_estimator_ckpt, device):
        super().__init__()
        self.panoptic_model = init_detector(
            panoptic_config, panoptic_ckpt, device=device
        )
        self.state_predictor = build_state_estimator(state_estimator_ckpt, device)
        self.state_predictor.eval()

        self.mask_rcnn = None
        self.device = device

    def _panoptic_to_semantic(self, panoptic_result):
        # one hot vector, 900 x 900 resolution
        # device = panoptic_result.device
        semantic_seg = torch.zeros(142, 900, 900, dtype=torch.half, device=self.device)
        dense_seg = panoptic_result % INSTANCE_OFFSET
        cats = np.unique(dense_seg.flatten())
        # cats = np.unique(panoptic_result.flatten())
        for idx, this_id in enumerate(cats):
            name = ObjectClass.id_to_name(this_id + 1)
            seagull_idx = ObjectClass[name].value
            # this_region = this_id == panoptic_result
            this_region = this_id == dense_seg
            if name in ["Wall", "Door"]:
                semantic_seg[seagull_idx][this_region] = 0.4
            else:
                semantic_seg[seagull_idx][this_region] = 1.0
        return semantic_seg

    def _parse_instance_result(self, result, img):
        out = parse_detection_result(result)
        (
            obj_frames,
            class_ids,
            class_names,
            interested_states,
            result_idxs,
        ) = prepare_state_in(out, img)
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
        m2former_result = inference_detector(self.panoptic_model, img)
        pan_results = m2former_result["pan_results"]
        ins_results = m2former_result["ins_results"]
        semantic_map = self._panoptic_to_semantic(pan_results)
        instance_detection_result = self._parse_instance_result(ins_results, img)
        return instance_detection_result, semantic_map
