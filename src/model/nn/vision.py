import os
import types
import torch
import contextlib

import torch.nn as nn

from torchvision import models, transforms
from torchvision.transforms import functional as F

# from alfred.gen import constants
from ..utils import data_util

constants = {}


class Resnet18(nn.Module):
    """
    pretrained Resnet18 from torchvision
    """

    def __init__(self, device, checkpoint_path=None, share_memory=False):
        super().__init__()
        self.device = device
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        if checkpoint_path is not None:
            print("Loading ResNet checkpoint from {}".format(checkpoint_path))
            model_state_dict = torch.load(checkpoint_path, map_location=device)
            model_state_dict = {
                key: value
                for key, value in model_state_dict.items()
                if "GU_" not in key and "text_pooling" not in key
            }
            model_state_dict = {
                key: value
                for key, value in model_state_dict.items()
                if "fc." not in key
            }
            model_state_dict = {
                key.replace("resnet.", ""): value
                for key, value in model_state_dict.items()
            }
            self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(torch.device(device))
        self.model = self.model.eval()
        if share_memory:
            self.model.share_memory()
        self._transform = Transforms.get_transform("default")

    def extract(self, x):
        x = self._transform(x).to(torch.device(self.device))
        return self.model(x)


class RCNN(nn.Module):
    """
    pretrained FasterRCNN or MaskRCNN from torchvision
    """

    def __init__(
        self,
        archi,
        device="cuda",
        checkpoint_path=None,
        share_memory=False,
        load_heads=False,
    ):
        super().__init__()
        self.device = device
        self.feat_layer = "3"
        if archi == "maskrcnn":
            self.model = models.detection.maskrcnn_resnet50_fpn(
                pretrained=(checkpoint_path is None),
                pretrained_backbone=(checkpoint_path is None),
                min_size=800,
            )
        elif archi == "fasterrcnn":
            self.model = models.detection.fasterrcnn_resnet50_fpn(
                pretrained=(checkpoint_path is None),
                pretrained_backbone=(checkpoint_path is None),
                min_size=224,
            )
        else:
            raise ValueError("Unknown model type = {}".format(archi))

        if archi == "maskrcnn":
            self._transform = self.model.transform
        else:
            self._transform = Transforms.get_transform("default")
        if not load_heads:
            for attr in ("backbone", "body"):
                self.model = getattr(self.model, attr)

        if checkpoint_path is not None:
            self.load_from_checkpoint(
                checkpoint_path, load_heads, device, archi, "backbone.body"
            )
        self.model = self.model.to(torch.device(device))
        self.model = self.model.eval()
        if share_memory:
            self.model.share_memory()

    def extract(self, images):
        if isinstance(
            self._transform, models.detection.transform.GeneralizedRCNNTransform
        ):
            images_normalized = self._transform(
                torch.stack([F.to_tensor(img) for img in images])
            )[0].tensors
        else:
            images_normalized = torch.stack([self._transform(img) for img in images])
        images_normalized = images_normalized.to(torch.device(self.device))
        model_body = self.model
        if hasattr(self.model, "backbone"):
            model_body = self.model.backbone.body
        features = model_body(images_normalized)
        return features[self.feat_layer]

    def load_from_checkpoint(self, checkpoint_path, load_heads, device, archi, prefix):
        print("Loading RCNN checkpoint from {}".format(checkpoint_path))
        state_dict = torch.load(checkpoint_path, map_location=device)
        if not load_heads:
            # load only the backbone
            state_dict = {
                k.replace(prefix + ".", ""): v
                for k, v in state_dict.items()
                if prefix + "." in k
            }
        else:
            # load a full model, replace pre-trained head(s) with (a) new one(s)
            num_classes, in_features = state_dict[
                "roi_heads.box_predictor.cls_score.weight"
            ].shape
            box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
            self.model.roi_heads.box_predictor = box_predictor
            if archi == "maskrcnn":
                # and replace the mask predictor with a new one
                in_features_mask = (
                    self.model.roi_heads.mask_predictor.conv5_mask.in_channels
                )
                hidden_layer = 256
                mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
                    in_features_mask, hidden_layer, num_classes
                )
                self.model.roi_heads.mask_predictor = mask_predictor
        self.model.load_state_dict(state_dict)

    def predict_objects(self, image, confidence_threshold=0.0, verbose=False):
        image = F.to_tensor(image).to(torch.device(self.device))
        output = self.model(image[None])[0]
        preds = []
        for pred_idx in range(len(output["scores"])):
            score = output["scores"][pred_idx].cpu().item()
            if score < confidence_threshold:
                continue
            box = output["boxes"][pred_idx].cpu().numpy()
            label = self.vocab_pred[output["labels"][pred_idx].cpu().item()]
            if verbose:
                print("{} at {}".format(label, box))
            pred = types.SimpleNamespace(label=label, box=box, score=score)
            if "masks" in output:
                pred.mask = output["masks"][pred_idx].cpu().numpy()
            preds.append(pred)
        return preds


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        archi,
        device="cuda",
        checkpoint=None,
        batch_size=32,
        share_memory=False,
        compress_type=None,
        load_heads=False,
    ):
        super().__init__()
        self.feat_shape = data_util.get_feat_shape(archi, compress_type)
        self.eval_mode = True
        if archi == "resnet18":
            assert not load_heads
            self.model = Resnet18(device, checkpoint, share_memory)
        else:
            self.model = RCNN(
                archi, device, checkpoint, share_memory, load_heads=load_heads
            )
        self.compress_type = compress_type

    def featurize(self, images, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        feats = []
        with (
            torch.set_grad_enabled(False)
            if not self.model.model.training
            else contextlib.nullcontext()
        ):
            for i in range(0, len(images), batch_size):
                images_batch = images[i : i + batch_size]
                feats.append(self.model.extract(images_batch))
        feat = torch.cat(feats, dim=0)
        if self.compress_type is not None:
            feat = data_util.feat_compress(feat, self.compress_type)
        assert self.feat_shape[1:] == feat.shape[1:]
        return feat

    def append_feature(self, image, features):
        feat = self.model.extract([image])
        if self.compress_type is not None:
            feat = data_util.feat_compress(feat, self.compress_type)
        return torch.cat([features, feat], dim=0)

    def predict_objects(self, image, verbose=False):
        with torch.set_grad_enabled(False):
            pred = self.model.predict_objects(image, verbose=verbose)
        return pred

    def train(self, mode):
        if self.eval_mode:
            return
        for module in self.children():
            module.train(mode)


class Transforms(object):
    @staticmethod
    def resize(img_size=224):
        # expects a PIL Image
        return transforms.Resize((img_size, img_size))

    @staticmethod
    def affine(degree=5, translate=0.04, scale=0.02):
        # expects a PIL Image
        return transforms.RandomAffine(
            degrees=(-degree, degree),
            translate=(translate, translate),
            scale=(1 - scale, 1 + scale),
            shear=None,
        )

    @staticmethod
    def random_crop(img_size=224):
        # expects a PIL Image
        return transforms.RandomCrop((img_size, img_size))

    @staticmethod
    def normalize():
        # expects a PIL Image
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def cutout(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0):
        # expects a tensor
        return transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    @staticmethod
    def get_transform(transform="default"):
        if transform == "default":
            return transforms.Compose([Transforms.resize(224), Transforms.normalize()])
        elif transform == "none":
            return transforms.ToTensor()
        elif transform == "crops":
            return transforms.Compose(
                [
                    Transforms.resize(240),
                    Transforms.random_crop(224),
                    Transforms.normalize(),
                ]
            )
        elif transform == "cutout":
            return transforms.Compose(
                [Transforms.resize(224), Transforms.normalize(), Transforms.cutout()]
            )
        elif transform == "affine":
            return transforms.Compose(
                [Transforms.resize(224), Transforms.affine(), Transforms.normalize()]
            )
        elif transform == "affine_crops":
            return transforms.Compose(
                [
                    Transforms.resize(240),
                    Transforms.random_crop(224),
                    Transforms.affine(),
                    Transforms.normalize(),
                ]
            )
        elif transform == "affine_crops_cutout":
            return transforms.Compose(
                [
                    Transforms.resize(240),
                    Transforms.random_crop(224),
                    Transforms.affine(),
                    Transforms.normalize(),
                    Transforms.cutout(),
                ]
            )
        elif transform == "affine_cutout":
            return transforms.Compose(
                [
                    Transforms.resize(224),
                    Transforms.affine(),
                    Transforms.normalize(),
                    Transforms.cutout(),
                ]
            )
        else:
            raise ValueError(
                "Image augmentation {} is not implemented".format(transform)
            )
