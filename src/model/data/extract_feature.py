import os, re
import torch
import types
from torch import nn
import json
from sacred import Ingredient, Experiment
from attrdict import AttrDict

import contextlib
from torchvision import models
from torchvision.transforms import functional as F

from pprint import pprint

import lmdb
import numpy as np
import time
from PIL import Image
from torchvision import transforms


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
            scale=(1-scale, 1+scale),
            shear=None)

    @staticmethod
    def random_crop(img_size=224):
        # expects a PIL Image
        return transforms.RandomCrop((img_size, img_size))

    @staticmethod
    def normalize():
        # expects a PIL Image
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    @staticmethod
    def cutout(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.):
        # expects a tensor
        return transforms.RandomErasing(
            p=p, scale=scale, ratio=ratio, value=value)

    @staticmethod
    def get_transform(transform='default'):
        if transform == 'default':
            return transforms.Compose([
                Transforms.resize(224),
                Transforms.normalize()])
        elif transform == 'none':
            return transforms.ToTensor()
        elif transform == 'crops':
            return transforms.Compose([
                Transforms.resize(240),
                Transforms.random_crop(224),
                Transforms.normalize()])
        elif transform == 'cutout':
            return transforms.Compose([
                Transforms.resize(224),
                Transforms.normalize(),
                Transforms.cutout()])
        elif transform == 'affine':
            return transforms.Compose([
                Transforms.resize(224),
                Transforms.affine(),
                Transforms.normalize()])
        elif transform == 'affine_crops':
            return transforms.Compose([
                Transforms.resize(240),
                Transforms.random_crop(224),
                Transforms.affine(),
                Transforms.normalize()])
        elif transform == 'affine_crops_cutout':
            return transforms.Compose([
                Transforms.resize(240),
                Transforms.random_crop(224),
                Transforms.affine(),
                Transforms.normalize(),
                Transforms.cutout()])
        elif transform == 'affine_cutout':
            return transforms.Compose([
                Transforms.resize(224),
                Transforms.affine(),
                Transforms.normalize(),
                Transforms.cutout()])
        else:
            raise ValueError('Image augmentation {} is not implemented'.format(transform))



def extract_features(images, extractor):
    if images is None:
        return None
    feat = extractor.featurize(images, batch=8)
    return feat.cpu()

def get_feat_shape(visual_archi, compress_type=None):
    '''
    Get feat shape depending on the training archi and compress type
    '''
    if visual_archi == 'fasterrcnn':
        # the RCNN model should be trained with min_size=224
        feat_shape = (-1, 2048, 7, 7)
    elif visual_archi == 'maskrcnn':
        # the RCNN model should be trained with min_size=800
        feat_shape = (-1, 2048, 10, 10)
    elif visual_archi == 'resnet18':
        feat_shape = (-1, 512, 7, 7)
    else:
        raise NotImplementedError('Unknown archi {}'.format(visual_archi))

    if compress_type is not None:
        if not re.match(r'\d+x', compress_type):
            raise NotImplementedError('Unknown compress type {}'.format(compress_type))
        compress_times = int(compress_type[:-1])
        feat_shape = (
            feat_shape[0], feat_shape[1] // compress_times,
            feat_shape[2], feat_shape[3])
    return feat_shape


def read_images(image_path_list):
    images = []
    for image_path in image_path_list:
        image_orig = Image.open(image_path)
        images.append(image_orig.copy())
        image_orig.close()
    return images


def feat_compress(feat, compress_type):
    '''
    Compress features by channel average pooling
    '''
    assert re.match(r'\d+x', compress_type) and len(feat.shape) == 4
    times = int(compress_type[:-1])
    assert feat.shape[1] % times == 0
    feat = feat.reshape((
        feat.shape[0], times,
        feat.shape[1] // times,
        feat.shape[2],
        feat.shape[3]))
    feat = feat.mean(dim=1)
    return feat

class Resnet18(nn.Module):
    '''
    pretrained Resnet18 from torchvision
    '''
    def __init__(self,
                 device,
                 checkpoint_path=None,
                 share_memory=False):
        super().__init__()
        self.device = device
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        if checkpoint_path is not None:
            print('Loading ResNet checkpoint from {}'.format(checkpoint_path))
            model_state_dict = torch.load(checkpoint_path, map_location=device)
            model_state_dict = {
                key: value for key, value in model_state_dict.items()
                if 'GU_' not in key and 'text_pooling' not in key}
            model_state_dict = {
                key: value for key, value in model_state_dict.items()
                if 'fc.' not in key}
            model_state_dict = {
                key.replace('resnet.', ''): value
                for key, value in model_state_dict.items()}
            self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(torch.device(device))
        self.model = self.model.eval()
        if share_memory:
            self.model.share_memory()
        self._transform = Transforms.get_transform('default')

    def extract(self, x):
        x = self._transform(x).to(torch.device(self.device))
        return self.model(x)


class RCNN(nn.Module):
    '''
    pretrained FasterRCNN or MaskRCNN from torchvision
    '''
    def __init__(self,
                 archi,
                 device='cuda',
                 checkpoint_path=None,
                 share_memory=False,
                 load_heads=False):
        super().__init__()
        self.device = device
        self.feat_layer = '3'
        if archi == 'maskrcnn':
            self.model = models.detection.maskrcnn_resnet50_fpn(
                pretrained=(checkpoint_path is None),
                pretrained_backbone=(checkpoint_path is None),
                min_size=800)
        elif archi == 'fasterrcnn':
            self.model = models.detection.fasterrcnn_resnet50_fpn(
                pretrained=(checkpoint_path is None),
                pretrained_backbone=(checkpoint_path is None),
                min_size=224)
        else:
            raise ValueError('Unknown model type = {}'.format(archi))

        if archi == 'maskrcnn':
            self._transform = self.model.transform
        else:
            self._transform = Transforms.get_transform('default')
        if not load_heads:
            for attr in ('backbone', 'body'):
                self.model = getattr(self.model, attr)

        if checkpoint_path is not None:
            self.load_from_checkpoint(
                checkpoint_path, load_heads, device, archi, 'backbone.body')
        self.model = self.model.to(torch.device(device))
        self.model = self.model.eval()
        if share_memory:
            self.model.share_memory()


    def extract(self, images):
        if isinstance(
                self._transform, models.detection.transform.GeneralizedRCNNTransform):
            images_normalized = self._transform(
                torch.stack([F.to_tensor(img) for img in images]))[0].tensors
        else:
            images_normalized = torch.stack(
                [self._transform(img) for img in images])
        images_normalized = images_normalized.to(torch.device(self.device))
        model_body = self.model
        if hasattr(self.model, 'backbone'):
            model_body = self.model.backbone.body
        features = model_body(images_normalized)
        return features[self.feat_layer]

    def load_from_checkpoint(self, checkpoint_path, load_heads, device, archi, prefix):
        print('Loading RCNN checkpoint from {}'.format(checkpoint_path))
        state_dict = torch.load(checkpoint_path, map_location=device)
        if not load_heads:
            # load only the backbone
            state_dict = {
                k.replace(prefix + '.', ''): v
                for k, v in state_dict.items() if prefix + '.' in k}
        else:
            # load a full model, replace pre-trained head(s) with (a) new one(s)
            num_classes, in_features = state_dict[
                'roi_heads.box_predictor.cls_score.weight'].shape
            box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes)
            self.model.roi_heads.box_predictor = box_predictor
            if archi == 'maskrcnn':
                # and replace the mask predictor with a new one
                in_features_mask = \
                    self.model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
                    in_features_mask, hidden_layer, num_classes)
                self.model.roi_heads.mask_predictor = mask_predictor
        self.model.load_state_dict(state_dict)

    def predict_objects(self, image, confidence_threshold=0.0, verbose=False):
        image = F.to_tensor(image).to(torch.device(self.device))
        output = self.model(image[None])[0]
        preds = []
        for pred_idx in range(len(output['scores'])):
            score = output['scores'][pred_idx].cpu().item()
            if score < confidence_threshold:
                continue
            box = output['boxes'][pred_idx].cpu().numpy()
            label = self.vocab_pred[output['labels'][pred_idx].cpu().item()]
            if verbose:
                print('{} at {}'.format(label, box))
            pred = types.SimpleNamespace(
                label=label, box=box, score=score)
            if 'masks' in output:
                pred.mask = output['masks'][pred_idx].cpu().numpy()
            preds.append(pred)
        return preds


class FeatureExtractor(nn.Module):
    def __init__(self,
                 archi,
                 device='cuda',
                 checkpoint=None,
                 share_memory=False,
                 compress_type=None,
                 load_heads=False):
        super().__init__()
        self.feat_shape = get_feat_shape(archi, compress_type)
        self.eval_mode = True
        if archi == 'resnet18':
            assert not load_heads
            self.model = Resnet18(device, checkpoint, share_memory)
        else:
            self.model = RCNN(
                archi, device, checkpoint, share_memory, load_heads=load_heads)
        self.compress_type = compress_type


    def featurize(self, images, batch=32):
        feats = []
        with (torch.set_grad_enabled(False) if not self.model.model.training
              else contextlib.nullcontext()):
            for i in range(0, len(images), batch):
                images_batch = images[i:i+batch]
                feats.append(self.model.extract(images_batch))
        feat = torch.cat(feats, dim=0)
        if self.compress_type is not None:
            feat = feat_compress(feat, self.compress_type)
        assert self.feat_shape[1:] == feat.shape[1:]
        return feat

    def predict_objects(self, image, verbose=False):
        with torch.set_grad_enabled(False):
            pred = self.model.predict_objects(image, verbose=verbose)
        return pred

    def train(self, mode):
        if self.eval_mode:
            return
        for module in self.children():
            module.train(mode)

args_ingredient = Ingredient('args')
ex = Experiment('create_data', ingredients=[args_ingredient])
@args_ingredient.config
def cfg_args():
    # name of the output dataset
    data_output = 'lmdb_2.1.1'
    # where to load the original ALFRED dataset images and jsons from
    data_input = 'generated_2.1.0'
    # whether to overwrite old data in case it exists
    overwrite = False
    # number of processes to run the data processing in (0 for main thread)
    num_workers = 0
    # debug run with only 16 entries
    fast_epoch = False

    # VISUAL FEATURES SETTINGS
    # visual archi (resnet18, fasterrcnn, maskrcnn)
    visual_archi = 'fasterrcnn'
    # where to load a pretrained model from
    visual_checkpoint = None
    # which images to use (by default: RGBs)
    image_folder = 'raw_images'
    # feature compression
    compress_type = '4x'
    # which device to use
    device = 'cuda'

    # LANGUAGE ANNOTATIONS SETTINGS
    # generate dataset with subgoal annotations instead of human annotations
    subgoal_ann = False
    # use an existing vocabulary if specified (None for starting from scratch)
    vocab_path = 'files/base.vocab'



# @ex.automain
def eval(args):
    args = AttrDict(**args)
    if args.data_output is None:
        raise RuntimeError('Please, specify the name of output dataset')
    
    extracted = {}

    data_path = '/gpfs/accounts/chaijy_root/chaijy1/Yichi_Jiahao_shared/teach/teach-dataset/images/'
    save_path = '/gpfs/accounts/chaijy_root/chaijy1/Yichi_Jiahao_shared/teach/teach-dataset/vis_feats/'
    data_type_list=['train','valid_seen','valid_unseen']
    game_id_dict={}
    extractor = FeatureExtractor(args.visual_archi, args.device, args.visual_checkpoint,share_memory=True, compress_type=args.compress_type)
    for data_type in data_type_list:
        feat_save_path = os.path.join(save_path, data_type)
        # print(feat_save_path)
        lmdb_feats = lmdb.open(feat_save_path)
        extracted[data_type] = {}
        with lmdb_feats.begin() as txn_feats:
            image_dir=data_path+data_type
            game_id_list=os.listdir(image_dir)
            for game_id in game_id_list:
                print("game id",game_id)
                time_list_path=image_dir+'/'+game_id
                # print("time_list_path",time_list_path)
                time_list_unsorted=os.listdir(time_list_path)
                time_list=[]
                #sort the file by the time line
                fn_dict={}
                flag=False
                ending_file="end"
                for image_file_name in time_list_unsorted:
                    file_name_split=image_file_name.split('.')
                    if file_name_split[0]=='driver':
                        if len(file_name_split)==5:
                            time_string=file_name_split[2]+'.'+file_name_split[3]
                            fn_dict[image_file_name]= float(time_string)
                        else:
                            flag=True
                            ending_file=image_file_name
                fn_dict=dict(sorted(fn_dict.items(), key=lambda item: item[1]))
                time_list=list(fn_dict.keys())
                if flag:
                    time_list.append(ending_file)
                
                
                for image_file_name in time_list:
                    file_name_split=image_file_name.split('.')
                    if file_name_split[0]=='driver':
                        if len(file_name_split)==5:
                            time_string=file_name_split[2]+'.'+file_name_split[3]
                        else:
                            time_string=file_name_split[2]
                    else:
                        continue
                    try:
                        feat_numpy=np.frombuffer(txn_feats.get(f'{game_id}/{time_string}'.encode()),dtype='float32').reshape(-1,512,7,7)
                    except:
                        break
                    # print(f'{game_id}/{time_string}', feat_numpy.shape)
                    if feat_numpy is not None: 
                        extracted[data_type][f'{game_id}/{time_string}'] = 1
                # print(feat_numpy)
                # break
        for k, v in extracted.items():
            print(k, len(v))
        lmdb_feats.close()
    with open('processed.json', 'w') as f:
        json.dump(extracted, f, indent=2)

@ex.automain
def main(args):
    args = AttrDict(**args)
    if args.data_output is None:
        raise RuntimeError('Please, specify the name of output dataset')
    
    with open('processed.json', 'r') as f:
        processed = json.load(f)
    
    data_path = '/gpfs/accounts/chaijy_root/chaijy1/Yichi_Jiahao_shared/teach/teach-dataset/images/'
    save_path = '/gpfs/accounts/chaijy_root/chaijy1/Yichi_Jiahao_shared/teach/teach-dataset/vis_feats/'
    data_type_list=['train','valid_seen','valid_unseen']
    game_id_dict={}
    extractor = FeatureExtractor(args.visual_archi, args.device, args.visual_checkpoint,share_memory=True, compress_type=args.compress_type)
    for data_type in data_type_list:
        if data_type == 'train':
            size = 200*1024**3
        elif '_seen' in data_type:
            size = 30*1024**3
        else:
            size = 100*1024**3
        feat_save_path = os.path.join(save_path, data_type)
        if not os.path.exists(feat_save_path):
            os.makedirs(feat_save_path)
        lmdb_feats = lmdb.open(feat_save_path, size, writemap=True)
        with lmdb_feats.begin(write=True) as txn_feats:
            image_dir=data_path+data_type
            game_id_list=os.listdir(image_dir)
            print(len(game_id_list))
            for game_id_index in range(len(game_id_list)):
                game_id=game_id_list[game_id_index]
                start=time.time()
                time_list_path=image_dir+'/'+game_id
                # print("time_list_path",time_list_path)
                time_list_unsorted=os.listdir(time_list_path)
                time2index={}
                time_list=[]
                #sort the file by the time line
                fn_dict={}
                flag=False
                ending_file="end"
                for image_file_name in time_list_unsorted:
                    file_name_split=image_file_name.split('.')
                    if file_name_split[0]=='driver':
                        if len(file_name_split)==5:
                            time_string=file_name_split[2]+'.'+file_name_split[3]
                            fn_dict[image_file_name]= float(time_string)
                        else:
                            flag=True
                            ending_file=image_file_name
                fn_dict=dict(sorted(fn_dict.items(), key=lambda item: item[1]))
                time_list=list(fn_dict.keys())
                if flag:
                    time_list.append(ending_file)
                #process
                feat_list=[]
                for image_file_name in time_list:
                    file_name_split=image_file_name.split('.')
                    if file_name_split[0]=='driver':
                        if len(file_name_split)==5:
                            time_string=file_name_split[2]+'.'+file_name_split[3]
                        else:
                            time_string=file_name_split[2]
                    else:
                        continue
                    if f'{game_id}/{time_string}' in processed[data_type]:
                        print( f'{game_id}/{time_string} has been processed')
                        continue
                    img_path=[time_list_path+'/'+image_file_name]
                    images = read_images(img_path)
                    feat = extract_features(images, extractor).numpy().squeeze()
                    txn_feats.put(f'{game_id}/{time_string}'.encode(), feat.tobytes())
                game_id_dict[game_id]=time2index
                
                end=time.time()
                print("{}\t".format(game_id_index),"time:",end-start)
                # break
        lmdb_feats.close()
        print("finish{}".format(data_type))
    
    # with open('/gpfs/accounts/chaijy_root/chaijy1/Yichi_Jiahao_shared/teach/teach-dataset/vis_feats/game_id2time2index_dict.json', 'w') as fp:
    #     json.dump(game_id_dict, fp, indent=4)

