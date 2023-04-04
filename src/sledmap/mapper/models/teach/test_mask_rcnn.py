import os, json
import torch
from torch import nn
from matplotlib import pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from definitions.teach_objects import (
    THING_NAMES, 
    STUFF_NAMES, 
    get_object_affordance,
    get_object_receptacle_compatibility, 
    ObjectClass
)

class MaskRCNNDetector(nn.Module):
	def __init__(self, ckpt_file, device="cuda", conf_thresh=0.7):
		super().__init__()
		self.ckpt_file = ckpt_file
		self.confidence_thresh = conf_thresh if conf_thresh is not None else 0.5
		self.device = device
		self.objects = ['AlarmClock', 'AluminumFoil', 'Apple', 'AppleSliced', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bottle', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Cabinet', 'Candle', 'CellPhone', 'Chair', 'Cloth', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Desk', 'DeskLamp', 'DiningTable', 'DishSponge', 'Drawer', 'Dresser', 'Dumbbell', 'Egg', 'EggCracked', 'Faucet', 'FloorLamp', 'Fork', 'Fridge', 'GarbageCan', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Microwave', 'Mug', 'Newspaper', 'Ottoman', 'Pan', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerCurtain', 'ShowerDoor', 'ShowerHead', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'Stool', 'StoveBurner', 'StoveKnob', 'TVStand', 'TableTopDecor', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'Tomato', 'TomatoSliced', 'Towel', 'TowelHolder', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
		self.config_maskrcnn()
		self.predictor = DefaultPredictor(self.cfg)

	def config_maskrcnn(self):
		cfg = get_cfg()
		cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
		# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
		cfg.MODEL.WEIGHTS = self.ckpt_file
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
			self.confidence_thresh
		)  # set a custom testing threshold
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.objects)   
		cfg.INPUT.MASK_FORMAT = "bitmask"
		cfg.MODEL.DEVICE = self.device
		self.cfg = cfg

	def get_predictions(self, frame):
		"""perform object detection and instance segmentation on a single frame
		:param frame: an image (numpy array) in RGB
		:return: a dict with data defined following "instances" class defined 
				in detectron2. for more details please refer to
				 https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
		"""
		# frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
		instances = self.predictor(frame)["instances"].get_fields()
		classes = instances['pred_classes']
		names = [self.objects[i] for i in classes]
		seagull_ids = torch.tensor([ObjectClass[n].value for n in names]).long().to(classes.device)
		instances['pred_classes'] = seagull_ids-1
		return instances