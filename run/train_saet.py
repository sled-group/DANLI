import os
import json
import inspect
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from attrdict import AttrDict

from config import data_ingredient, exp_ingredient
from config import model_ingredient, train_ingredient

from model.train import train
from model.model import controller
from model.model import model as pl_model
from model.nn import encoders, decoders
from model.data import teach

ingredients = [
    data_ingredient,
    exp_ingredient,
    model_ingredient,
    train_ingredient,
]
ex = Experiment("train", ingredients=ingredients)

src_montior_list = [train, controller, pl_model, encoders, decoders, teach]
for src in src_montior_list:
    ex.add_source_file(inspect.getfile(src))
ex.observers.append(FileStorageObserver("exp_runs"))


@ex.automain
def main(data_args, exp_args, model_args, train_args):
    """
    train a network using an lmdb dataset
    """
    core_mask_op = "taskset -pc %s %d" % ("0-40", os.getpid())
    os.system(core_mask_op)
    # parse args
    args = AttrDict(**data_args, **exp_args, **train_args, **model_args)

    if args.add_intention:
        args.exp_dir += "-intent"

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
        os.makedirs(os.path.join(args.exp_dir, "vocab"))
    with open(os.path.join(args.exp_dir, "config.json"), "w") as f:
        json.dump(dict(args), f, sort_keys=False, indent=4)

    train.main(args)
