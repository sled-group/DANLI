import os, json
import torch
import random
import shutil
import pprint
import numpy as np
from sacred import Experiment
from attrdict import AttrDict

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from ..data.teach import TeachDataset
from ..data.navigation import NaviDataset
from ..model.model import PLModelWrapper
from ..utils import data_util, model_util, data_sampler


def main(args):
    """
    train a network using an lmdb dataset
    """

    # set random seed
    pl.seed_everything(args.seed)

    # load dataset(s) and process vocabs
    print("create datasets")
    partitions = ["train", "valid_seen", "valid_unseen"]
    if args.exp_type in ["edh", "tfd", "game"]:
        datasets = {split: TeachDataset(split, args) for split in partitions}
    elif args.exp_type == "navi":
        datasets = {split: NaviDataset(split, args) for split in partitions}

    print("create bucket samplers and data loaders")
    samplers, loaders = {}, {}
    for split, dataset in datasets.items():
        samplers[split] = data_sampler.BucketBatchSampler(
            dataset, batch_size=args.batch_size, shuffle=True
        )

        # samplers[split] = data_sampler.DistributedBucketBatchSampler(dataset,
        #                                                              batch_size=args.batch_size,
        #                                                              num_replicas=num_workers,
        #                                                              sort_key=lambda x: len(x['action_input']),
        #                                                              shuffle=True)

        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            #  batch_size=args.batch_size,
            batch_sampler=samplers[split],
            num_workers=args.num_loader_workers,
            collate_fn=dataset.tensorize_and_pad_batch,
            pin_memory=True,
        )

    # Print number of batches in each split.
    # for split, loader in loaders.items():
    #     print('%s has %d batches' % (split, len(loader)))

    print("Loading vocabularies")
    vocab_dir = os.path.join(args.out_data_dir, "vocab")
    vocabs = data_util.load_vocab(vocab_dir)
    for k, v in vocabs.items():
        with open(os.path.join(args.exp_dir, "vocab", "%s.json" % k), "w") as f:
            json.dump(v, f, sort_keys=False, indent=4)

    print("create model")
    model_cls = {
        "edh": "controller",
        "tfd": "controller",
        "game": "controller",
        "lm": "speaker",
        "navi": "navigator",
    }[args.exp_type]
    args.batch_num_train = len(loaders["train"])
    model = PLModelWrapper(model_cls, args, vocabs)

    # create neuptune logger
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Y2MxNzFhNS01ZWIxLTRkZmYtODA0NS05ZDM0ZTljMGIyMDkifQ==",  # replace with your own
        project="594zyc/TEACh-Mental",
        log_model_checkpoints=False,
        # prefix=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )
    neptune_logger.log_hyperparams(params=args)
    neptune_logger.log_model_summary(model=model, max_depth=-1)
    # neptune_logger= None

    checkpoint_callback = ModelCheckpoint(
        monitor="avg_accu",
        dirpath=args.exp_dir,
        filename="ckpt-{epoch:02d}-{avg_accu:.4f}",
        save_top_k=3,
        mode="max",
    )

    # pl trainer
    print("begin traning")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu" if args.device == "cpu" else "gpu",
        gpus=-1 if args.device == "cuda" else None,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        detect_anomaly=True,
        logger=neptune_logger,
        log_every_n_steps=10,
        weights_save_path=args.exp_dir,
        callbacks=[checkpoint_callback],
        # logger=None,
        # log_every_n_steps=None,
        # resume_from_checkpoint=None
    )
    # trainer = pl.Trainer(gpus=-1,
    #                      strategy='ddp',
    #                      precision=32,
    #                      max_epochs=args.epochs,
    #                      # logger=None,
    #                      # log_every_n_steps=None,
    #                      # resume_from_checkpoint=None
    #                      )
    trainer.fit(
        model, loaders["train"], [loaders["valid_seen"], loaders["valid_unseen"]]
    )
