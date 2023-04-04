import os
import json
import math
import argparse
import torch
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import AutoTokenizer

from model.data.subgoal_seq2seq import SubgoalDataset
from model.model.model_plm import PLModelWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a pretrained transformers model for seq2seq subgoal prediction"
    )

    parser.add_argument(
        "--plm_model_name",
        type=str,
        default="facebook/bart-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.environ["DANLI_DATA_DIR"], "processed_20220610"),
        help="Directory of where training data is stored.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--max_encoder_seq_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_decoder_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--num_loader_workers",
        type=int,
        default=0,
        help="The number of processes to use for data loading.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=20,
        help="Batch size (per device) for the dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=20,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--seed", type=int, default=325, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If passed, run in debug mode with 2 batches of each dataset.",
    )

    parser.add_argument(
        "--use_symbol_embedding_as_input",
        action="store_true",
        help="If passed, use symbol embeddings as decoder inputs instead of translating them "
        "into language form.",
    )

    parser.add_argument(
        "--tie_encoder_decoder_embeddings",
        action="store_true",
        help="If passed, use symbol embeddings as decoder inputs instead of translating them "
        "into language form.",
    )

    parser.add_argument(
        "--encoder_input_type",
        type=str,
        default="text_dialog_and_act",
        help="Specify the context to use as encoder input. ",
        choices=[
            "text_dialog",
            "text_dialog_and_act",
            "text_dialog_and_sg",
            "text_dialog_and_act_and_sg",
        ],
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sg_todo_all",
        help="Specify the decoder prediction type.",
        choices=[
            "sg_done",
            "sg_todo_edh",
            "sg_todo_all",
            "sg_done_and_todo_edh",
            "sg_done_and_todo_all",
        ],
    )
    parser.add_argument(
        "--loss_weight_pred",
        type=float,
        default=1.0,
        help="Loss weight for predicate prediction.",
    )
    parser.add_argument(
        "--loss_weight_subj",
        type=float,
        default=1.0,
        help="Loss weight for predicate prediction.",
    )
    parser.add_argument(
        "--loss_weight_obj",
        type=float,
        default=1.0,
        help="Loss weight for predicate prediction.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    args.plm_training = True
    gpu_num = torch.cuda.device_count()

    if not args.output_dir:
        start_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        exp_dir_base = os.path.join(os.environ["DANLI_DATA_DIR"], "experiments")
        exp_name = "{}_{}_{}_{}_{}".format(
            start_time,
            args.plm_model_name.split("/")[-1],
            args.encoder_input_type,
            args.prediction_type,
            args.learning_rate,
        )
        exp_dir = os.path.join(exp_dir_base, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            print("Create output directory: {}".format(exp_dir))
        args.output_dir = exp_dir

    pl.seed_everything(args.seed)

    print("Training args:")
    print(args)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.plm_model_name)
    tokenizer.save_pretrained(os.path.join(exp_dir, "tokenizer/"))

    print("Create datasets and loaders...")
    partitions = ["train", "valid_seen", "valid_unseen"]
    datasets, loaders = {}, {}
    datasets["train"] = SubgoalDataset(tokenizer, args=args, split=["train"])
    datasets["valid_seen"] = SubgoalDataset(tokenizer, args=args, split=["valid_seen"])
    datasets["valid_unseen"] = SubgoalDataset(
        tokenizer, args=args, split=["valid_unseen"]
    )
    for split in partitions:
        loaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=args.per_device_batch_size,
            num_workers=args.num_loader_workers,
            collate_fn=datasets[split].data_collect,
            pin_memory=True,
        )

    num_update_steps_per_epoch = math.ceil(
        len(loaders["train"]) / (args.gradient_accumulation_steps * gpu_num)
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Print number of batches in each split.
    for split, loader in loaders.items():
        print("%s has %d batches" % (split, len(loader)))

    print("Create model...")
    model = PLModelWrapper(args, tokenizer)

    checkpoint_callback = ModelCheckpoint(
        monitor="avg_accu",
        dirpath=args.output_dir,
        filename="ckpt-{epoch:02d}-{avg_accu:.4f}",
        save_top_k=2,
        mode="max",
    )

    # pl trainer
    print("Create pl trainer...")
    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        strategy="ddp" if gpu_num > 1 else None,
        gpus=-1,
        gradient_clip_val=1.0,
        # check_val_every_n_epoch=1,
        val_check_interval=0.33,
        detect_anomaly=True,
        logger=None,
        log_every_n_steps=10,
        weights_save_path=args.output_dir,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.gradient_accumulation_steps,
        # logger=None,
        # log_every_n_steps=None,
        # resume_from_checkpoint=None
    )

    print("Begin traning...")
    trainer.fit(
        model, loaders["train"], [loaders["valid_seen"], loaders["valid_unseen"]]
    )
