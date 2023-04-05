import os
import random
import argparse
import json
import torch
import pytorch_lightning as pl
from attrdict import AttrDict
from pprint import pprint
from transformers import AutoTokenizer
from model.data.subgoal_seq2seq import SubgoalDataset
from model.model.model_plm import PLModelWrapper
from model.utils.data_util import process_edh_for_subgoal_prediction


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subgoal_predictor_dir",
        type=str,
        default=os.path.join(os.environ["DANLI_MODEL_DIR"], "subgoal_predictor"),
    )
    parser.add_argument("--subgoal_predictor_ckpt", type=str, default="ckpt.pth")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    eval_args = parse_args()
    model_args = AttrDict(
        json.load(open(os.path.join(eval_args.subgoal_predictor_dir, "args.json")))
    )
    model_args["data_dir"] = os.path.join(
        os.environ["DANLI_DATA_DIR"], "processed_20220610"
    )
    gpu_num = torch.cuda.device_count()

    pl.seed_everything(19950325)

    print("Model args:")
    print(model_args)
    print("Evaluation args:")
    print(eval_args)

    print("Loading tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(model_args.plm_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(eval_args.subgoal_predictor_dir, "tokenizer/")
    )

    # load dataset(s)
    print("Create datasets and loaders...")
    partitions = ["valid_seen"]
    datasets, loaders = {}, {}
    for split in partitions:
        datasets[split] = SubgoalDataset(tokenizer, args=model_args, split=[split])
        loaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=1,
            num_workers=0,
            collate_fn=datasets[split].data_collect,
            pin_memory=True,
        )
    # datasets["train"] = SubgoalDataset(tokenizer, args=model_args, split=None)

    device = "cuda:0"
    print("Create model...")
    model_ckpt = os.path.join(
        eval_args.subgoal_predictor_dir, eval_args.subgoal_predictor_ckpt
    )
    model_args.device = device
    model = PLModelWrapper.load_from_checkpoint(
        model_ckpt, args=model_args, tokenizer=tokenizer
    )
    model = model.to(device=device)
    model.eval()
    model.freeze()

    data_path = os.path.join(os.environ["DANLI_DATA_DIR"], "processed_20220610")
    raw_edh_path = os.path.join(
        os.environ["DANLI_DATA_DIR"], "edh_instances/valid_seen"
    )
    for edh_fn in random.sample(os.listdir(raw_edh_path), 10):
        raw_edh = json.load(open(os.path.join(raw_edh_path, edh_fn)))

        print("processing batch")
        edh, _ = process_edh_for_subgoal_prediction(raw_edh)
        batch = datasets["valid_seen"].data_collect(
            [edh], inference_mode=True, device=device
        )

        print("-" * 30, "encoder input", "-" * 30)
        pprint(tokenizer.batch_decode(batch["encoder_input_ids"]))
        print("")

        predictions = model.predict(batch, max_step=64)
        print("-" * 30, "predictions", "-" * 30)
        pprint(predictions)
        print("")

        print("-" * 30, "golden future actions", "-" * 30)
        pprint(raw_edh["future_subgoals"])
