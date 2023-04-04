import os, time
import json
import torch
import numpy as np
from pprint import pprint
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from .base import BaseDataset
from ..utils import data_util


class NaviDataset(BaseDataset):
    def __init__(self, partition, args, load_data=True, vocab_path=None):
        super().__init__(partition, args)
        # configuration
        self.data_path = os.path.join(args.out_data_dir, "encoded")
        self.data_type = args.exp_type
        self.vis_feat_path = os.path.join(args.vis_feat_dir, partition)
        assert self.data_type == "navi"
        self.data_file = "%s.%s.json" % (partition, args.exp_type)

        self.vocab_path = (
            os.path.join(args.out_data_dir, "vocab") if not vocab_path else vocab_path
        )
        self.pad_id = 0

        self.trunc_params = args.input_trunc
        self.enable_vision = args.enable_vision
        self.enable_action_history = args.enable_action_history
        self.action_history_length = args.action_history_length

        if load_data:
            self.load_data(self.data_path)

        # load vocabularies for input language and output actions
        vocabs = data_util.load_vocab(self.vocab_path)
        self.vocab_in_obj = vocabs["output_vocab_object"]
        self.vocab_out_action = vocabs["output_vocab_action_navi"]

    def load_data(self, json_path):
        super().load_data(json_path)
        self.traj_lengths = [len(d["action_input"]) for d in self.data]
        print("data loaded")

    def __getitem__(self, idx):
        if not hasattr(self, "data"):
            raise AttributeError("Data are not loaded")
        feat_dict = self.data[idx]
        if self.enable_vision:
            feat_dict["vis_feats"] = self.load_frames(
                feat_dict["game_id"], feat_dict["frames"]
            )
            assert len(feat_dict["vis_feats"]) == len(feat_dict["action_input"])
        return feat_dict

    def tensorize_and_pad_batch(self, batch, device=None):
        """
        cast values to torch tensors, put them to the correct device and pad sequences
        """
        # print(os.getpid(), 'tensorize_and_pad_batch')
        # device = torch.device(self.device)

        tensorize_field = ["goal", "action_output"]
        batch_input = {k: [] for k in tensorize_field}

        if self.enable_vision:
            tensorize_field.append("vis_feats")
            batch_input.update({"vis_feats": [], "ordering_frames": []})

        if self.enable_action_history:
            tensorize_field.append("action_input")
            batch_input.update({"action_input": [], "ordering_action": []})

        length_vis, length_act = [], []
        for sample in batch:
            traj_len = len(sample["action_input"])
            if (
                "trajectory" in self.trunc_params
                and traj_len > self.trunc_params["trajectory"]
            ):
                traj_len = self.trunc_params["trajectory"]
            
            if self.enable_vision:
                batch_input["ordering_frames"].append(torch.arange(0, traj_len))
                length_vis.append(traj_len)

            if self.enable_action_history:
                ord_action = torch.arange(0, traj_len)
                if self.action_history_length != "all":
                    ord_action = ord_action[-self.action_history_length :]
                batch_input["ordering_action"].append(ord_action)
                if self.action_history_length == "all":
                    length_act.append(traj_len)
                else:
                    length_act.append(self.action_history_length)

            for k in tensorize_field:
                if k not in sample:
                    continue
                v = sample[k]
                if k != "goal":
                    trunc = self.trunc_params.get("trajectory", len(v))
                    if k == "action_input" and self.action_history_length != "all":
                        trunc = min(trunc, self.action_history_length)
                    v = v[-trunc:]  # pre-trunction
                # tensorize
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                batch_input[k].append(v)

        if self.enable_vision:
            batch_input["lengths_vis"] = torch.tensor(length_vis)
        if self.enable_action_history:
            batch_input["lengths_action"] = torch.tensor(length_act)

        for k in list(batch_input.keys()):
            if batch_input[k] == []:
                del batch_input[k]

        # pack batch
        for k, v in batch_input.items():
            if "lengths" in k:
                pass
            else:
                batch_input[k] = pad_sequence(
                    v, batch_first=True, padding_value=self.pad_id
                )
        batch_input["ordering_goal"] = torch.zeros((len(batch), 1)).long()
        batch_input["lengths_goal"] = torch.ones(len(batch)).long() * 1

        if "game_id" in batch_input:
            batch_input["game_id"] = [i["game_id"] for i in batch]
        if "hidx" in batch_input:
            batch_input["hidx"] = [i["hidx"] for i in batch]

        # for k, v in batch_input.items():
        # # if hasattr(v, 'shape'):
        # if k == 'intent_todo_output':
        #     print(k, v)
        if device is not None:
            for k, v in batch_input.items():
                if isinstance(v, torch.Tensor):
                    batch_input[k] = v.to(device=device)
        return batch_input
