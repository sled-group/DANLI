import os, time
import json
import torch
import numpy as np
from pprint import pprint
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from .base import BaseDataset
from ..utils import data_util


class TeachDataset(BaseDataset):
    def __init__(self, partition, args, load_data=True, vocab_path=None):
        super().__init__(partition, args)
        # configuration
        self.data_path = os.path.join(args.out_data_dir, "encoded")
        self.data_type = args.exp_type
        suffix = "_with_intention" if args.add_intention else ""
        self.data_file = "%s.%s%s.json" % (partition, args.exp_type, suffix)
        self.vis_feat_path = os.path.join(args.vis_feat_dir, partition)
        # with open(os.path.join(args.vis_feat_dir, 'game_id2time2index_dict.json'), 'r') as f:
        #     self.frame_time2id = json.load(f)

        self.vocab_path = (
            os.path.join(args.out_data_dir, "vocab") if not vocab_path else vocab_path
        )
        self.pad_id = 0
        self.obj_none_id = 3

        self.trunc_params = args.input_trunc
        self.future_only = (
            args.exp_type == "edh" and args.edh_loss_type == "future_only"
        )
        self.add_intenttion = args.add_intention
        self.enable_vision = args.enable_vision
        self.enable_lang = args.enable_lang
        self.enable_action_history = args.enable_action_history

        if load_data:
            self.load_data(self.data_path)

        # load vocabularies for input language and output actions
        vocabs = data_util.load_vocab(self.vocab_path)
        self.vocab_in_lang = vocabs["input_vocab_word2id"]

        if self.data_type in {"edh", "tfd", "game"}:
            self.vocab_out_action = vocabs["output_vocab_action_high"]
            self.vocab_out_object = vocabs["output_vocab_object"]
        else:
            assert TypeError("Data type should be 'edh', 'tfd' or 'game'")

        self.vocab_out_intent = (
            vocabs["output_vocab_intent"] if args.add_intention else None
        )
        self.vocab_out_lang = (
            vocabs["input_vocab_word2id"] if args.exp_type == "lm" else None
        )

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

    def tensorize_and_pad_batch(self, batch, inference_mode=False, device=None):
        """
        cast values to torch tensors, put them to the correct device and pad sequences
        """
        # print(os.getpid(), 'tensorize_and_pad_batch')
        # device = torch.device(self.device)

        SKIP_FIELDS = {
            "game_id",
            "edh_file",
            "split",
            "frames",
            "max_intent_length",
            "ordering_step",
        }
        if not self.add_intenttion:
            SKIP_FIELDS = SKIP_FIELDS.union(
                {
                    "intent_done_input",
                    "intent_todo_input",
                    "intent_done_output",
                    "intent_todo_output",
                }
            )
        if not self.enable_lang:
            SKIP_FIELDS.add("dialog_input")
        if not self.enable_action_history:
            SKIP_FIELDS = SKIP_FIELDS.union(
                {"action_input", "arg1_input", "arg2_input"}
            )
        if not self.future_only:
            SKIP_FIELDS.add("pred_start_idx")

        batch_input = {k: [] for k in batch[0] if k not in SKIP_FIELDS}

        MAXLEN_PRED = 3
        if "max_intent_length" in batch[0]:
            MAXLEN_INT = max([i["max_intent_length"] for i in batch])
        MAXLEN_ORDER = 1023

        length_lang, length_traj, length_traj_pred = [], [], []
        for sample in batch:

            dial_len = len(sample["dialog_input"])
            if "dialog" in self.trunc_params and dial_len > self.trunc_params["dialog"]:
                dial_len = self.trunc_params["dialog"]
            length_lang.append(dial_len)

            traj_len = len(sample["action_input"])
            if (
                "trajectory" in self.trunc_params
                and traj_len > self.trunc_params["trajectory"]
            ):
                traj_len = self.trunc_params["trajectory"]

                # when a trajectory is truncated we need to change the prediction start index accordingly
                if "pred_start_idx" in sample:
                    sample["pred_start_idx"] += traj_len - len(sample["action_input"])
                    if sample["pred_start_idx"] < 0:
                        sample["pred_start_idx"] = 0
            length_traj.append(traj_len)
            # length_traj_pred.append(len(sample['action_input']) - sample['pred_start_idx'])
            for k, v in sample.items():
                if k in SKIP_FIELDS:  # skip fiels such as game_id
                    continue

                # Front truncation of dialog/trajectory histories that are too long
                # we do not use post-trunction because prediction positions for EDH
                # are usually at the end of each trajectory
                if "dialog" in k or k == "ordering_tokens":
                    trunc = self.trunc_params.get("dialog", len(v))
                    # v = v[:trunc]
                    if self.args.exp_type == "edh":
                        v = v[-trunc:]  # pre-trunction
                    else:
                        v = v[:trunc]  # post-trunction
                elif (
                    isinstance(v, list)
                    or isinstance(v, np.ndarray)
                    or isinstance(v, torch.Tensor)
                ):
                    trunc = self.trunc_params.get("trajectory", len(v))
                    v = v[-trunc:]  # pre-trunction

                # Each predicate (action, object) input is represented as a natural
                # language phrase. Pad them to the same length MAXLEN_PRED (=3).
                if k in {"action_input", "arg1_input", "arg2_input"}:
                    v = [vv + [0] * (MAXLEN_PRED - len(vv)) for vv in v]

                # Each intention is represented as a natural language sentence.
                # Pad them to the same length MAXLEN_INT.
                elif k in {
                    "intent_done_input",
                    "intent_todo_input",
                    "intent_done_output",
                    "intent_todo_output",
                }:
                    v = [vv + [0] * (MAXLEN_INT - len(vv)) for vv in v]

                # tensorize
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                    if k != "vis_feats":
                        v = v.long()
                batch_input[k].append(v)
        batch_input["lengths_lang"] = torch.tensor(length_lang)
        batch_input["lengths_traj"] = torch.tensor(length_traj)
        # Note: len(frames) == len(actions) == len(intents)

        # pack batch
        for k, v in batch_input.items():
            if "lengths" in k:
                pass
            elif k == "pred_start_idx":
                batch_input[k] = torch.stack(batch_input[k])
            # elif k == 'vis_feats':
            #     t = time.time()
            #     vis_feats = torch.zeros(len(batch), batch_input['lengths_traj'].max(), 512, 7, 7).float()
            #     for b in range(len(batch)):
            #         vis_feats[b][:len(batch_input[k][b])] = batch_input[k][b]
            #     batch_input[k] = vis_feats
            #     print('%s time:'%k, time.time()-t)
            else:
                batch_input[k] = pad_sequence(
                    v, batch_first=True, padding_value=self.pad_id
                )
                # if k == 'vis_feats':

                # else:
                #     batch_input[k] = pad_sequence(v, batch_first=True, padding_value=self.pad_id)

            # batch_input[k] = batch_input[k].to(device, non_blocking=True)

        # during inference we do not have labels, so return here
        if inference_mode:
            if device is not None:
                for k, v in batch_input.items():
                    if isinstance(v, torch.Tensor):
                        batch_input[k] = v.to(device=device)
            return batch_input

        # Given the fact that:
        # 1. Do not need to make predictions for padding positions
        # 2. Only the `Goto` action has the 2nd object argument indicating "where to go"
        # 3. In EDH `future_only` setting, do not need to predict history actions
        # We only collect necessary positions for faster computation
        pred_idx = batch_input["action_output"] != self.pad_id  # non-padding positions
        goto_idx = batch_input["arg2_output"] != self.obj_none_id
        # print('non padding idx:', pred_idx)
        if self.future_only:
            # future action positions for edh
            batch_input["pred_mask_edh"] = torch.zeros_like(
                batch_input["action_output"]
            ).bool()
            for idx, start in enumerate(batch_input["pred_start_idx"]):
                end = batch_input["lengths_traj"][idx]
                batch_input["pred_mask_edh"][idx][start:end] = 1
            pred_idx = pred_idx & batch_input["pred_mask_edh"]
            # print('non padding and future action idx:', pred_idx)
        batch_input["pred_idx_traj"] = pred_idx
        batch_input["pred_idx_arg2"] = pred_idx & goto_idx

        # print(batch_input['pred_idx_traj'].shape, batch_input['pred_idx_traj'].sum(),
        #       len(batch_input['pred_idx_traj'].view(-1)))

        # pprint(batch_input['ordering_dialog'])
        # pprint(batch_input['ordering_action'])

        # gather the output in the corresponding positions
        for dn in batch_input:
            # print('before:', dn, batch_input[dn].shape)
            if "output" in dn and dn not in {"dialog_output", "arg2_output"}:
                batch_input[dn] = batch_input[dn][batch_input["pred_idx_traj"]]
            elif dn == "arg2_output":
                batch_input[dn] = batch_input[dn][batch_input["pred_idx_arg2"]]
            # print('after:', dn, batch_input[dn].shape)
        # print(os.getpid(), 'all done')
        batch_input["game_id"] = [i["game_id"] for i in batch]
        if "edh_file" in batch[0]:
            batch_input["edh_file"] = [i["edh_file"] for i in batch]

        if device is not None:
            for k, v in batch_input.items():
                if isinstance(v, torch.Tensor):
                    batch_input[k] = v.to(device=device)
        # for k, v in batch_input.items():
        #     if hasattr(v, 'shape'):
        #     # if k == 'intent_todo_output':
        #         print(k, v.shape)
        return batch_input
