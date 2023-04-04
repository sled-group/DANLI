import torch
from torch import nn
from torch.nn import functional as F

from ..model import base
from ..nn.encoders import EncoderNaviGoal, EncoderVision, EncoderAction
from ..nn.encoders import MultiModalTransformer


class Model(base.Model):
    def __init__(self, args, vocabs):
        """
        transformer model of high-level planning
        """
        super().__init__(args, vocabs)

        # get input/output sizes
        self.obj_num = len(vocabs["input_vocab_word2id"])
        self.act_num = len(vocabs["output_vocab_action_navi"])
        self.output_vocab = {
            v: k for k, v in vocabs["output_vocab_action_navi"].items()
        }

        # create goal input embedding
        self.goal_embedder = nn.Embedding(self.obj_num, args.demb, padding_idx=0)
        nn.init.xavier_uniform_(self.goal_embedder.weight, gain=1.0)
        # create action input embedding
        self.action_embedder = nn.Embedding(self.act_num, args.demb, padding_idx=0)
        nn.init.xavier_uniform_(self.action_embedder.weight, gain=1.0)

        # create input processing modules
        self.enc_goal = EncoderNaviGoal(args, self.goal_embedder)
        if args.enable_vision:
            self.enc_vis = EncoderVision(args)
        if args.enable_action_history:
            self.enc_action = EncoderAction(args, self.action_embedder)

        # multi-modal transformer encoder
        self.enc_mm = MultiModalTransformer(args)

        # # initialize internal states (used for interactive evaluation)
        # self.reset()

    def forward(self, inputs, mode="training"):
        """
        forward the model for multiple time-steps (used for training)
        """

        # embed each modality
        input_embs, input_lens, input_ords, output_idx = [], [], [], []
        assert "goal" in inputs
        emb_goal = self.enc_goal(goal_input=inputs["goal"])
        input_embs.append(("lang", emb_goal))  # reuse 'lang' as the modality name
        input_lens.append(inputs["lengths_goal"])
        input_ords.append(inputs["ordering_goal"])
        output_idx.append(emb_goal.shape[1])

        assert "vis_feats" in inputs or "action_input" in inputs
        if "vis_feats" in inputs:
            emb_vis = self.enc_vis(inputs["vis_feats"])
            input_embs.append(("vis", emb_vis))
            input_lens.append(inputs["lengths_vis"])
            input_ords.append(inputs["ordering_frames"])
            output_idx.append(emb_vis.shape[1])

        if "action_input" in inputs:
            emb_action = self.enc_action(
                action_input=inputs["action_input"].unsqueeze(-1)
            )
            input_embs.append(("action", emb_action))
            input_lens.append(inputs["lengths_action"])
            input_ords.append(inputs["ordering_action"])
            output_idx.append(emb_action.shape[1])

        # if "vis_feats" in inputs and "action_input" in inputs:
        #     assert emb_vis.shape == emb_action.shape, (emb_vis.shape, emb_action.shape)

        # print([a[0] for a in input_embs])
        # print(input_lens)
        # encode
        enc_mm_outputs, mask_pad = self.enc_mm(
            input_embeddings=input_embs,
            input_lengths=input_lens,
            input_orderings=input_ords,
            inputs=inputs,
        )

        # gather encoder outputs
        offset = 0
        enc_out = {}
        for idx, (modality, _) in enumerate(input_embs):
            enc_out[modality] = enc_mm_outputs[:, offset : offset + output_idx[idx]]
            offset += output_idx[idx]

        # decide inputs to each decoders
        dec_inp_action = enc_out["vis"] if "vis" in enc_out else enc_out["action"]

        # decode
        logits = {}

        # training/validation mode: compute losses
        if mode in {"training", "validation"}:
            # compute logits (B, T_traj, H) -> (B, T_traj, V)
            logits["action_output"] = dec_inp_action.matmul(
                self.action_embedder.weight.t()
            )
            return self.compute_loss_and_accuracy(logits, inputs)

        # inference mode: return prediction probabilities of the latest step
        elif mode == "inference":  # TODO
            # for inference we only predict the next action
            assert dec_inp_action.shape[0] == 1
            dec_inp_action = dec_inp_action[:, -1].unsqueeze(1)
            logits = dec_inp_action.matmul(self.action_embedder.weight.t())
            probs = F.softmax(logits.squeeze(), dim=0)

            k = min(self.args.dec_params["keep_topk_prob"], len(probs))
            probs_sorted, idx_sorted = probs.topk(k)
            pred_strs = [self.output_vocab[pid.item()] for pid in idx_sorted]
            predictions = list(
                zip(pred_strs, idx_sorted.tolist(), probs_sorted.tolist())
            )

            return predictions

        else:
            raise ValueError("mode must be either `training` or `inference`")

    def get_probs(self, logits, dec_params):
        """
        get prediction probabilities
        """
        probs = {}
        return probs

    def prediction_id_to_string(self, prediction_ids, output_type):
        v = self.output_vocabs[output_type]
        return [v[pid.item()] for pid in prediction_ids]

    def compute_loss_and_accuracy(self, logits_dict, labels_dict):
        """
        comptute the loss for all the predictions
        """
        losses, accuracies = {}, {}
        for pred_type, logits in logits_dict.items():
            logits = logits.view(-1, logits.shape[-1])
            label = labels_dict[pred_type].view(-1)
            losses[pred_type] = F.cross_entropy(
                logits, label, ignore_index=0, reduction="mean"  # padding index
            )

            _, pred = logits.topk(1, dim=1)
            accuracies[pred_type] = (pred.squeeze() == label).sum() / len(label)

        return losses, accuracies

    def step(self, inputs, new_input, data_handler, visfeat_extractor):
        """
        return single-step predictions during inference
        """

        # add visual feature
        if data_handler.enable_vision:
            if "vis_feats" not in inputs:  # start a new navigation sub-procedure
                inputs["vis_feats"] = visfeat_extractor.featurize(
                    [new_input["frame"]], batch_size=1
                )
            else:  # resume navigating
                inputs["vis_feats"] = visfeat_extractor.append_feature(
                    new_input["frame"], inputs["vis_feats"]
                )

        # add last action
        if data_handler.enable_action_history:
            if "action" in new_input:
                inputs["action_input"].append(new_input["action"])

        # tensorize
        inputs_t = data_handler.tensorize_and_pad_batch(
            [inputs], device=self.goal_embedder.weight.device
        )

        # predict
        predictions = self(inputs_t, mode="inference")

        return predictions, inputs
