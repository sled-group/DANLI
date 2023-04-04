import pprint
import torch
from torch import nn
from torch.nn import functional as F

from ..model import base
from ..nn.encoders import EncoderDialog, EncoderVision, EncoderAction
from ..nn.encoders import EncoderIntent, MultiModalTransformer
from ..nn.decoders import DecoderIntent
from ..utils import model_util


class Model(base.Model):
    def __init__(self, args, vocabs):
        """
        transformer model of high-level planning
        """
        super().__init__(args, vocabs)

        self.args = args
        self.vocabs = vocabs
        if hasattr(args, "dec_params"):
            self.dec_params = args.dec_params

        # get input/output sizes
        self.input_size = len(vocabs["input_vocab_word2id"])
        self.output_size_action = len(vocabs["output_vocab_action_high"])
        self.output_size_object = len(vocabs["output_vocab_object"])
        self.output_vocabs = {
            "action_output": {
                v: k for k, v in vocabs["output_vocab_action_high"].items()
            },
            "arg1_output": {v: k for k, v in vocabs["output_vocab_object"].items()},
            "arg2_output": {v: k for k, v in vocabs["output_vocab_object"].items()},
        }

        self.add_intention = args.add_intention
        if self.add_intention:
            self.output_size_intent = len(vocabs["output_vocab_intent"])
            self.output_vocabs.update(
                {
                    "intent_done_output": {
                        v: k for k, v in vocabs["output_vocab_intent"].items()
                    },
                    "intent_todo_output": {
                        v: k for k, v in vocabs["output_vocab_intent"].items()
                    },
                }
            )

        # create language input embedding
        self.input_embedder = nn.Embedding(self.input_size, args.demb, padding_idx=0)
        nn.init.xavier_uniform_(self.input_embedder.weight, gain=1.0)

        # create input processing modules
        self.enc_lang = EncoderDialog(args, self.input_embedder)
        self.enc_vis = EncoderVision(args)
        self.enc_action = EncoderAction(args, self.input_embedder)
        self.enc_intent = EncoderIntent(args, self.input_embedder)

        # multi-modal transformer encoder
        self.enc_mm = MultiModalTransformer(args)

        # decoders
        self.dec_action = nn.Linear(args.demb, self.output_size_action)
        self.dec_arg1 = nn.Linear(args.demb, self.output_size_object)
        self.dec_arg2 = nn.Linear(args.demb, self.output_size_object)
        if args.add_intention:
            self.dec_intent = DecoderIntent(
                args, self.output_size_intent, self.input_embedder
            )

        # # initialize internal states (used for interactive evaluation)
        # self.reset()

    def forward(self, inputs, mode="training"):
        """
        forward the model for multiple time-steps (used for training)
        """

        # embed each modality
        input_embs, input_lens, input_ords, output_idx = [], [], [], []
        if "dialog_input" in inputs:
            emb_lang = self.enc_lang(
                lang_input=inputs["dialog_input"],
                tok_pos_input=inputs["ordering_tokens"],
                role_input=inputs["dialog_role"],
            )
            input_embs.append(("lang", emb_lang))
            input_lens.append(inputs["lengths_lang"])
            input_ords.append(inputs["ordering_dialog"])
            output_idx.append(emb_lang.shape[1])

        if "vis_feats" in inputs:
            emb_vis = self.enc_vis(inputs["vis_feats"])
            input_embs.append(("vis", emb_vis))
            input_lens.append(inputs["lengths_traj"])
            input_ords.append(inputs["ordering_frames"])
            output_idx.append(emb_vis.shape[1])

        if "action_input" in inputs:
            emb_action = self.enc_action(
                action_input=inputs["action_input"],
                arg1_input=inputs["arg1_input"],
                # arg2_input=inputs["arg2_input"],
            )
            input_embs.append(("action", emb_action))
            input_lens.append(inputs["lengths_traj"])
            input_ords.append(inputs["ordering_action"])
            output_idx.append(emb_action.shape[1])

        if self.add_intention and "intent_done_input" in inputs:
            emb_intent = self.enc_intent(
                intent_done=inputs["intent_done_input"],
                intent_todo=inputs["intent_todo_input"],
            )
            input_embs.append(("intent", emb_intent))
            if mode != "inference_intent":
                input_lens.append(inputs["lengths_traj"])
            else:
                input_lens.append(
                    torch.tensor([emb_intent.shape[1]], device=emb_intent.device)
                )
            input_ords.append(inputs["ordering_intent"])
            output_idx.append(emb_intent.shape[1])

        if "vis_feats" in inputs and "action_input" in inputs and self.add_intention:
            assert emb_vis.shape == emb_action.shape, (emb_vis.shape, emb_action.shape)
            if mode != "inference_intent":
                assert emb_action.shape == emb_intent.shape, (
                    emb_intent.shape,
                    emb_action.shape,
                )

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
        if self.add_intention:
            if mode in {"training", "validation", "inference_action"}:
                dec_inp_action = enc_out["intent"]
            if mode in {"training", "validation", "inference_intent"}:
                dec_inp_intent = (
                    enc_out["action"] if "action" in enc_out else enc_out["vis"]
                )
        else:
            dec_inp_action = (
                enc_out["action"] if "action" in enc_out else enc_out["vis"]
            )

        # decode
        logits = {}

        # training/validation mode: compute losses
        if mode in {"training", "validation"}:

            # (B, T_traj) -> (N,)
            logits["action_output"] = self.dec_action(
                dec_inp_action[inputs["pred_idx_traj"]]
            )
            logits["arg1_output"] = self.dec_arg1(
                dec_inp_action[inputs["pred_idx_traj"]]
            )
            logits["arg2_output"] = self.dec_arg2(
                dec_inp_action[inputs["pred_idx_arg2"]]
            )

            if self.add_intention:
                # (B, T_traj, T_dec) -> (N, T_dec)
                input_idx_done = inputs["intent_done_input"][inputs["pred_idx_traj"]]
                input_idx_todo = inputs["intent_todo_input"][inputs["pred_idx_traj"]]
                # (B, T_traj, demb) -> (N, demb)
                dec_inp_intent = dec_inp_intent[inputs["pred_idx_traj"]]
                logits_done, logits_todo = self.dec_intent(
                    tok_inputs_done=input_idx_done,
                    tok_inputs_todo=input_idx_todo,
                    ctx_inputs=dec_inp_intent,
                )
                logits["intent_done_output"] = logits_done
                logits["intent_todo_output"] = logits_todo

            losses, accuracies = self.compute_loss_and_accuracy(logits, inputs)
            return losses, accuracies

        # inference mode: return prediction probabilities of the latest step
        elif mode == "inference_action":

            # for inference we only predict the next action
            assert dec_inp_action.shape[0] == 1
            logits["action_output"] = self.dec_action(dec_inp_action[0, -1])
            logits["arg1_output"] = self.dec_arg1(dec_inp_action[0, -1])
            # logits["arg2_output"] = self.dec_arg2(dec_inp_action[0, -1])

            predictions = {}
            for key, logit in logits.items():
                probs = F.softmax(logit.squeeze(), dim=0)
                k = min(self.args.dec_params["keep_topk_prob"], len(probs))
                probs_sorted, idx_sorted = probs.topk(k)
                pred_strs = self.prediction_id_to_string(idx_sorted, key)
                predictions[key] = list(
                    zip(pred_strs, idx_sorted.tolist(), probs_sorted.tolist())
                )

            return predictions

        elif mode == "inference_intent":
            word2id = self.vocabs["input_vocab_word2id"]
            intent2id = self.vocabs["output_vocab_intent"]
            seq_done, seq_todo = self.dec_intent.inference(
                ctx_inputs=dec_inp_intent[:, -1],
                vocabs=self.vocabs,
                max_dec_length=self.dec_params["max_intent_length"],
                decoding_strategy=self.dec_params["intent_decoding"],
            )
            return seq_done, seq_todo

        else:
            raise ValueError(
                "mode must be either `training`, `validation` or `inference`"
            )

    def prediction_id_to_string(self, prediction_ids, output_type):
        v = self.output_vocabs[output_type]
        return [v[pid.item()] for pid in prediction_ids]

    def compute_loss_and_accuracy(self, logits_dict, labels_dict):
        """
        comptute the loss for all the predictions
        """
        losses, accuracies = {}, {}
        for pred_type, logits in logits_dict.items():
            if "arg2" in pred_type:
                # remove arg2 prediction
                continue
            logits = logits.view(-1, logits.shape[-1])
            label = labels_dict[pred_type].view(-1)
            losses[pred_type] = F.cross_entropy(
                logits, label, ignore_index=0, reduction="mean"  # padding index
            )

            if "intent" in pred_type:
                logits = logits[label != 0]
                label = label[label != 0]
            _, pred = logits.topk(1, dim=1)
            accuracies[pred_type] = (pred.squeeze() == label).sum() / len(label)

        # use focal loss for object prediction due to its label imbalance nature
        # if self.args.loss['focal_gamma'] > 0:
        #     for pred_type in ['arg1_output', 'arg2_output']:
        #         B, T, obj_cls_num = preds[pred_type].shape
        #         logits = preds[pred_type].view(-1, obj_cls_num)
        #         probs = F.softmax(logits, dim=1)[range(B * T), labels[pred_type].view(-1)]
        #         losses[pred_type] *= (1 - probs) ** self.args.focal_gamma

        # for pred_type, loss in losses.items():
        #     losses[pred_type] = loss.mean()

        return losses, accuracies

    def step(
        self,
        inputs,
        new_input,
        data_handler,
        visfeat_extractor,
    ):
        """
        return single-step predictions during inference
        """

        # add visual feature
        if data_handler.enable_vision:
            if "vis_feats" not in inputs:
                inputs["vis_feats"] = visfeat_extractor.featurize(
                    [new_input["frame"]], batch_size=1
                )
            else:
                inputs["vis_feats"] = visfeat_extractor.append_feature(
                    new_input["frame"], inputs["vis_feats"]
                )
            inputs["ordering_frames"].append(inputs["ordering_step"])

        # add last action
        if "action" in new_input:
            word2id = self.vocabs["input_vocab_word2id"]
            pred2lang = self.vocabs["predicate_to_language"]
            inputs["action_input"].append(
                [word2id.get(w, 1) for w in pred2lang[new_input["action"]].split()]
            )
            inputs["arg1_input"].append(
                [word2id.get(w, 1) for w in pred2lang[new_input["arg1"]].split()]
            )
            # inputs["arg2_input"].append(
            #     [word2id.get(w, 1) for w in pred2lang[new_input["arg2"]].split()]
            # )
            inputs["ordering_action"].append(inputs["ordering_step"])

        inputs["ordering_step"] += 1
        device = self.input_embedder.weight.device

        predictions = {}
        if self.add_intention:
            intent_preds, inputs = self.predict_intent(inputs, data_handler)
            predictions.update(intent_preds)
            inputs["ordering_intent"].append(inputs["ordering_step"])
            inputs["ordering_step"] += 1

        if inputs["ordering_step"] > 1020:
            # a bad fix of the positional embedding out-of-boundary bug
            inputs["ordering_step"] = 1020

        inputs_t = data_handler.tensorize_and_pad_batch(
            [inputs], inference_mode=True, device=device
        )
        action_preds = self(inputs_t, mode="inference_action")
        predictions.update(action_preds)

        return predictions, inputs

    def predict_intent(self, inputs, data_handler):
        device = self.input_embedder.weight.device
        inputs_t = data_handler.tensorize_and_pad_batch(
            [inputs], inference_mode=True, device=device
        )
        id_seq_done, id_seq_todo = self(inputs_t, mode="inference_intent")
        str_seq_done = self.prediction_id_to_string(id_seq_done, "intent_done_output")
        str_seq_todo = self.prediction_id_to_string(id_seq_todo, "intent_todo_output")
        predictions = {"intent_done": str_seq_done, "intent_todo": str_seq_todo}

        if "intent_done_input" not in inputs:
            inputs["intent_done_input"] = []
            inputs["intent_todo_input"] = []
            inputs["ordering_intent"] = []
            inputs["max_intent_length"] = 0
        w2id = self.vocabs["input_vocab_word2id"]
        inputs["intent_done_input"].append([w2id.get(w, 1) for w in str_seq_done])
        inputs["intent_todo_input"].append([w2id.get(w, 1) for w in str_seq_todo])
        intent_length = max(len(str_seq_done), len(str_seq_todo))
        if intent_length > inputs["max_intent_length"]:
            inputs["max_intent_length"] = intent_length

        return predictions, inputs
