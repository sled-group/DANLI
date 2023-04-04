import pprint
import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoModel, BartModel, BartConfig

from definitions.teach_tasks import GoalArguments, GoalConditions, GoalReceptacles
from definitions.symbol_to_language import symbol_to_phrase, subgoal_to_language


class SubgoalPredictor(nn.Module):
    def __init__(self, args, tokenizer):
        """
        transformer model of high-level planning
        """
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.prediction_type = args.prediction_type

        # # pretrained seq2seq language model: BART or T5
        if not hasattr(args, 'is_inference'):
            self.model = AutoModel.from_pretrained(args.plm_model_name)
        else:
            self.model = BartModel(BartConfig.from_pretrained(args.plm_model_name))
        if hasattr(self.args, "device"):
            self.model.to(self.args.device)

        # output vocabulary
        self.output_vocab = {
            "pred": {i.value: i.name for i in GoalConditions},
            "subj": {i.value: i.name for i in GoalArguments},
            "obj": {i.value: i.name for i in GoalReceptacles},
        }

        # language input embedding
        if not args.use_symbol_embedding_as_input:
            self.embed = self.model.shared
        else:
            self.input_size, self.demb = self.model.shared.weight.shape
            len_pred = len(GoalConditions)
            len_subj = len(GoalArguments)
            len_obj = len(GoalReceptacles)
            self.embed = nn.Embedding(len_pred + len_subj + len_obj + 1, self.demb)
            self.args.tie_encoder_decoder_embeddings = False

        # build decoder weights using input embeddings
        self._build_output_layers()

        # special tokens
        self.eos_idx = GoalConditions["EOS"].value
        self.none_idx = 0

    def _build_output_layers(self):

        predicates = [symbol_to_phrase(i.name) for i in GoalConditions]
        predicates = self.tokenizer.batch_encode_plus(
            predicates, add_special_tokens=False, padding=True, return_tensors="pt"
        )["input_ids"]

        subjects = [symbol_to_phrase(i.name) for i in GoalArguments]
        subjects = self.tokenizer.batch_encode_plus(
            subjects, add_special_tokens=False, padding=True, return_tensors="pt"
        )["input_ids"]

        objects = [symbol_to_phrase(i.name) for i in GoalReceptacles]
        objects = self.tokenizer.batch_encode_plus(
            objects, add_special_tokens=False, padding=True, return_tensors="pt"
        )["input_ids"]

        self.output_phrase_ids = {
            "pred": predicates,
            "subj": subjects,
            "obj": objects,
        }
        self.output_ids_on_device = False

        self.input_size, self.demb = self.embed.weight.shape
        self.output_size_pred = len(GoalConditions)
        self.output_size_subj = len(GoalArguments)
        self.output_size_obj = len(GoalReceptacles)

        if self.args.tie_encoder_decoder_embeddings:
            self.decoders = nn.ModuleDict(
                {
                    "pred": nn.Sequential(nn.Linear(self.demb, self.demb), nn.ReLU()),
                    "subj": nn.Sequential(nn.Linear(self.demb, self.demb), nn.ReLU()),
                    "obj": nn.Sequential(nn.Linear(self.demb, self.demb), nn.ReLU()),
                }
            )
        else:
            self.decoder_pred = nn.Linear(self.demb, self.output_size_pred)
            self.decoder_subj = nn.Linear(self.demb, self.output_size_subj)
            self.decoder_obj = nn.Linear(self.demb, self.output_size_obj)

    def forward(self, inputs, is_inference=False):
        """
        forward the model for multiple time-steps (used for training)
        """
        if not self.output_ids_on_device:
            device = self.embed.weight.device
            for key, value in self.output_phrase_ids.items():
                self.output_phrase_ids[key] = value.to(device)
            self.output_ids_on_device = True

        # plm outputs
        if not self.args.use_symbol_embedding_as_input:
            outputs = self.model(
                input_ids=inputs["encoder_input_ids"],
                attention_mask=inputs["encoder_attention_mask"],
                decoder_input_ids=inputs["decoder_input_ids"],
                decoder_attention_mask=inputs["decoder_attention_mask"],
            )
        else:
            decoder_inputs_embeds = self.embed(inputs["decoder_input_ids"])
            outputs = self.model(
                input_ids=inputs["encoder_input_ids"],
                attention_mask=inputs["encoder_attention_mask"],
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=inputs["decoder_attention_mask"],
            )

        # get predictions
        if not is_inference:
            pred_idx = inputs["prediction_idx"]
            dec_outs = outputs.last_hidden_state[pred_idx]
        else:
            assert outputs.last_hidden_state.shape[0] == 1
            dec_outs = outputs.last_hidden_state[0, -2]
            # Note: during inference, we assume that batch size is 1 and only
            # predict the latest (next) token. Thus we use the position of the
            # last ";" as the prediciton index. It is position -2 since the last
            # token should be </s> (though it is not seen by the model due to
            # causal masking).

        if self.args.tie_encoder_decoder_embeddings:
            logits = {}
            for out in ["pred", "subj", "obj"]:
                output_weights = self.embed(self.output_phrase_ids[out]).sum(dim=1)
                dec_outs = self.decoders[out](dec_outs)
                logits[out] = torch.matmul(dec_outs, output_weights.transpose(0, 1))
        else:
            logits = {
                "pred": self.decoder_pred(dec_outs),
                "subj": self.decoder_subj(dec_outs),
                "obj": self.decoder_obj(dec_outs),
            }

        # training/validation mode: compute losses
        if not is_inference:
            losses, accuracies = self.compute_loss_and_accuracy(logits, inputs)
            return losses, accuracies

        # inference mode: compute predictions
        predictions = {}
        for key, logit in logits.items():
            probs = F.softmax(logit.squeeze(), dim=0)
            probs[self.none_idx] = 0  # remove none prediction
            probs_sorted, idx_sorted = probs.topk(len(probs))
            decoded_strings = [self.output_vocab[key][i.item()] for i in idx_sorted]
            predictions[key] = list(
                zip(decoded_strings, idx_sorted.tolist(), probs_sorted.tolist())
            )

        return predictions

    def compute_loss_and_accuracy(self, dict_of_logits, dict_of_labels):
        """
        comptute the loss for all the predictions
        """
        losses, accuracies = {}, {}
        for pred_type, logits in dict_of_logits.items():
            label_key = "idx_%s_%s" % (self.prediction_type, pred_type)
            labels = dict_of_labels[label_key].view(-1)
            non_pad_labels = labels[labels.ne(0)]

            loss = F.cross_entropy(logits, labels, ignore_index=0, reduction="mean")
            losses[pred_type] = (
                loss if len(non_pad_labels) > 0 else torch.zeros_like(loss)
            )

            non_pad_preds = torch.argmax(logits, dim=1)[labels.ne(0)]
            # accu = (non_pad_preds == non_pad_labels).sum() / len(non_pad_labels)
            # accuracies[pred_type] = (
            #     accu if len(non_pad_labels) > 0 else torch.zeros_like(accu)
            # )
            accuracies[pred_type] = (non_pad_preds == non_pad_labels).sum()
            accuracies[pred_type + "_cnt"] = len(non_pad_labels)

        return losses, accuracies

    def generate(self, inputs, max_step=64, decoding_strategy="greedy"):
        """
        step the model for one time-step (used for inference)
        """
        if decoding_strategy == "greedy":
            return self.greedy_decoding(inputs, max_step=max_step)
        else:
            raise NotImplementedError(decoding_strategy + " is not implemented")

    def greedy_decoding(self, batch_inputs, max_step=32):
        """
        greedy decoding
        """
        device = self.embed.weight.device
        predictions = []
        for b in range(batch_inputs["encoder_input_ids"].shape[0]):

            predictions.append(
                {"decoding_text": "", "subgoals_done": [], "subgoals_future": []}
            )

            inputs = {
                "encoder_input_ids": batch_inputs["encoder_input_ids"][b].unsqueeze(0),
                "encoder_attention_mask": batch_inputs["encoder_attention_mask"][
                    b
                ].unsqueeze(0),
            }

            if "sg_done" in self.prediction_type:
                decoding_text = "Follower completed subgoals:"
                pred_record_key = "subgoals_done"
            else:
                decoding_text = "Future subgoals:"
                pred_record_key = "subgoals_future"

            decoder_inputs = self.tokenizer.batch_encode_plus(
                [decoding_text],
                max_length=self.args.max_decoder_seq_length,
                padding=True,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            inputs["decoder_input_ids"] = decoder_inputs["input_ids"].to(device)
            inputs["decoder_attention_mask"] = decoder_inputs["attention_mask"].to(device)

            switch_record_key = False
            for i in range(max_step):
                step_prediction = self(inputs, is_inference=True)

                pred_name, pred_idx, pred_prob = step_prediction["pred"][0]
                subj_name, subj_idx, subj_prob = step_prediction["subj"][0]
                obj_name, obj_idx, obj_prob = step_prediction["obj"][0]

                if pred_idx == self.eos_idx:
                    decoding_text += " none" if decoding_text[-1] == ":" else ""
                    sg = ("NONE", "EOS", "NONE")
                    if (
                        "_and_" in self.prediction_type
                        and "Future" not in decoding_text
                    ):
                        decoding_text += " Future subgoals:"
                        switch_record_key = True
                    else:
                        predictions[b][pred_record_key].append(sg)
                        predictions[b]["decoding_text"] = decoding_text
                        break
                else:
                    sg = (subj_name, pred_name, obj_name)
                    sg_nl = subgoal_to_language(sg)
                    decoding_text += " " + sg_nl + ";"

                decoder_inputs = self.tokenizer.batch_encode_plus(
                    [decoding_text],
                    max_length=self.args.max_decoder_seq_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                inputs["decoder_input_ids"] = decoder_inputs["input_ids"].to(device)
                inputs["decoder_attention_mask"] = decoder_inputs["attention_mask"].to(device)

                predictions[b][pred_record_key].append(sg)
                if switch_record_key:
                    pred_record_key = "subgoals_future"

            predictions[b]["decoding_text"] = decoding_text

        return predictions
