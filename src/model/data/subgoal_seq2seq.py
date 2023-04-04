import os
import re
import json
import torch
from torch.nn.utils.rnn import pad_sequence


class SubgoalDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, args, split=None):
        self.args = args
        self.tokenizer = tokenizer

        self.encoder_input_type = args.encoder_input_type
        self.prediction_type = args.prediction_type

        self.pad_id = 0
        self.colon_id = tokenizer(":", add_special_tokens=False)["input_ids"][0]
        self.semicolon_id = tokenizer(";", add_special_tokens=False)["input_ids"][0]

        self.data = []
        if split:
            for s in split:
                self.data_dir = args.data_dir
                self.data_name = "seq2seq_sg_pred_%s.json" % s
                self.data_file = os.path.join(self.data_dir, self.data_name)
                with open(self.data_file, "r") as f:
                    self.data.extend(json.load(f))
                if self.args.debug:
                    self.data = self.data[: 2 * self.args.per_device_batch_size]
                print("[%s] data loaded from %s" % (split, self.data_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def data_collect(self, batch, inference_mode=False, device=None):

        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        if self.encoder_input_type == "text_dialog":
            encoder_text_inputs = batch["text_dialog"]
        elif self.encoder_input_type == "text_dialog_and_act":
            encoder_text_inputs = batch["text_dialog_and_act"]
        elif self.encoder_input_type == "text_dialog_and_sg":
            encoder_text_inputs = list(zip(batch["text_dialog"], batch["text_sg_done"]))
        elif self.encoder_input_type == "text_dialog_and_act_and_sg":
            encoder_text_inputs = list(
                zip(batch["text_dialog_and_act"], batch["text_sg_done"])
            )

        # tokenize encoder inputs
        encoder_inputs = self.tokenizer.batch_encode_plus(
            encoder_text_inputs,
            max_length=self.args.max_encoder_seq_length,
            padding=True,
            truncation="only_first",
            add_special_tokens=True,
            return_tensors="pt",
        )

        model_inputs = {
            "encoder_input_ids": encoder_inputs["input_ids"],
            "encoder_attention_mask": encoder_inputs["attention_mask"],
        }

        if inference_mode:
            if device is not None:
                for k, v in model_inputs.items():
                    model_inputs[k] = v.to(device)
            return model_inputs

        if self.args.use_symbol_embedding_as_input:
            assert (
                self.prediction_type == "sg_done"
            ), "Use symbol embedding as input only for sg_done"

            # get labels
            for pred_type in ["pred", "subj", "obj"]:
                k = "idx_%s_%s" % (self.prediction_type, pred_type)
                model_inputs[k] = torch.concat([torch.tensor(i) for i in batch[k]])

            batch_size = len(batch["idx_%s_pred" % self.prediction_type])
            longest_seq_length = max(
                [len(i) for i in batch["idx_%s_pred" % self.prediction_type]]
            )

            pred_idx = []
            decoder_input_ids = torch.zeros(
                batch_size, 3 * (longest_seq_length - 1) + 1
            ).long()
            decoder_input_attentions = torch.zeros(
                batch_size, 3 * (longest_seq_length - 1) + 1
            ).bool()
            prediction_idx = torch.zeros(
                batch_size, 3 * (longest_seq_length - 1) + 1
            ).bool()

            for b in range(batch_size):
                # add beginning of sequence token
                decoder_input_ids[b, 0] = 11 + 177 + 43
                decoder_input_attentions[b, 0] = 1
                prediction_idx[b, 0] = 1

                for i in range(len(batch["idx_%s_pred" % self.prediction_type][b]) - 1):

                    decoder_input_ids[b, 1 + i * 3 + 1] = batch[
                        "idx_%s_subj" % self.prediction_type
                    ][b][i]
                    decoder_input_ids[b, 1 + i * 3 + 2] = (
                        batch["idx_%s_obj" % self.prediction_type][b][i] + 43
                    )
                    decoder_input_ids[b, 1 + i * 3] = (
                        batch["idx_%s_pred" % self.prediction_type][b][i] + 43 + 177
                    )

                    prediction_idx[b, 1 + i * 3] = 1
                    decoder_input_attentions[b, 1 + i * 3 : 1 + (i + 1) * 3] = 1

            model_inputs["decoder_input_ids"] = decoder_input_ids
            model_inputs["decoder_attention_mask"] = decoder_input_attentions
            model_inputs["prediction_idx"] = prediction_idx

            return model_inputs

        if self.prediction_type == "sg_todo_edh":
            decoder_text_inputs = batch["text_sg_todo_edh"]
        elif self.prediction_type == "sg_todo_all":
            decoder_text_inputs = batch["text_sg_todo_all"]
        elif self.prediction_type == "sg_done":
            decoder_text_inputs = batch["text_sg_done"]
        elif self.prediction_type == "sg_done_and_todo_edh":
            decoder_text_inputs = []
            sg_done = batch["text_sg_done"]
            sg_todo = batch["text_sg_todo_edh"]
            for b in range(len(batch["text_sg_done"])):
                dec_inp = sg_done[b] + " " + sg_todo[b] if sg_done[b] else sg_done[b]
                decoder_text_inputs.append(dec_inp)
        elif self.prediction_type == "sg_done_and_todo_all":
            decoder_text_inputs = []
            sg_done = batch["text_sg_done"]
            sg_todo = batch["text_sg_todo_all"]
            for b in range(len(batch["text_sg_done"])):
                dec_inp = sg_done[b] + " " + sg_todo[b] if sg_done[b] else sg_done[b]
                decoder_text_inputs.append(dec_inp)

        # tokenize decoder inputs
        decoder_inputs = self.tokenizer.batch_encode_plus(
            decoder_text_inputs,
            max_length=self.args.max_decoder_seq_length,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        model_inputs["decoder_input_ids"] = decoder_inputs["input_ids"]
        model_inputs["decoder_attention_mask"] = decoder_inputs["attention_mask"]

        # get positions to make predictions
        pred_idx_colon = decoder_inputs["input_ids"] == self.colon_id
        pred_idx_semicolon = decoder_inputs["input_ids"] == self.semicolon_id
        model_inputs["prediction_idx"] = pred_idx_colon | pred_idx_semicolon

        # get labels
        for pred_type in ["pred", "subj", "obj"]:
            k = "idx_%s_%s" % (self.prediction_type, pred_type)
            model_inputs[k] = torch.concat([torch.tensor(i) for i in batch[k]])

            # number of lables should equal to the number of predictions
            assert len(model_inputs[k]) == model_inputs["prediction_idx"].sum()

        return model_inputs
