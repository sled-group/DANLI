import torch
from pprint import pprint
import pytorch_lightning as pl
from importlib import import_module
from torch.optim.lr_scheduler import StepLR

from model.model.subgoal_predictor import SubgoalPredictor
from transformers import AdamW, get_linear_schedule_with_warmup


class PLModelWrapper(pl.LightningModule):
    def __init__(self, args, tokenizer):
        """
        Use Pytorch Lighning for training management
        """
        super().__init__()

        self.args = args
        self.save_hyperparameters(args)
        self.model = SubgoalPredictor(args, tokenizer)

    def forward(self, inputs, is_inference=False):
        # in lightning, forward defines the prediction/inference actions
        return self.model(inputs, is_inference=is_inference)

    def predict(self, *args, **kargs):
        return self.model.generate(*args, **kargs)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        self.loss_weights = {
            "pred": self.args.loss_weight_pred,
            "subj": self.args.loss_weight_subj,
            "obj": self.args.loss_weight_obj,
        }

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        losses, accuracies = self(batch)
        loss_record, accu_record, accu_cnt_record = {}, {}, {}
        total_loss = 0
        for pred_type, loss in losses.items():
            weight = self.loss_weights[pred_type]
            total_loss += weight * loss
            loss_record[pred_type] = weight * loss.item()
            accu_record[pred_type] = accuracies[pred_type].item()
            accu_cnt_record[pred_type + "_cnt"] = accuracies[pred_type + "_cnt"]
        loss_record["total"] = total_loss.item()

        # record logs
        for param_group in self.optimizers().param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, prog_bar=True, logger=True)

        bs = len(batch["encoder_input_ids"])
        for pred_type, loss in loss_record.items():
            self.log(
                "train/loss/%s" % pred_type,
                loss,
                batch_size=bs,
                prog_bar=False,
                logger=True,
            )
        for pred_type, correct_cnt in accu_record.items():
            cnt = accu_cnt_record[pred_type + "_cnt"]
            accu = correct_cnt / cnt if cnt > 0 else 0
            self.log(
                "train/accu/%s" % pred_type,
                accu,
                batch_size=bs,
                prog_bar=False,
                logger=True,
            )

        return total_loss

    def validation_step(self, val_batch, batch_idx, dataset_idx):
        losses, accuracies = self(val_batch)
        loss_record, accu_record, accu_cnt_record = {}, {}, {}
        total_loss = 0
        for pred_type, loss in losses.items():
            weight = self.loss_weights[pred_type]
            total_loss += weight * loss
            loss_record[pred_type] = weight * loss.item()
            accu_record[pred_type] = accuracies[pred_type].item()
            accu_cnt_record[pred_type + "_cnt"] = accuracies[pred_type + "_cnt"]
        loss_record["total"] = total_loss.item()

        return loss_record, accu_record, accu_cnt_record

    def validation_epoch_end(self, valid_metrics):
        valid_loss_avg, valid_accu_avg, valid_accu_cnt = {}, {}, {}
        for loader_idx, val_outputs in enumerate(valid_metrics):
            valid_type = ["seen", "unseen"][loader_idx]
            for losses, accuracies, counts in val_outputs:
                for pred_type, loss in losses.items():
                    name = "valid_%s/loss/%s" % (valid_type, pred_type)
                    if name not in valid_loss_avg:
                        valid_loss_avg[name] = 0
                    valid_loss_avg[name] += loss

                for pred_type, accu in accuracies.items():
                    name = "valid_%s/accu/%s" % (valid_type, pred_type)
                    if name not in valid_accu_avg:
                        valid_accu_avg[name] = 0
                        valid_accu_cnt[name] = 0
                    valid_accu_avg[name] += accu
                    valid_accu_cnt[name] += counts[pred_type + "_cnt"]

        # record logs
        for name in valid_loss_avg:
            valid_loss_avg[name] /= len(valid_metrics[0 if "_seen" in name else 1])
            self.log(name, valid_loss_avg[name], prog_bar=False, logger=True)
        for name in valid_accu_avg:
            if valid_accu_cnt[name] == 0:
                valid_accu_cnt[name] = 1
            valid_accu_avg[name] /= valid_accu_cnt[name]  # update to micro avg
            self.log(name, valid_accu_avg[name], prog_bar=False, logger=True)

        total_avg_accu = 0
        acc_set = ["pred", "obj", "subj"]
        for s in ["seen", "unseen"]:
            for p in acc_set:
                total_avg_accu += valid_accu_avg["valid_%s/accu/%s" % (s, p)]
                self.log(
                    "%s_acc_%s" % (s, p),
                    valid_accu_avg["valid_%s/accu/%s" % (s, p)],
                    prog_bar=True,
                )

        self.log(
            "avg_accu", total_avg_accu / (2 * len(acc_set)), prog_bar=True, logger=True
        )
