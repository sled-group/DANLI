import torch
from pprint import pprint
import pytorch_lightning as pl
from importlib import import_module
from torch.optim.lr_scheduler import StepLR

class PLModelWrapper(pl.LightningModule):
    def __init__(self, model_cls, args, vocabs, inference_params=None):
        """
        Use Pytorch Lighning for training management
        """
        super().__init__()

        self.model_cls = model_cls
        self.args = args
        self.vocabs = vocabs
        self.decoding_params = args.dec_params if hasattr(args, "dec_params") else None
        self.batch_num_train = (
            args.batch_num_train if hasattr(args, "batch_num_train") else None
        )

        self.save_hyperparameters(args)

        # create the model to be trained
        ModelClass = import_module("model.model.{}".format(model_cls)).Model
        self.model = ModelClass(args, vocabs)

    def forward(self, inputs, mode):
        # in lightning, forward defines the prediction/inference actions
        return self.model(inputs, mode=mode)

    def predict(self, *args, **kargs):
        return self.model.step(*args, **kargs)

    def configure_optimizers(self):
        assert self.args.optimizer in ("adam", "adamw")
        OptimizerClass = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[
            self.args.optimizer
        ]
        optimizer = OptimizerClass(
            self.model.parameters(),
            lr=self.args.lr["init"],
            weight_decay=self.args.weight_decay,
        )

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                             gamma=self.args.lr['decay_scale'],
        #                                             step_size=self.args.lr['decay_epoch'])

        # def warmup_lr_lambda(current_step):
        #     if current_step < self.args.lr['warmup_steps']:
        #         return float(current_step) / float(max(1, self.args.lr['warmup_steps']))
        #     else:
        #         return 1.0

        print("warmup steps:", self.args.lr["warmup_steps"])
        print("total steps:", self.args.epochs * self.batch_num_train)

        # def get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
        # ):
        #     global lr_lambda
        #     def lr_lambda(current_step: int):
        #         if current_step < num_warmup_steps:
        #             return float(current_step) / float(max(1, num_warmup_steps))
        #         return max(
        #             0.0,
        #             float(num_training_steps - current_step)
        #             / float(max(1, num_training_steps - num_warmup_steps)),
        #         )

        #     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     self.args.lr["warmup_steps"],
        #     self.args.epochs * self.batch_num_train,
        # )
        scheduler = StepLR(optimizer, step_size=self.args.lr['decay_epoch'], gamma=self.args.lr['decay_scale'])
        # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #                                                      lr_lambda=warmup_lr_lambda)

        return (
            [optimizer],
            [
                {
                    "name": "lr",
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
                # {
                #     'name': 'lr',
                #     'scheduler': scheduler,
                #     'interval': 'step',
                #     'frequency': 1
                # }
            ],
        )

    def training_step(self, batch, batch_idx):
        batch["bidx"] = batch_idx
        if "pred_idx_traj" in batch and batch["pred_idx_traj"].sum().item() == 0:
            # no valid prediction positions
            return None
        # if batch_idx < 65:
        #     return None
        losses, accuracies = self(batch, mode="training")
        loss_record, accu_record = {}, {}
        total_loss = 0
        for pred_type, loss in losses.items():
            weight = self.args.loss["weights"].get(pred_type, 1.0)
            total_loss += weight * loss
            loss_record[pred_type[:-7]] = weight * loss.item()
            accu_record[pred_type[:-7]] = accuracies[pred_type].item()
        loss_record["total"] = total_loss.item()

        # record logs
        for param_group in self.optimizers().param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, prog_bar=True, logger=True)

        bs = len(batch["action_output"])
        for pred_type, loss in loss_record.items():
            self.log(
                "train/loss/%s" % pred_type,
                loss,
                batch_size=bs,
                prog_bar=False,
                logger=True,
            )
        for pred_type, accu in accu_record.items():
            self.log(
                "train/accu/%s" % pred_type,
                accu,
                batch_size=bs,
                prog_bar=False,
                logger=True,
            )

        # self.log("l_a", loss_record["action"], batch_size=bs, prog_bar=True)
        # self.log("ac_a", accu_record["action"], batch_size=bs, prog_bar=True)
        # if self.args.exp_type != "navi":
        #     self.log("l_o", loss_record["arg1"], batch_size=bs, prog_bar=True)
        #     self.log("ac_o", accu_record["arg1"], batch_size=bs, prog_bar=True)
        #     if self.args.add_intention:
        #         # loss_i = (loss_record['intent_done'] + loss_record['intent_todo']) / 2
        #         # self.log("l_i", loss_i, batch_size=bs, prog_bar=True)
        #         self.log(
        #             "ac_i_d", accu_record["intent_done"], batch_size=bs, prog_bar=True
        #         )
        #         self.log(
        #             "ac_i_t", accu_record["intent_todo"], batch_size=bs, prog_bar=True
        #         )

        # for param_group in self.optimizers().param_groups:
        #     print('learning rate:', param_group['lr'])
        # if batch_idx in [66]:
        # for i in range(len(batch['action_input'])):
        #     print(i)
        #     print('input', batch['action_input'][i])
        # for i in range(len(batch['action_output'])):
        #     print(i)
        #     print('label', batch['action_output'][i])
        #     print('pred', preds['action_output'][i])
        # print('pred', preds['action_output'].shape, preds['action_output'])
        # quit()
        return total_loss

    def validation_step(self, val_batch, batch_idx, dataset_idx):
        if (
            "pred_idx_traj" in val_batch
            and val_batch["pred_idx_traj"].sum().item() == 0
        ):
            # no valid prediction positions
            return None

        if dataset_idx == 0:
            valid_type = "seen"
        else:
            valid_type = "unseen"

        losses, accuracies = self(val_batch, mode="validation")
        loss_record, accu_record = {}, {}
        total_loss = 0
        for pred_type, loss in losses.items():
            weight = self.args.loss["weights"].get(pred_type, 1.0)
            total_loss += weight * loss
            loss_record[pred_type[:-7]] = weight * loss.item()
            accu_record[pred_type[:-7]] = accuracies[pred_type].item()
        loss_record["total"] = total_loss.item()

        return loss_record, accu_record

    def validation_epoch_end(self, valid_metrics):
        valid_loss_avg, valid_accu_avg = {}, {}
        for loader_idx, val_outputs in enumerate(valid_metrics):
            valid_type = ["seen", "unseen"][loader_idx]
            for losses, accuracies in val_outputs:
                for pred_type, loss in losses.items():
                    name = "valid_%s/loss/%s" % (valid_type, pred_type)
                    if name not in valid_loss_avg:
                        valid_loss_avg[name] = 0
                    valid_loss_avg[name] += loss

                for pred_type, accu in accuracies.items():
                    name = "valid_%s/accu/%s" % (valid_type, pred_type)
                    if name not in valid_accu_avg:
                        valid_accu_avg[name] = 0
                    valid_accu_avg[name] += accu

        # record logs
        for name in valid_loss_avg:
            valid_loss_avg[name] /= len(valid_metrics[0 if "_seen" in name else 1])
            self.log(name, valid_loss_avg[name], prog_bar=False, logger=True)
        for name in valid_accu_avg:
            valid_accu_avg[name] /= len(valid_metrics[0 if "_seen" in name else 1])
            self.log(name, valid_accu_avg[name], prog_bar=False, logger=True)

        total_avg_accu = 0
        acc_set = ["action"] if self.args.exp_type == 'navi' else ["action", "arg1"]
        for s in ["seen", "unseen"]:
            for acc in acc_set:
                total_avg_accu += valid_accu_avg["valid_%s/accu/%s" % (s, acc)]
        self.log("avg_accu", total_avg_accu / 2*len(acc_set), prog_bar=True, logger=True)

        self.log("val_unseen_total_loss", valid_loss_avg["valid_unseen/loss/total"])
        self.log("vse_acc_a", valid_accu_avg["valid_seen/accu/action"], prog_bar=True)
        self.log("vun_acc_a", valid_accu_avg["valid_unseen/accu/action"], prog_bar=True)
        if self.args.exp_type != "navi":
            self.log("vse_acc_o", valid_accu_avg["valid_seen/accu/arg1"], prog_bar=True)
            self.log(
                "vun_acc_o", valid_accu_avg["valid_unseen/accu/arg1"], prog_bar=True
            )
            if self.args.add_intention:
                # loss_i = (loss_record['intent_done'] + loss_record['intent_todo']) / 2
                # self.log("l_i", loss_i, batch_size=bs, prog_bar=True)
                self.log(
                    "vse_acc_i_d",
                    valid_accu_avg["valid_seen/accu/intent_done"],
                    prog_bar=True,
                )
                self.log(
                    "vse_acc_i_t",
                    valid_accu_avg["valid_seen/accu/intent_todo"],
                    prog_bar=True,
                )
                self.log(
                    "vun_acc_i_d",
                    valid_accu_avg["valid_unseen/accu/intent_done"],
                    prog_bar=True,
                )
                self.log(
                    "vun_acc_i_t",
                    valid_accu_avg["valid_unseen/accu/intent_todo"],
                    prog_bar=True,
                )
