from torch import nn

from ..utils import data_util


class Model(nn.Module):
    def __init__(self, args, vocabs):
        '''
        Abstract model
        '''
        nn.Module.__init__(self)
        self.args = args
        self.vocabs = vocabs

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose):
        '''
        compute model-specific metrics and put it to metrics dict
        '''
        raise NotImplementedError

    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        raise NotImplementedError()

    def inference(self, vocab, **inputs):
        '''
        make a single-step prediction during inference
        '''
        raise NotImplementedError()

    def compute_batch_loss(self, model_out, gt_dict):
        '''
        compute the loss function for a single batch
        '''
        raise NotImplementedError()

    # def compute_loss(self, model_outs, gt_dicts):
    #     '''
    #     compute the loss function for several batches
    #     '''
    #     # compute losses for each batch
    #     losses = {}
    #     for dataset_key in model_outs.keys():
    #         losses[dataset_key] = self.compute_batch_loss(
    #             model_outs[dataset_key], gt_dicts[dataset_key])
    #     return losses
