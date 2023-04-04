import os
import torch
import math
from torch import nn, Tensor


class PosEncoding(nn.Module):
    '''
    Transformer-style positional encoding with wavelets
    '''

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe[None])

    # def forward(self, lang, frames, actions, lens_lang, lens_frames, pos=None):
    #     if pos is None:
    #         enc = self.pe[:, :lang.shape[1] + frames.shape[1]]
    #     else:
    #         enc = [[] for _ in range(len(lang))]
    #         for batch_idx in range(pos.shape[0]):
    #             for pos_idx in range(lang.shape[1] + frames.shape[1]):
    #                 enc[batch_idx].append(self.pe[0, pos[batch_idx, pos_idx]])
    #         enc = torch.stack([torch.stack(pos_batch) for pos_batch in enc])
    #     enc = enc / math.sqrt(self.d_model)
    #     lang = lang + enc[:, :lang.shape[1]]
    #     for i in range(frames.shape[0]):
    #         frames[i] = frames[i] + enc[0, lens_lang[i]: lens_lang[i] + frames.shape[1]]
    #     # use the same position indices for actions as for the frames
    #     for i in range(actions.shape[0]):
    #         actions[i] = actions[i] + enc[0, lens_lang[i]: lens_lang[i] + actions.shape[1]]
    #     return lang, frames, actions

    def forward(self, x: Tensor, position: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe.squeeze()[position]


class LearnedEncoding(nn.Module):
    '''
    Learned additive encoding implemented on top of nn.Embedding
    '''

    # def __init__(self, d_model, vocab_size, init_range=0.1, padding_idx=0):
    def __init__(self, d_model, vocab_size, gain=1.0, padding_idx=0):
        super().__init__()
        if padding_idx is None:
            self.emb = nn.Embedding(vocab_size, d_model)
        else:
            self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # self.emb.weight.data.uniform_(-init_range, init_range)
        nn.init.xavier_uniform_(self.emb.weight, gain=gain)

    def forward(self, x, tokens):
        tokens_emb = self.emb(tokens)
        return x + tokens_emb
