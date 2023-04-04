import os
import torch
from torch import nn
from torch.nn import functional as F


class DecoderIntent(nn.Module):
    """
    intention predictor
    """

    def __init__(self, args, output_size, embedder):
        super().__init__()
        self.hid_size = args.demb
        self.out_size = output_size
        self.embedder = embedder
        self.n_layers = args.dec_intent["layers"]

        self.dec_done = nn.GRU(
            input_size=args.demb * 2,
            hidden_size=args.demb,
            num_layers=args.dec_intent["layers"],
            batch_first=True,
        )
        self.dec_todo = nn.GRU(
            input_size=args.demb * 2,
            hidden_size=args.demb,
            num_layers=args.dec_intent["layers"],
            batch_first=True,
        )
        self.pred_head_done = nn.Linear(args.demb, output_size)
        self.pred_head_todo = nn.Linear(args.demb, output_size)

    def forward(self, tok_inputs_done, tok_inputs_todo, ctx_inputs):
        """Decode what is just done and what to do next for training

        :param dec_input: decoding result of the previous intent decoding step
                          For Training : Tensor of shape (N, T_dec)
                          For Inference: Tensor of shape (1, 1, H)
        :param ctx_input: encoder output of the multi-modal transformer at the trajectory step
                          The same context vector is fed to each intent decoding step.
                          Tensor of shape (N, H)

        :return:
            outs_done: predictions of what has been done
            outs_todo: predictions of what to do next
        """
        outs_done, _ = self.decode(tok_inputs_done, ctx_inputs, self.dec_done)
        outs_todo, _ = self.decode(tok_inputs_todo, ctx_inputs, self.dec_todo)
        preds_done = self.pred_head_done(outs_done)
        preds_todo = self.pred_head_todo(outs_todo)
        return preds_done, preds_todo

    def inference(
        self,
        ctx_inputs,
        vocabs,
        max_dec_length=64,
        decoding_strategy="greedy",
    ):
        """Decode what is just done and what to do next during inference

        :param ctx_inputs: [description]
        :param done_beg: [description]
        :param todo_beg: [description]
        :param done_end: [description]
        :param todo_end: [description]
        """
        if decoding_strategy != "greedy":
            raise NotImplementedError("Have to use greedy decoding now!")

        device = self.embedder.weight.device

        word2id = vocabs["input_vocab_word2id"]
        intent2id = vocabs["output_vocab_intent"]
        done_beg_idx = word2id["[BEG_DONE]"]
        done_end_idx = intent2id["[END_DONE]"]
        todo_beg_idx = word2id["[BEG_TODO]"]
        todo_end_idx = intent2id["[END_TODO]"]
        if not hasattr(self, "intent_out_id_to_in_id"):
            print("create a mapping from output intent ids to input intent ids")
            self.intent_out_id_to_in_id = {}
            for word, out_id in intent2id.items():
                inp_id = word2id.get(word, 1)
                self.intent_out_id_to_in_id[out_id] = inp_id

        inp_seq_done = torch.tensor([[done_beg_idx]], device=device)
        inp_seq_todo = torch.tensor([[todo_beg_idx]], device=device)
        last_hidden_done, last_hidden_todo = None, None
        dec_completed_done, dec_completed_todo = False, False
        out_seq_done, out_seq_todo = [], []
        for t in range(max_dec_length):
            if not dec_completed_done:
                outs_done, last_hidden_done = self.decode(
                    inp_seq_done, ctx_inputs, self.dec_done, last_hidden_done
                )
                preds_done = self.pred_head_done(outs_done)  # [B, T, V]
                pred_done = preds_done[:, -1, :]
                prob_done, out_token_done = torch.topk(pred_done, 1, dim=-1)
                out_seq_done.append(out_token_done)
                inp_token_done = self.intent_out_id_to_in_id[out_token_done.item()]
                inp_token_done = torch.tensor([[inp_token_done]], device=device)
                inp_seq_done = torch.cat([inp_seq_done, inp_token_done], dim=1)
                if out_token_done.item() == done_end_idx:
                    dec_completed_done = True

            if not dec_completed_todo:
                outs_todo, last_hidden_todo = self.decode(
                    inp_seq_todo, ctx_inputs, self.dec_todo, last_hidden_todo
                )
                preds_todo = self.pred_head_todo(outs_todo)  # [B, T, V]
                pred_todo = preds_todo[:, -1, :]
                prob_todo, out_token_todo = torch.topk(pred_todo, 1, dim=-1)
                out_seq_todo.append(out_token_todo)
                inp_token_todo = self.intent_out_id_to_in_id[out_token_todo.item()]
                inp_token_todo = torch.tensor([[inp_token_todo]], device=device)
                inp_seq_todo = torch.cat([inp_seq_todo, inp_token_todo], dim=1)
                if out_token_todo.item() == todo_end_idx:
                    dec_completed_todo = True

            if dec_completed_done and dec_completed_todo:
                break

        return out_seq_done, out_seq_todo

    def decode(self, tok_inputs, ctx_inputs, decoder, h0=None):
        T_dec, H = tok_inputs.shape[-1], ctx_inputs.shape[-1]

        tok_inputs = self.embedder(tok_inputs)
        # tok_inputs = tok_inputs.view(-1, T_dec, H)  # (B*T_traj, T_dec, H)
        ctx_inputs = ctx_inputs.unsqueeze(1)  # (N, 1, H)
        ctx_inputs = ctx_inputs.expand_as(tok_inputs)  # (N, Tdec, H)

        dec_inputs = torch.cat([tok_inputs, ctx_inputs], dim=2)  # (N, T_dec, 2H)
        outputs, hiddens = (
            decoder(dec_inputs) if h0 is None else decoder(dec_inputs, h0)
        )

        return outputs, hiddens  # (N, T_dec, H), (D*num_layers, H)


# :param attn_inputs: intents embeddings (before multi-modal processing) of each trajectory step.
#                     Tensor of shape (B, T_traj*2, H)   (*2 due to concat of Done and Todo)
# :param attn_mask: mask to prevent attending to future intentions
#                   Tensor of shape (B, T_traj, T_traj*2)
