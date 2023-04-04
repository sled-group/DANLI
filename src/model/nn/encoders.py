import os
import torch
import numpy as np
from torch import nn

from .encodings import LearnedEncoding, PosEncoding
from ..utils import model_util


class EncoderDialog(nn.Module):
    def __init__(self, args, embedder):
        """
        transformer encoder for language inputs
        """
        super().__init__()
        self.embedder = embedder
        self.max_utter_len = 100 if args.exp_type != "tfd" else 1024
        self.role_num = 2

        # Learnable positional encoding within each utterance (token position)
        self.enc_pos_tok = (
            LearnedEncoding(args.demb, self.max_utter_len)
            if args.enc_lang["pos_tok_enc"]
            else None
        )

        # Learnable role encoding (user/bot)
        self.enc_role = (
            LearnedEncoding(args.demb, self.role_num + 1)
            if args.enc_lang["role_enc"]
            else None
        )

        self.enc_dropout = nn.Dropout(args.dropout["lang"], inplace=True)
        self.enc_layernorm = nn.LayerNorm(args.demb)

    def forward(self, lang_input, tok_pos_input, role_input):
        """
        embed dialog inputs, add dailog role and positional encoding for each utterance
        """
        # pad the input language sequences and embed them with a linear layer
        lang_emb = self.embedder(lang_input)
        # add positional encodings, apply layernorm and dropout
        if self.enc_pos_tok:
            lang_emb = self.enc_pos_tok(lang_emb, tok_pos_input)
        if self.enc_role:
            lang_emb = self.enc_role(lang_emb, role_input)
        lang_emb = self.enc_dropout(lang_emb)
        lang_emb = self.enc_layernorm(lang_emb)
        return lang_emb


class EncoderVision(nn.Module):
    """
    a few conv layers to flatten features that come out of ResNet
    """

    def __init__(self, args):
        super().__init__()

        self.input_shape = args.visual_tensor_shape
        self.output_size = args.demb
        if self.input_shape[0] == -1:
            self.input_shape = self.input_shape[1:]

        layers, activation_shape = self.init_cnn(
            self.input_shape, channels=[256, 64], kernels=[1, 1], paddings=[0, 0]
        )
        layers += [
            nn.Flatten(),
            nn.Linear(np.prod(activation_shape), self.output_size),
            nn.Dropout(args.dropout["vis"], inplace=True),
        ]
        self.layers = nn.Sequential(*layers)

    def init_cnn(self, input_shape, channels, kernels, paddings):
        layers = []
        planes_in, spatial = input_shape[0], input_shape[-1]
        for planes_out, kernel, padding in zip(channels, kernels, paddings):
            # do not use striding
            stride = 1
            layers += [
                nn.Conv2d(
                    planes_in,
                    planes_out,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(planes_out),
                nn.ReLU(inplace=True),
            ]
            planes_in = planes_out
            spatial = (spatial - kernel + 2 * padding) // stride + 1
        activation_shape = (planes_in, spatial, spatial)
        return layers, activation_shape

    def forward(self, vis_feats):
        """[summary]


        :param vis_feats: pre-encoded frame features of shape (B, T, *input_shape)
        :return: [description]
        """
        B, T = vis_feats.shape[0], vis_feats.shape[1]
        vis_feats = vis_feats.view(-1, *self.input_shape)

        activation = self.layers(vis_feats)
        return activation.view(B, T, -1)


class EncoderAction(nn.Module):
    def __init__(self, args, embedder):
        """
        encoder for action and object inputs
        """
        super().__init__()
        self.embedder = embedder

        # predicate type (action, arg1 or arg2) embedding
        self.enc_type = LearnedEncoding(args.demb, 3)

        self.enc_dropout = nn.Dropout(args.dropout["action"], inplace=True)
        self.enc_layernorm = nn.LayerNorm(args.demb)

    def forward(self, action_input, arg1_input=None, arg2_input=None):
        """
        embed action input (word indexes of phrase)
        """
        # pad the input action and object sequences and embed them with a linear layer
        action_emb = self.embedder(action_input).mean(dim=-2)
        type_idx = torch.tensor(0, device=action_emb.device)
        action_emb = self.enc_type(action_emb, type_idx)

        if arg1_input is not None:
            arg1_emb = self.embedder(arg1_input).mean(dim=-2)
            type_idx = torch.tensor(1, device=action_emb.device)
            action_emb += self.enc_type(arg1_emb, type_idx)

        if arg2_input is not None:
            arg2_emb = self.embedder(arg2_input).mean(dim=-2)
            type_idx = torch.tensor(2, device=action_emb.device)
            action_emb += self.enc_type(arg2_emb, type_idx)

        # apply layernorm and dropout
        action_emb = self.enc_dropout(action_emb)
        action_emb = self.enc_layernorm(action_emb)
        return action_emb


class EncoderNaviGoal(nn.Module):
    def __init__(self, args, embedder):
        """
        encoder for navigation goal
        """
        super().__init__()
        self.embedder = embedder

        self.enc_dropout = nn.Dropout(args.dropout["action"], inplace=True)
        self.enc_layernorm = nn.LayerNorm(args.demb)

    def forward(self, goal_input):
        """
        embed action input (word indexes of phrase)
        """
        # embed goals
        goal_emb = self.embedder(goal_input)

        # apply layernorm and dropout
        goal_emb = self.enc_dropout(goal_emb)
        goal_emb = self.enc_layernorm(goal_emb)
        return goal_emb


class EncoderIntent(nn.Module):
    def __init__(self, args, embedder):
        """
        transformer encoder for intention inputs
        """
        super().__init__()
        self.embedder = embedder

        self.encoder_type = args.enc_intent["type"]
        if self.encoder_type == "pool":
            self.encoder = lambda x: x.mean(dim=-2)
        elif "gru" in self.encoder_type:
            self.encoder = nn.GRU(
                input_size=args.demb,
                hidden_size=args.demb,
                num_layers=args.enc_intent["layers"],
                bidirectional="bi_" in self.encoder_type,
                batch_first=True,
            )
        else:
            raise NotImplementedError(
                "%s is not a valid intention encoder type!" % self.encoder_type
            )

        self.enc_dropout = nn.Dropout(args.dropout["intent"], inplace=True)
        self.enc_layernorm = nn.LayerNorm(args.demb)

        # Learnable intention type encoding (todo/done)
        self.enc_intent_type = LearnedEncoding(args.demb, 2)

    def forward(self, intent_done, intent_todo):
        """
        embed intention input (word indexes of phrase)
        """
        # embed intentions (done/todo) and add a encoding to distinguish them
        intent_done_emb = self.embedder(intent_done)
        intent_done_emb = self.encode(intent_done_emb)
        done_pos = torch.tensor(0, device=intent_done.device)
        intent_done_emb = self.enc_intent_type(intent_done_emb, done_pos)

        intent_todo_emb = self.embedder(intent_todo)
        intent_todo_emb = self.encode(intent_todo_emb)
        todo_pos = torch.tensor(1, device=intent_todo.device)
        intent_todo_emb = self.enc_intent_type(intent_todo_emb, todo_pos)

        intent_emb = intent_done_emb + intent_todo_emb
        intent_emb = self.enc_dropout(intent_emb)
        intent_emb = self.enc_layernorm(intent_emb)
        return intent_emb

    def encode(self, input_emb):
        if self.encoder_type == "pool":
            return self.encoder(input_emb)
        assert "gru" in self.encoder_type

        # input_emb shape: B x T x Tperd x demb
        B, T, Tpred, demb = input_emb.shape
        input_emb = input_emb.view(-1, Tpred, demb)
        out, h = self.encoder(input_emb)  # h: 2*num_layer x (B*T) x demb
        return h.sum(dim=0).view(B, T, -1)


class MultiModalTransformer(nn.Module):
    def __init__(self, args):
        """
        multi-modal transformer encoder to fuse inputs in different modalities:
        (1) language (2) visual frames (3) actions and (4) intentions
        """
        super(MultiModalTransformer, self).__init__()
        self.use_causal_attention = args.use_causal_attn
        self.head_num = args.encoder_heads
        self.pad_idx = 0
        self.ordering_pad_idx = 1023

        # transofmer layers
        encoder_layer = nn.TransformerEncoderLayer(
            args.demb,
            args.encoder_heads,
            args.demb,
            args.dropout["transformer"]["encoder"],
            batch_first=True,
        )
        self.enc_transformer = nn.TransformerEncoder(encoder_layer, args.encoder_layers)

        # positional encodings (provide causal ordering)
        # self.enc_position = PosEncoding(args.demb) if args.enc_mm['pos'] else None
        if args.enc_mm["pos"]:
            self.enc_position = LearnedEncoding(
                args.demb, self.ordering_pad_idx + 1, padding_idx=self.ordering_pad_idx
            )

        # modality encodings
        self.modality_mapping = {"lang": 1, "vis": 2, "action": 3, "intent": 4}
        self.enc_modality = (
            LearnedEncoding(args.demb, 5) if args.enc_mm["modality"] else None
        )

        # layer norm and dropout before feeding to the transformers
        self.enc_dropout = nn.Dropout(
            args.dropout["transformer"]["input"], inplace=True
        )
        self.enc_layernorm = nn.LayerNorm(args.demb)

    def forward(
        self, input_embeddings, input_lengths, input_orderings=None, inputs=None
    ):
        """Fuse embedded inputs of each modality using a transformer encoder

        :param input_embeddings: list of tuples in the form of (modality_name, embedding_tensor)
        :param input_lengths: list of input lengths
        :kwarg input_orderings: list of input orderings
        :return: [description]
        """

        emb_inputs = []
        for modality_name, emb in input_embeddings:
            if self.enc_modality is not None:
                modality_id = self.modality_mapping[modality_name]
                modality_id = torch.tensor(modality_id, device=emb.device)
                emb = self.enc_modality(emb, modality_id)
            emb_inputs.append(emb)

        emb_inputs = torch.cat(emb_inputs, dim=1)

        if input_orderings is not None:
            input_orderings = torch.cat(input_orderings, dim=1)
            input_orderings[input_orderings == self.pad_idx] = self.ordering_pad_idx
        """
        Here we toggle the causal mask matrix (shape: B*head_num x T x T). 
        For each self-attention causal mask matrix of shape T x T, we change
        all values in padding index rows from  "True" to "False". 
        By doing this we avoid outputing "NAN" in the padding positions. 
        Note that "mask_pad" is still required since we do not want to this to
        affect our outputs, and the role of the padding mask is to make the 
        non-padding positions avoid attending to the padded postiions. 

        To my understanding this bug is due to the inappropriate implementation 
        when 3D attention mask is used together with padding mask. 
        """

        # add positional encoding (causal information)
        if self.enc_position is not None:
            assert input_orderings is not None
            emb_inputs = self.enc_position(emb_inputs, input_orderings)
        # apply dropout and layer normalizaion
        emb_inputs = self.enc_dropout(emb_inputs)
        emb_inputs = self.enc_layernorm(emb_inputs)

        # create a padding mask for batch input
        mask_pad = model_util.get_pad_mask(input_lengths, emb_inputs.device)

        # create a attention mask
        if self.use_causal_attention:
            # causal: tokens at order t cannot see tokens at order > t
            assert input_orderings is not None
            # print('input_orderings', input_orderings)
            mask_causal = model_util.get_causal_mask(input_orderings, emb_inputs.device)
            mask_causal = mask_causal.repeat_interleave(self.head_num, dim=0)
        else:
            # non-causal: allow every token to attend to all others
            mask_causal = None

        # encode the inputs
        output = self.enc_transformer(emb_inputs, mask_causal, mask_pad)
        return output, mask_pad
