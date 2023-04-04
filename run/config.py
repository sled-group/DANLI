import os
from sacred import Ingredient
from sacred.settings import SETTINGS
from datetime import datetime

data_ingredient = Ingredient("data_args")
exp_ingredient = Ingredient("exp_args")
model_ingredient = Ingredient("model_args")
train_ingredient = Ingredient("train_args")
eval_ingredient = Ingredient("eval_args")

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


@data_ingredient.config
def cfg_data():
    # # name of the output dataset
    raw_data_dir = os.environ["DANLI_DATA_DIR"]
    out_data_dir = os.path.join(os.environ["DANLI_DATA_DIR"], "processed_20220610")
    ithor_assets_dir = os.path.join(os.environ["DANLI_ROOT_DIR"], "ithor_assets")
    vis_feat_dir = os.path.join(os.environ["DANLI_DATA_DIR"], "vis_feats")

    # number of processes to run the data processing in (0 for main thread)
    num_workers = 20
    # debug run with only 10 games
    debug = False
    # what to process. Note that 'preprocess' must be done at first.
    data_processing_jobs = [
        "preprocess",
        "encode_with_intent",
        "encode_without_intent",
        "encode_ori_baseline",
    ]
    # prepare training instances for the navigation module
    prepare_navi = True
    # whether to write readable data files
    write_readables = False

    # Input truncation
    # (384 + 128 *3 (frame/action/intention) = 768)
    # (288 +  96 *3 (frame/action/intention) = 576)
    input_trunc = {"dialog": 288, "trajectory": 96}


@exp_ingredient.config
def cfg_exp():
    # HIGH-LEVEL MODEL SETTINGS
    # experiment starting time
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # experiment type
    exp_type = "navi"  # ['edh', 'tfd', 'game', 'navi', 'lm']
    # train edh with loss of only future actions or the entire trajectory
    edh_loss_type = "entire_traj"  # ['future_only', 'entire_traj']
    if exp_type != "edh":
        edh_loss_type = None

    # experiment directory
    exp_dir_base = os.path.join(os.environ["DANLI_DATA_DIR"], "experiments")
    exp_dir = exp_dir_base + "%s-%s" % (start_time, exp_type)

    # model to use
    model = "transformer"
    # which device to use
    device = "cuda"
    # number of data loading workers (0 for main thread)
    num_loader_workers = 4


@model_ingredient.config
def cfg_model():
    # TRANSFORMER settings
    # size of transformer embeddings
    demb = 768
    # number of heads in multi-head attention
    encoder_heads = 12
    # number of layers in transformer encoder
    encoder_layers = 6
    # shape of visual features obtained from pretrained vision model
    visual_tensor_shape = (512, 7, 7)
    # whether to add intention prediction or not
    add_intention = True
    # whether to use visual frames as inputs
    enable_vision = True
    # whether to use dialog as inputs
    enable_lang = True
    # whether to use action history as inputs
    enable_action_history = True
    #
    action_history_length = "all"  # integer or 'all'
    #
    if not enable_vision and not enable_action_history:
        raise ValueError(
            "Should enable at least one of `visual frames`"
            " or `action history` to set up action prediction heads"
        )
    # causal attention
    use_causal_attn = True
    # which encoder to use for language encoder (by default no encoder)
    enc_lang = {
        "pos_tok_enc": True,
        "role_enc": True,
    }
    # which encoder to use for intention encoder
    enc_intent = {
        "type": "bi_gru",  # ['bi_gru', 'gru', 'pool']
        "layers": 1,  # only for GRUs
    }

    # transformer encodings
    enc_mm = {
        # use positional encoding
        "pos": True,
        # use learned modality (langauge/visual/action/intention) encoding
        "modality": True,
    }

    dec_intent = {
        "type": "gru",  # ['gru']
        "layers": 1,
    }

    # DROPOUTS
    dropout = {
        # dropout rate for dialog inputs
        "lang": 0.3,
        # dropout rate for action and object inputs
        "action": 0.3,
        # dropout rate for intention inputs
        "intent": 0.3,
        # dropout rate for Resnet feats
        "vis": 0.3,
        # dropout rate for processed lang and visual embeddings
        # 'emb': 0.0,
        # transformer model specific dropouts
        "transformer": {
            # dropout for transformer input
            "input": 0.3,
            # dropout for transformer encoder
            "encoder": 0.3,
        },
    }


@train_ingredient.config
def cfg_train():
    # GENERAL TRANING SETTINGS
    # random seed
    seed = 0
    # load a checkpoint from a previous epoch (if available)
    resume = True
    # whether to print execution time for different parts of the code
    profile = False

    # HYPER PARAMETERS
    # batch size
    batch_size = 48
    # fast epoch for debug
    fast_epoch = False
    # number of epochs
    epochs = 50
    # optimizer type, must be in ('adam', 'adamw')
    optimizer = "adamw"
    # L2 regularization weight
    weight_decay = 0.2
    # learning rate settings
    lr = {
        # learning rate initial value
        "init": 5e-4,
        # lr scheduler type: {'StepLR'}
        "profile": "StepLR",
        # (LINEAR PROFILE) num epoch to adjust learning rate
        "decay_epoch": 3,
        # (LINEAR PROFILE) scaling multiplier at each milestone
        "decay_scale": 0.5,
        # warm up period length in steps
        "warmup_steps": 200,
    }
    loss = {
        # loss weight
        "weights": {
            "action_output": 1.0,
            "arg1_output": 1.0,
            "arg2_output": 1.0,
            "intent_done_output": 1.0,
            "intent_todo_output": 1.0,
            "dialog_output": 1.0,
        }
    }
