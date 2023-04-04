import os
import copy
import ujson as json
import numpy as np
from tqdm import tqdm

import pprint

# from pprint import pprint
from multiprocessing import Pool

from .utils.ai2thor_utils import get_obj_type_from_oid


DATA_INSTANCE_FIELDS = [
    "dialog_input",
    "dialog_output",
    "dialog_role",
    "frames",
    "action_input",
    "action_output",
    "arg1_input",
    "arg1_output",
    "arg2_input",
    "arg2_output",
    "intent_done_input",
    "intent_done_output",
    "intent_todo_input",
    "intent_todo_output",
    "ordering_dialog",
    "ordering_tokens",
    "ordering_frames",
    "ordering_action",
    "ordering_intent",
]

NAVI_DATA_INSTANCE_FIELD = [
    "goal",
    "frames",
    "postures",
    "action_input",
    "action_output",
]

# (rotation%360)/90,
# 'horizon': horizon/30,


def encode_navi(game_preprocessed, vocabs):
    ord2id = vocabs["input_vocab_word2id"]
    id2word = {v: k for k, v in vocabs["input_vocab_word2id"].items()}
    action2id_navi = vocabs["output_vocab_action_navi"]
    object_type2id = vocabs["output_vocab_object"]

    beg_idx = action2id_navi["[BEG]"]
    end_idx = action2id_navi["[END]"]

    all_low_actions = game_preprocessed["low_actions"]
    all_high_actions = game_preprocessed["high_actions"]

    game_id = game_preprocessed["game_id"]

    hidx = 0
    prev_bot_action = None

    navi_hidx = -1
    post_action_frame = None

    navi_instances = []

    instance = None

    hidx_to_lidx_mapping = {}
    for lidx, a in enumerate(all_low_actions):
        _, hidx, role, success, atype, aname, arg, curr_time, pos = a
        if hidx not in hidx_to_lidx_mapping:
            hidx_to_lidx_mapping[hidx] = []
        hidx_to_lidx_mapping[hidx].append(lidx)

    for high_act in all_high_actions:
        hidx, role, atype, aname, arg, arg_r, curr_time, _, intent = high_act

        if role != "BOT" or atype != "navi":
            continue

        instance = {"game_id": game_id, "hidx": hidx}
        arg2, arg1 = arg.split()  # arg1 is the target we want to find
        o1, o2 = get_obj_type_from_oid(arg1), get_obj_type_from_oid(arg2)
        instance["goal"] = [object_type2id[o1]] #, object_type2id[o2]]
        instance["frames"] = []
        instance["action_input"] = [beg_idx]

        for low_act_idx in hidx_to_lidx_mapping[hidx]:
            low_act = all_low_actions[low_act_idx]
            lidx, hidx, role, success, atype, aname, arg, curr_time, pos = low_act
            if aname not in action2id_navi:
                continue
            instance["frames"].append(curr_time)
            instance["action_input"].append(action2id_navi[aname])

        if low_act_idx + 1 < len(all_low_actions):
            post_action_frame = all_low_actions[low_act_idx + 1][7]
        else:
            post_action_frame = "end"
        instance["frames"].append(post_action_frame)
        instance["action_output"] = instance["action_input"][1:] + [end_idx]

        if len(instance["frames"]) >= 3:
            navi_instances.append(instance)

    # ############## sanity check ######################
    # pp = pprint.PrettyPrinter(width=200)
    # v = {n: {v: k for k, v in vocabs[n].items()} for n in vocabs}
    # v_dec = v['output_vocab_action_navi']
    # v_o = v['output_vocab_object']
    # for ins in navi_instances:
    #     assert len(ins['frames']) == len(ins['action_input'])
    #     assert len(ins['action_output']) == len(ins['action_input'])

    #     ainp = ins['action_input']
    #     aout = ins['action_output']
    #     vis = ins['frames']

    #     decoded = [(i, v_dec[i], aout[idx], v_dec[aout[idx]], vis[idx]) for idx, i in enumerate(ainp)]
    #     print(ins['game_id'], ins['hidx'], v_o[ins['goal'][0]], v_o[ins['goal'][1]])
    #     pp.pprint(decoded)

    return navi_instances


def encode_game(game_preprocessed, vocabs, add_intention, edh_info={}):
    word2id = vocabs["input_vocab_word2id"]
    id2word = {v: k for k, v in vocabs["input_vocab_word2id"].items()}
    action2id_high = vocabs["output_vocab_action_high"]
    object_type2id = vocabs["output_vocab_object"]
    intent2id = vocabs["output_vocab_intent"]
    pred2lang = vocabs["predicate_to_language"]

    game = {"game_id": game_preprocessed["game_id"]}
    game.update({k: [] for k in DATA_INSTANCE_FIELDS})

    if edh_info:
        game["edh_file"] = edh_info["file_name"]

    # if game_preprocessed['game_id'] == '8f7791c488b51c98_4c51':
    #     print(edh_info)
    #     print(pred_start_idx, pred_end_idx)

    role_map = {"USR": 1, "BOT": 2}  # leave 0 for padding

    all_low_actions = game_preprocessed["low_actions"]
    all_high_actions = game_preprocessed["high_actions"]
    pred_start_idx = edh_info.get("pred_start_high_idx", 0)
    pred_end_idx = edh_info.get("pred_end_high_idx", len(all_high_actions))

    hidx = 0
    ordering_step = 1  # leave 0 for padding
    max_intent_length = 0
    prev_bot_action = None
    user_utter_token_ids = None

    # add start tokens
    game["action_input"].append([word2id["[BEG_TRAJ]"]])
    game["arg1_input"].append([word2id["[PAD]"]])
    game["arg2_input"].append([word2id["[PAD]"]])
    game["ordering_action"].append(ordering_step)

    # process trajectory
    for curr_action in all_high_actions:

        hidx, role, atype, aname, arg, arg_r, curr_time, _, intent = curr_action

        # For EDH instances, the trajectory should terminate at pred_end_idx
        if hidx >= pred_end_idx:
            break

        # skip empty bot dialog actions
        if role == "BOT" and atype == "dial" and arg == "":
            continue

        if role == "USR":  # collect user utterance as observations
            assert atype == "dial"
            utter_tokens = arg.split()
            if len(utter_tokens) != 0:  # empty utterance
                if user_utter_token_ids is None:
                    user_utter_token_ids = [word2id.get(t, 1) for t in utter_tokens]
                else:
                    token_ids = [word2id.get(t, 1) for t in utter_tokens]
                    if id2word[user_utter_token_ids[-1]].isalpha():
                        token_ids = [word2id["."]] + token_ids
                    user_utter_token_ids.extend(token_ids)

        else:  # add a BOT action to the trajectory
            """For each bot action, we first add the observation frame.
            Then we check if the user says anything, if they do, add it.
            If intention is modeled, we add the intention.
            Finally add the action and corresponding object argument."""

            # add visual observation
            game["frames"].append(str(curr_time))
            game["ordering_frames"].append(ordering_step)
            ordering_step += 1

            # add language observation
            if user_utter_token_ids is not None:
                utter_len = len(user_utter_token_ids) + 1
                game["dialog_input"].extend([word2id["[USR]"]] + user_utter_token_ids)
                game["dialog_output"].extend(
                    user_utter_token_ids + [word2id["[EOS_USR]"]]
                )
                game["dialog_role"].extend([role_map["USR"]] * utter_len)
                game["ordering_tokens"].extend(range(1, utter_len + 1))
                game["ordering_dialog"].extend([ordering_step] * utter_len)
                user_utter_token_ids = None

                ordering_step += 1

            # add intention (mental state)
            if add_intention:
                intent_done = (
                    prev_bot_action[8][3] if prev_bot_action is not None else ""
                )
                intent_todo = curr_action[8][2]
                beg1, beg2 = word2id["[BEG_DONE]"], word2id["[BEG_TODO]"]
                end1, end2 = intent2id["[END_DONE]"], intent2id["[END_TODO]"]
                game["intent_done_output"].append(
                    [intent2id[w] for w in intent_done.split()] + [end1]
                )
                game["intent_todo_output"].append(
                    [intent2id[w] for w in intent_todo.split()] + [end2]
                )
                game["intent_done_input"].append(
                    [beg1] + [word2id.get(w, 1) for w in intent_done.split()]
                )
                game["intent_todo_input"].append(
                    [beg2] + [word2id.get(w, 1) for w in intent_todo.split()]
                )

                intent_length = max(
                    len(game["intent_todo_input"][-1]),
                    len(game["intent_done_input"][-1]),
                )
                if intent_length > max_intent_length:
                    max_intent_length = intent_length

                game["ordering_intent"].append(ordering_step)
                ordering_step += 1

            # add action
            if atype == "dial":
                utter_tokens = arg.split()
                assert len(utter_tokens) != 0  # empty utterance
                utter_token_ids = [word2id.get(t, 1) for t in utter_tokens]
                utter_len = len(utter_token_ids) + 1
                game["dialog_input"].extend([word2id["[BOT]"]] + utter_token_ids)
                game["dialog_output"].extend(utter_token_ids + [word2id["[EOS_BOT]"]])
                game["dialog_role"].extend([role_map["BOT"]] * utter_len)
                game["ordering_tokens"].extend(range(1, utter_len + 1))
                game["ordering_dialog"].extend([ordering_step] * utter_len)

                arg1_otype = arg2_otype = "None"

            else:
                arg2_otype = "None"
                if aname == "Goto":
                    arg2, arg1 = arg.split()
                    arg2_otype = get_obj_type_from_oid(arg2)
                elif aname == "Pickup":
                    arg1 = arg.split()[0]
                elif aname in ["Place", "Pour"]:
                    arg1 = arg.split()[1]
                else:
                    arg1 = arg
                arg1_otype = get_obj_type_from_oid(arg1)

            game["action_output"].append(action2id_high[aname])
            game["arg1_output"].append(object_type2id[arg1_otype])
            game["arg2_output"].append(object_type2id[arg2_otype])

            game["action_input"].append(
                [word2id.get(w, 1) for w in pred2lang[aname].split()]
            )
            game["arg1_input"].append(
                [word2id.get(w, 1) for w in pred2lang[arg1_otype].split()]
            )
            game["arg2_input"].append(
                [word2id.get(w, 1) for w in pred2lang[arg2_otype].split()]
            )

            game["ordering_action"].append(ordering_step)
            # Note: ordering_step should not step forward here since the visual observation
            # of the action_input has not been added yet.

            prev_bot_action = curr_action

            # for edh instances, record the start point of predictions
            if hidx == pred_start_idx:
                game["pred_start_idx"] = len(game["action_input"]) - 2
            # minus 2 because we use the previous action as input of predicting the target action

            # print('[%s %03d %s] %s' % (str(curr_action[6])[:6], hidx, curr_action[1], curr_action[5]))
            """ Players may say something during a navigation process. Instead of using the
                exact frame of those utterances, we use the first starting frame of the navigation 
            """

    """ add the last intention and the end-of-task action """
    hidx += 1
    last_obs = str(all_high_actions[hidx][6]) if hidx < len(all_high_actions) else "end"
    game["frames"].append(last_obs)
    game["ordering_frames"].append(ordering_step)
    ordering_step += 1

    if add_intention:
        intent_done = prev_bot_action[8][3] if prev_bot_action is not None else ""
        intent_todo = curr_action[8][2]
        beg1, beg2 = word2id["[BEG_DONE]"], word2id["[BEG_TODO]"]
        end1, end2 = intent2id["[END_DONE]"], intent2id["[END_TODO]"]
        game["intent_done_input"].append(
            [beg1] + [word2id.get(w, 1) for w in intent_done.split()]
        )
        game["intent_todo_input"].append(
            [beg2] + [word2id.get(w, 1) for w in intent_todo.split()]
        )
        game["intent_done_output"].append(
            [intent2id[w] for w in intent_done.split()] + [end1]
        )
        game["intent_todo_output"].append(
            [intent2id[w] for w in intent_todo.split()] + [end2]
        )
        game["ordering_intent"].append(ordering_step)
        ordering_step += 1

        intent_length = max(
            len(game["intent_done_input"][-1]), len(game["intent_done_input"][-1])
        )
        if intent_length > max_intent_length:
            max_intent_length = intent_length

        game["max_intent_length"] = max_intent_length

    game["action_output"].append(action2id_high["[END]"])
    game["arg1_output"].append(object_type2id["None"])
    game["arg2_output"].append(object_type2id["None"])

    """ Prediction alignment: 
        action_input -> intent_output
        intent_input -> action_output   """

    return game


def encode_game_baseline(game_preprocessed, vocabs, add_intention, edh_info={}):
    word2id = vocabs["input_vocab_word2id"]
    action2id_high = vocabs["output_vocab_action_all"]
    object_type2id = vocabs["output_vocab_object"]

    game = {
        "game_id": game_preprocessed["game_id"],
        "frames": [],
        "dialog_input": [],
        "dialog_output": [],
        "action_input": [action2id_high["[BEG]"]],
        "action_output": [],
        "arg_output": [],
        "pred_start_idx": 0,
    }
    all_low_actions = game_preprocessed["low_actions"]
    beg = edh_info.get("pred_start_low_idx", 0)
    end = edh_info.get("pred_end_low_idx", len(all_low_actions))
    if edh_info:
        game["edh_file"] = edh_info["file_name"]

    for a in all_low_actions[: end + 1]:
        lidx, hidx, role, success, atype, aname, arg, curr_time, pos = a

        if lidx == beg:
            game["pred_start_idx"] = len(game["frames"])

        if atype == "dial":
            utter_tokens = arg.split()
            if len(utter_tokens) != 0:  # non-empty utterance
                dial_input = ["[%s]" % role] + utter_tokens
                dial_output = utter_tokens + ["[EOS_%s]" % role]
                game["dialog_input"].extend(([word2id.get(t, 1) for t in dial_input]))
                game["dialog_output"].extend(([word2id.get(t, 1) for t in dial_output]))

        if success and role == "BOT":
            game["frames"].append(str(curr_time))
            try:
                game["action_input"].append(action2id_high[aname])
            except:
                print(a, game_preprocessed["game_id"])
                quit()
            game["action_output"].append(action2id_high[aname])

            if aname == "Text":
                arg = None
            elif isinstance(arg, str) and " " not in arg:
                pass
            elif aname == "Pickup":
                arg = arg.split()[0]
            elif aname in ["Place", "Pour"]:
                arg = arg.split()[1]
            arg_otype = get_obj_type_from_oid(arg) if arg is not None else "None"
            game["arg_output"].append(object_type2id[arg_otype])

    last_obs = "end"
    for a in all_low_actions[hidx:]:
        lidx, hidx, role, success, atype, aname, arg, time, pos = a
        if success and role == "BOT":
            last_obs = time
            break

    game["frames"].append(last_obs)
    game["action_output"].append(action2id_high["[END]"])
    game["arg_output"].append(object_type2id["None"])

    return game


def prepare_edh_tfd_from_games(game_id, split, vocabs, arg_dict):
    processed_game_file = os.path.join(
        arg_dict["save_path"], "processed_gamefiles", "%s.json" % game_id
    )
    with open(processed_game_file, "r", encoding="utf-8") as f:
        game_preprocessed = json.load(f)

    # process the entire game
    # game_encoded = encode_game(game_preprocessed, vocabs, arg_dict['add_intention'])
    f = encode_game_baseline if arg_dict["process_baseline"] else encode_game
    game_encoded = f(game_preprocessed, vocabs, arg_dict["add_intention"])
    game_encoded["split"] = split
    # tfd_file = os.path.join(arg_dict['save_path'], 'tfd_instances', '%s.tfd.json' % game_id)
    # with open(tfd_file, 'r', encoding='utf-8') as f:
    #     tfd_raw = json.load(f)

    if not arg_dict["process_baseline"]:
        decoded = sanity_check(game_encoded, vocabs, arg_dict["add_intention"])
    # else:
    #     print(game_encoded['game_id'])
    #     print('frame length:', len(game_encoded['frames']))
    #     print('action input length:', len(game_encoded['action_input']))
    #     print('action output length:', len(game_encoded['action_output']))
    #     print('object output length:', len(game_encoded['arg_output']))
    #     print('pred_start_idx', game_encoded['pred_start_idx'])

    # intent_str = '_with_intention' if arg_dict['add_intention'] else ''
    # save_dir = os.path.join(arg_dict['save_path'], 'encoded_games%s' % intent_str)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # with open(os.path.join(save_dir, '%s.json' % game_id), 'w') as f:
    #     json.dump(decoded, f, indent=4)

    # prepare the tfd instance
    tfd_encoded = None
    tfd_file = os.path.join(
        arg_dict["data_path"], "tfd_instances", split, "%s.tfd.json" % game_id
    )
    if os.path.exists(tfd_file):
        tfd_encoded = copy.deepcopy(game_encoded)
        dial_len = len(tfd_encoded["dialog_input"])
        tfd_encoded["ordering_dialog"] = [0] * dial_len
        tfd_encoded["ordering_tokens"] = list(range(dial_len))
        tfd_encoded["split"] = split

    # prepare the edh instances
    edh_encoded_all = []
    if "edh_info" in game_preprocessed:
        edh_info_all = game_preprocessed["edh_info"]
        for edh_info in edh_info_all:
            f = encode_game_baseline if arg_dict["process_baseline"] else encode_game
            edh_encoded = f(
                game_preprocessed, vocabs, arg_dict["add_intention"], edh_info
            )
            if "pred_start_idx" not in edh_encoded:
                print("bad instance:", split, game_id, edh_info["file_name"])
                continue
            edh_encoded["split"] = split
            edh_encoded_all.append(edh_encoded)

    navi_encoded = None
    # prepare navigation instances
    if arg_dict["prepare_navi"]:
        navi_encoded = encode_navi(game_preprocessed, vocabs)

    return game_id, game_encoded, tfd_encoded, edh_encoded_all, navi_encoded


def prepare_data_mp_wrapper(args):
    return prepare_edh_tfd_from_games(*args)


def sanity_check(encoded_game, vocabs, add_intentions):

    # image_folder = "/gpfs/accounts/chaijy_root/chaijy1/Yichi_Jiahao_shared/teach/teach-dataset/images"
    # image_folder = "/home/ubuntu/teach-dataset/images"
    image_folder = "/data/simbot/teach-dataset/images"

    decoded = {}
    v = {n: {v: k for k, v in vocabs[n].items()} for n in vocabs}
    v_in = v["input_vocab_word2id"]
    v_oa = v["output_vocab_action_high"]
    v_oi = v["output_vocab_intent"]
    v_oo = v["output_vocab_object"]
    assert len(encoded_game["action_input"]) == len(encoded_game["ordering_action"])
    assert len(encoded_game["arg1_input"]) == len(encoded_game["ordering_action"])
    assert len(encoded_game["dialog_input"]) == len(encoded_game["ordering_dialog"])
    assert len(encoded_game["dialog_input"]) == len(encoded_game["ordering_tokens"])
    assert len(encoded_game["action_input"]) == len(encoded_game["frames"])
    ord_a = encoded_game["ordering_action"]
    ord_f = encoded_game["ordering_frames"]
    ord_d = encoded_game["ordering_dialog"]
    ord_t = encoded_game["ordering_tokens"]

    decoded["dialog_input"] = [
        (i, v_in[i], ord_d[idx], ord_t[idx])
        for idx, i in enumerate(encoded_game["dialog_input"])
    ]
    decoded["dialog_output"] = [
        (i, v_in[i], ord_d[idx], ord_t[idx])
        for idx, i in enumerate(encoded_game["dialog_output"])
    ]

    split, gid = encoded_game["split"], encoded_game["game_id"]
    decoded["frames"] = [
        (
            i,
            os.path.exists(
                os.path.join(image_folder, split, gid, "driver.frame.%s.jpeg" % i)
            ),
            ord_f[idx],
        )
        for idx, i in enumerate(encoded_game["frames"])
    ]
    for i, img, _ in decoded[
        "frames"
    ]:  # check whether all the images exist in the train/valid dataset
        assert (
            img or "test" in encoded_game["split"]
        ), "Missing image %s in game %s!" % (i, encoded_game["game_id"])

    decoded["action_input"] = [
        [(i, v_in[i], ord_a[idx]) for i in j]
        for idx, j in enumerate(encoded_game["action_input"])
    ]
    decoded["arg1_input"] = [
        [(i, v_in[i], ord_a[idx]) for i in j]
        for idx, j in enumerate(encoded_game["arg1_input"])
    ]
    decoded["arg2_input"] = [
        [(i, v_in[i], ord_a[idx]) for i in j]
        for idx, j in enumerate(encoded_game["arg2_input"])
    ]
    if add_intentions:
        assert len(encoded_game["intent_done_input"]) == len(
            encoded_game["ordering_intent"]
        ), encoded_game["game_id"]
        assert len(encoded_game["intent_done_input"]) == len(
            encoded_game["action_output"]
        ), encoded_game["game_id"]
        assert len(encoded_game["intent_done_output"]) == len(
            encoded_game["action_input"]
        ), encoded_game["game_id"]
        # except:
        #     pp = pprint.PrettyPrinter(width=200)
        #     pp.pprint(encoded_game)
        ord_i = encoded_game["ordering_intent"]
        decoded["intent_done_input"] = [
            [(i, v_in[i], ord_i[idx]) for i in j]
            for idx, j in enumerate(encoded_game["intent_done_input"])
        ]
        decoded["intent_todo_input"] = [
            [(i, v_in[i], ord_i[idx]) for i in j]
            for idx, j in enumerate(encoded_game["intent_todo_input"])
        ]
        decoded["intent_done_output"] = [
            [(i, v_oi[i], ord_a[idx]) for i in j]
            for idx, j in enumerate(encoded_game["intent_done_output"])
        ]
        decoded["intent_todo_output"] = [
            [(i, v_oi[i], ord_a[idx]) for i in j]
            for idx, j in enumerate(encoded_game["intent_todo_output"])
        ]

    ord_oa = ord_a if not add_intentions else ord_i
    decoded["action_output"] = [
        (i, v_oa[i], ord_oa[idx]) for idx, i in enumerate(encoded_game["action_output"])
    ]
    decoded["arg1_output"] = [
        (i, v_oo[i], ord_oa[idx]) for idx, i in enumerate(encoded_game["arg1_output"])
    ]
    decoded["arg2_output"] = [
        (i, v_oo[i], ord_oa[idx]) for idx, i in enumerate(encoded_game["arg2_output"])
    ]

    # if encoded_game['game_id'] in ['8f7791c488b51c98_4c51']:
    # pp = pprint.PrettyPrinter(width=200)
    # pp.pprint(decoded)

    return decoded


def main(args):

    arg_dict = {
        "data_path": args.raw_data_dir,
        "save_path": args.out_data_dir,
        "add_intention": args.encode_with_intention,
        "process_baseline": args.encode_original_baseline,
        "prepare_navi": args.prepare_navi,
    }

    # load vocabularies
    vocabs = {}
    vocab_names = [
        "input_vocab_word2id",
        "output_vocab_action_high",
        "output_vocab_action_navi",
        "output_vocab_action_all",
        "output_vocab_intent",
        "output_vocab_object",
    ]
    for vn in vocab_names:
        with open(os.path.join(args.out_data_dir, "vocab", "%s.json" % vn), "r") as f:
            vocabs[vn] = json.load(f)
    with open(
        os.path.join(args.ithor_assets_dir, "predicate_to_language.json"), "r"
    ) as f:
        vocabs["predicate_to_language"] = json.load(f)

    # partition of games
    game_id_to_split_file = os.path.join(args.out_data_dir, "game_id_to_split.json")
    with open(game_id_to_split_file, "r") as f:
        game_id_to_split = json.load(f)

    gamefiles_dir = os.path.join(args.raw_data_dir, "all_game_files")
    all_paths = []
    for idx, game_fn in enumerate(os.listdir(gamefiles_dir)):
        if args.debug and idx == 20:
            break
        # if '0b068eb7c0ef5e6a_b570' not in game_fn_path:
        #     continue
        game_id = game_fn.split(".")[0]
        split = game_id_to_split.get(game_id, "test")
        all_paths.append((game_id, split, vocabs, arg_dict))

    print(
        "Encoding %d game files by %d workers ... " % (len(all_paths), args.num_workers)
    )
    with Pool(args.num_workers) as p:
        result = p.map_async(
            prepare_data_mp_wrapper,
            all_paths,
            chunksize=len(all_paths) // args.num_workers + 1,
        )
        result.wait()

    encoded_games, encoded_tfds, encoded_edhs, encoded_navi = {}, {}, {}, {}
    partitions = ["train", "valid_seen", "valid_unseen", "test"]
    for split in partitions:
        encoded_games[split] = []
        encoded_tfds[split] = []
        encoded_edhs[split] = []
        encoded_navi[split] = []

    for (
        game_id,
        encoded_game,
        encoded_tfd,
        encoded_edhs_one_game,
        encoded_navi_intances,
    ) in result.get():
        split = encoded_game["split"]
        encoded_games[split].append(encoded_game)
        encoded_tfds[split].append(encoded_tfd)
        for encoded_edh in encoded_edhs_one_game:
            encoded_edhs[split].append(encoded_edh)
        if arg_dict["prepare_navi"]:
            encoded_navi[split].extend(encoded_navi_intances)

    assert sum([len(i) for i in encoded_games.values()]) == len(all_paths)

    print("Writing encoded results to jsons ... ")
    str1 = "_with_intention" if args.encode_with_intention else ""
    str2 = "_original_baseline" if args.encode_original_baseline else ""
    save_dir = os.path.join(args.out_data_dir, "encoded")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for split in partitions:
        with open(
            os.path.join(save_dir, "%s.game%s%s.json" % (split, str1, str2)), "w"
        ) as f:
            json.dump(encoded_games[split], f, sort_keys=False, indent=4)
        with open(
            os.path.join(save_dir, "%s.tfd%s%s.json" % (split, str1, str2)), "w"
        ) as f:
            json.dump(encoded_tfds[split], f, sort_keys=False, indent=4)
        with open(
            os.path.join(save_dir, "%s.edh%s%s.json" % (split, str1, str2)), "w"
        ) as f:
            json.dump(encoded_edhs[split], f, sort_keys=False, indent=4)
        if arg_dict["prepare_navi"]:
            with open(os.path.join(save_dir, "%s.navi.json" % (split)), "w") as f:
                json.dump(encoded_navi[split], f, sort_keys=False, indent=4)

    # pprint(encoded_games)
