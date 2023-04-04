import os
import re
import copy
import json
from attrdict import AttrDict
from pprint import pprint
from dataset.process_data_for_sg import QUESTION_WORDS
from definitions.symbol_to_language import (
    obj_cls_symbol_to_language,
    action_to_language,
)
from dataset.utils.ai2thor_utils import get_obj_type_from_oid, get_obj_name
from dataset.preprocess_gamefiles import register_object


def process_edh_for_subgoal_prediction(edh_instance):

    follower_action_history = copy.deepcopy(edh_instance["driver_action_history"])
    dialog_history = copy.deepcopy(edh_instance["dialog_history_cleaned"])

    # Note: edh_instance["interactions"] is forbidden to use since the ground truth
    #       actions are recoreded there. Here we use it only to get the time of user's
    #       history dialog actions, without using any forbidden information.

    # NOTE: CANNOT DO THIS DUE TO CHALLENGE RULE
    turn_idx = 0
    for action in edh_instance["interactions"]:
        if "utterance" in action:
            dialog_history[turn_idx].append(action["time_start"])
            turn_idx += 1
        if turn_idx == len(dialog_history):
            break
    
    
    follower_dialog_time = []
    for action in follower_action_history:
        if action["action_name"] == "Text":
            follower_dialog_time.append(action["time_start"])
    
    # follower_turn_idx = 0
    # last_follower_utter_time = -1
    # start_time = 0
    # for turn_idx, (d_role, d_utter) in enumerate(dialog_history):
    #     if d_role == "Commander":
    #         if last_follower_utter_time == -1:
    #             dialog_history[turn_idx].append(start_time)
    #             start_time += 0.001
    #         else:
    #             # assum the commander speaks right after the follower
    #             dialog_history[turn_idx].append(last_follower_utter_time + 0.001)

    #     elif d_role == "Driver":
    #         last_follower_utter_time = follower_dialog_time[follower_turn_idx]
    #         dialog_history[turn_idx].append(last_follower_utter_time)
    #         follower_turn_idx += 1
    #     else:
    #         raise ValueError("Unknown role {}".format(d_role))
    
    # if dialog_history[-1][0] == "Commander":
    #     dialog_history[-1][2] = 100000000000
    #     # because edh starts after an utterance, if the last utterance is from the
    #     # commander, we should move it to the last


    interaction_history = {a["time_start"]: a for a in follower_action_history}
    for d_role, d_utter, d_time in dialog_history:
        if d_time in interaction_history:
            interaction_history[d_time]["utterance"] = d_utter
            interaction_history[d_time]["role"] = d_role
        else:
            interaction_history[d_time] = {
                "action_name": "Text",
                "obj_interaction_action": 0,
                "utterance": d_utter,
                "role": d_role,
                "time_start": d_time,
            }

    text_dialog = ""
    text_dialog_and_act = ""

    last_dialog_role = None
    last_action = None
    last_non_dial_bot_action = None
    oid_to_obj_cls = {}
    obj_cls_cnt = {}
    agent_inventory = None
    is_navigating = False

    # process encoder context: dialog only or dialog and action history
    for time_start in sorted(interaction_history.keys()):

        action = interaction_history[time_start]

        if action["action_name"] == "Text":
            utter = action["utterance"].capitalize().strip()
            role = action["role"]
            if utter == "":
                # skip if the utterance is empty
                continue

            # add punctuation
            if utter[-1] not in [".", "?", "!"]:
                # add punctuation if the last character is not punctuation
                if utter.split()[0] in QUESTION_WORDS:
                    utter += "? "
                else:
                    utter += ". "

            # add space
            if text_dialog and text_dialog[-1] != " ":
                text_dialog += " "
            if text_dialog_and_act and text_dialog_and_act[-1] != " ":
                text_dialog_and_act += " "

            # add speaker role and utterance
            if role == last_dialog_role:
                text_dialog += utter
            else:
                role_prefix = "Commander: " if role == "Commander" else "Follower: "
                text_dialog += role_prefix + utter

            if role == last_dialog_role and last_action["action_name"] == "Text":
                text_dialog_and_act += utter
            else:
                role_prefix = "Commander: " if role == "Commander" else "Follower: "
                if last_action and last_action["action_name"] != "Text":
                    role_prefix = ". " + role_prefix
                    text_dialog_and_act = text_dialog_and_act[:-1]
                text_dialog_and_act += role_prefix + utter

            last_dialog_role = role

        else:
            if not action["obj_interaction_action"]:
                # ignore low-level (primitive) navigation actions
                continue

            # Before adding any actions, add the correct prefix
            if last_action and last_action["action_name"] == "Text":
                role_prefix = " Follower "
                text_dialog_and_act = text_dialog_and_act.strip() + role_prefix
            else:
                text_dialog_and_act = text_dialog_and_act.strip() + ", "

            # Convert action, object into natural language forms
            aname = action["action_name"]
            oid = action["oid"]
            oid_to_obj_cls, obj_cls_cnt = register_object(
                oid, oid_to_obj_cls, obj_cls_cnt
            )
            obj_cls_with_id = get_obj_name(oid, oid_to_obj_cls)
            obj_cls, obj_idx = obj_cls_with_id.split("_")
            obj_name = obj_cls_symbol_to_language(obj_cls)

            act = action_to_language(aname)

            # Insert a high-level navigation action if there is navigation
            # before the manipulative action
            if (
                last_non_dial_bot_action
                and not last_non_dial_bot_action["obj_interaction_action"]
            ):
                text_dialog_and_act += f"go for {obj_name} {obj_idx}"

            # Add this manipulative action

            if aname in ["Place", "Pour"]:
                p_obj_cls, p_obj_id = agent_inventory.split("_")
                p_obj_name = obj_cls_symbol_to_language(p_obj_cls)
                act = action_to_language(aname).format(
                    f"{p_obj_name} {p_obj_id}", f"{obj_name} {obj_idx}"
                )
                text_dialog_and_act += act
                if aname == "Place":
                    agent_inventory = None
            else:
                act = action_to_language(aname)
                text_dialog_and_act += f"{act} {obj_name} {obj_idx}"
                if aname == "Pickup":
                    agent_inventory = obj_cls_with_id

            last_non_dial_bot_action = action

        last_action = action

    edh_context = {
        "text_dialog": text_dialog,
        "text_dialog_and_act": text_dialog_and_act,
    }

    return edh_context, dialog_history


DATA_INSTANCE_FIELDS = [
    "dialog_input",
    "dialog_role",
    "frames",
    "action_input",
    "arg1_input",
    "arg2_input",
    "ordering_dialog",
    "ordering_tokens",
    "ordering_frames",
    "ordering_action",
]


def process_edh_for_inference(
    edh_instance,
    edh_history_images,
    vocabs,
    tokenizer,
    add_intention=False,
    controller=None,
    data_handler=None,
    processed_edh_instance=None,
):
    word2id = vocabs["input_vocab_word2id"]
    id2word = {v: k for k, v in vocabs["input_vocab_word2id"].items()}
    pred2lang = vocabs["predicate_to_language"]
    navi_actions = set(vocabs["output_vocab_action_navi"].keys())
    role_map = {"USR": 1, "BOT": 2}  # leave 0 for padding

    # print(history_img_feats.shape)

    action_history = edh_instance["driver_action_history"]
    assert len(action_history) == len(edh_history_images)
    dialog_history = edh_instance["dialog_history_cleaned"]

    # Note: edh_instance["interactions"] is forbidden to use since the ground truth
    #       is recoreded there. Here we use it only to get the time of user's
    #       history dialog actions, without using any forbidden information.
    turn_idx = 0
    for action in edh_instance["interactions"]:
        if "utterance" in action:
            dialog_history[turn_idx].append(action["time_start"])
            turn_idx += 1
        if turn_idx == len(dialog_history):
            break

    # for idx, d in enumerate(dialog_history):
    #     print(idx, d)

    edh = {}
    edh.update({k: [] for k in DATA_INSTANCE_FIELDS})

    next_dialog_idx = 0
    ordering_step = 1  # leave 0 for padding
    user_utter_token_ids = None
    navi_start_idx = None

    # add start tokens
    edh["action_input"].append([word2id["[BEG_TRAJ]"]])
    edh["arg1_input"].append([word2id["[PAD]"]])
    edh["arg2_input"].append([word2id["[PAD]"]])
    edh["ordering_action"].append(ordering_step)

    if not action_history:
        for role, utter, time in dialog_history:
            assert role == "Commander"
            utter_tokens = [t.text for t in tokenizer(utter.lower())]
            if len(utter_tokens) != 0:  # non-empty utterance
                if user_utter_token_ids is None:
                    user_utter_token_ids = [word2id.get(t, 1) for t in utter_tokens]
                else:
                    token_ids = [word2id.get(t, 1) for t in utter_tokens]
                    if id2word[user_utter_token_ids[-1]].isalpha():
                        token_ids = [word2id["."]] + token_ids
                    user_utter_token_ids.extend(token_ids)
        if user_utter_token_ids is not None:
            utter_len = len(user_utter_token_ids) + 1
            edh["dialog_input"].extend([word2id["[USR]"]] + user_utter_token_ids)
            edh["dialog_role"].extend([role_map["USR"]] * utter_len)
            edh["ordering_tokens"].extend(range(1, utter_len + 1))
            edh["ordering_dialog"].extend([ordering_step] * utter_len)
            user_utter_token_ids = None
            ordering_step += 1  # causal index

    for idx, a in enumerate(action_history):

        # print(idx, a, next_dialog_idx)

        # insert the user dialog actions at the correct time
        if idx + 1 != len(action_history) and next_dialog_idx + 1 != len(
            dialog_history
        ):
            next_bot_act_time = action_history[idx + 1]["time_start"]
            next_da_role, utter, next_da_time = dialog_history[next_dialog_idx]
            while next_da_role == "Commander" and next_da_time < next_bot_act_time:
                utter_tokens = [t.text for t in tokenizer(utter.lower())]
                if len(utter_tokens) != 0:  # non-empty utterance
                    if user_utter_token_ids is None:
                        user_utter_token_ids = [word2id.get(t, 1) for t in utter_tokens]
                    else:
                        token_ids = [word2id.get(t, 1) for t in utter_tokens]
                        if id2word[user_utter_token_ids[-1]].isalpha():
                            token_ids = [word2id["."]] + token_ids
                        user_utter_token_ids.extend(token_ids)
                    # print("add user dialog:", utter)

                next_dialog_idx += 1
                next_da_role, utter, next_da_time = dialog_history[next_dialog_idx]

        # skip out-of-action-space actions (bugs)
        if a["action_name"] in ["Navigation", "SearchObject", "OpenProgressCheck"]:
            continue

        # skip empty bot dialog actions
        if (
            a["action_name"] == "Text"
            and len(dialog_history[next_dialog_idx][1].split()) == 0
        ):
            next_dialog_idx += 1
            prev_action = a
            # print("skip empty bot dialog!")
            continue

        # skip navigation actions as they are not input to our model
        if a["action_name"] in navi_actions:
            "CHANGE_THE_AGENT_POSE"  # TODO
            # optional: record object detection result/map/snapshot
            if navi_start_idx is None:
                navi_start_idx = idx
            continue

        """ Begin adding a BOT action to the trajectory 
            For each bot action, we first add the observation frame. 
            Then we check if the user says anything, if they do, add it. 
            If intention is modeled, we add the intention (predicted). 
            Finally add the action and the corresponding object argument. """

        # if the bot navigates before taking a manipulation action
        # add it in front of the current action we are processing
        if navi_start_idx is not None and a["oid"] is not None:
            # add the visual observation
            edh["frames"].append(edh_history_images[navi_start_idx])
            edh["ordering_frames"].append(ordering_step)
            ordering_step += 1  # causal index

            # add user dialog (if any)
            if user_utter_token_ids is not None:
                utter_len = len(user_utter_token_ids) + 1
                edh["dialog_input"].extend([word2id["[USR]"]] + user_utter_token_ids)
                edh["dialog_role"].extend([role_map["USR"]] * utter_len)
                edh["ordering_tokens"].extend(range(1, utter_len + 1))
                edh["ordering_dialog"].extend([ordering_step] * utter_len)
                user_utter_token_ids = None

                ordering_step += 1  # causal index

            # add intention (subgoals)
            if add_intention:
                # TODO: have to use the model to predict intentions iteractively! OMG
                predictions, edh = controller.predict_intent(edh, data_handler)
                edh["ordering_intent"].append(ordering_step)
                ordering_step += 1

            # add action
            target_obj = get_obj_type_from_oid(a["oid"])
            edh["action_input"].append(
                [word2id.get(w, 1) for w in pred2lang["Goto"].split()]
            )
            edh["arg1_input"].append(
                [word2id.get(w, 1) for w in pred2lang[target_obj].split()]
            )
            edh["arg2_input"].append(
                [word2id.get(w, 1) for w in pred2lang[target_obj].split()]
            )
            edh["ordering_action"].append(ordering_step)

            navi_start_idx = None

        """  finally we can add the current action """
        # add the current visual observation
        edh["frames"].append(edh_history_images[idx])
        edh["ordering_frames"].append(ordering_step)
        ordering_step += 1  # causal index

        # add user dialog (if any)
        if user_utter_token_ids is not None:
            utter_len = len(user_utter_token_ids) + 1
            edh["dialog_input"].extend([word2id["[USR]"]] + user_utter_token_ids)
            edh["dialog_role"].extend([role_map["USR"]] * utter_len)
            edh["ordering_tokens"].extend(range(1, utter_len + 1))
            edh["ordering_dialog"].extend([ordering_step] * utter_len)
            user_utter_token_ids = None

            ordering_step += 1  # causal index

        # add intention (subgoals)
        if add_intention:
            # TODO: have to use the model to predict intentions iteractively! OMG
            predictions, edh = controller.predict_intent(edh, data_handler)
            edh["ordering_intent"].append(ordering_step)
            ordering_step += 1

        # add action
        if a["action_name"] == "Text":
            utter = dialog_history[next_dialog_idx][1]
            utter_tokens = [t.text for t in tokenizer(utter.lower())]
            # print("add bot dialog", utter)
            assert (
                len(utter_tokens) != 0
            )  # empty utterance should have been filter out earlier
            utter_token_ids = [word2id.get(t, 1) for t in utter_tokens]
            utter_len = len(utter_token_ids) + 1
            edh["dialog_input"].extend([word2id["[BOT]"]] + utter_token_ids)
            edh["dialog_role"].extend([role_map["BOT"]] * utter_len)
            edh["ordering_tokens"].extend(range(1, utter_len + 1))
            edh["ordering_dialog"].extend([ordering_step] * utter_len)
            arg1_otype = arg2_otype = "None"

            next_dialog_idx += 1

        else:
            arg1_otype = get_obj_type_from_oid(a["oid"])
            arg2_otype = "None"

        edh["action_input"].append(
            [word2id.get(w, 1) for w in pred2lang[a["action_name"]].split()]
        )
        edh["arg1_input"].append(
            [word2id.get(w, 1) for w in pred2lang[arg1_otype].split()]
        )
        edh["arg2_input"].append(
            [word2id.get(w, 1) for w in pred2lang[arg2_otype].split()]
        )
        edh["ordering_action"].append(ordering_step)
        # Note: ordering_step should not step forward here since the visual observation
        # of the action_input has not been added yet.

    # end of loop
    assert len(edh["action_input"]) == len(edh["frames"]) + 1
    edh["ordering_step"] = ordering_step
    # Note that the "current observation" after performing the last action in the history
    # is not added yet. Thus the number of frames is 1 less than the number of input actions

    # ##### sanity check
    # for field in edh:
    #     print("____________ FIELD: %s _____________" % field)
    #     print("raw", edh[field])
    #     # print(
    #     #     "pro",
    #     #     processed_edh_instance[field] if field in processed_edh_instance else None,
    #     # )
    #     print("__________________________________________________________")

    return edh


def load_vocab(vocab_path):
    """
    load a vocabulary from the dataset
    """
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
        with open(os.path.join(vocab_path, "%s.json" % vn), "r") as f:
            vocabs[vn] = json.load(f)

    return vocabs


def get_obj_type_from_oid(oid):
    parts = oid.split("|")
    if len(parts) == 4:
        return parts[0]
    else:
        return parts[-1].split("_")[0]


def get_feat_shape(visual_archi, compress_type=None):
    """
    Get feat shape depending on the training archi and compress type
    """
    if visual_archi == "fasterrcnn":
        # the RCNN model should be trained with min_size=224
        feat_shape = (-1, 2048, 7, 7)
    elif visual_archi == "maskrcnn":
        # the RCNN model should be trained with min_size=800
        feat_shape = (-1, 2048, 10, 10)
    elif visual_archi == "resnet18":
        feat_shape = (-1, 512, 7, 7)
    else:
        raise NotImplementedError("Unknown archi {}".format(visual_archi))

    if compress_type is not None:
        if not re.match(r"\d+x", compress_type):
            raise NotImplementedError("Unknown compress type {}".format(compress_type))
        compress_times = int(compress_type[:-1])
        feat_shape = (
            feat_shape[0],
            feat_shape[1] // compress_times,
            feat_shape[2],
            feat_shape[3],
        )
    return feat_shape


def feat_compress(feat, compress_type):
    """
    Compress features by channel average pooling
    """
    assert re.match(r"\d+x", compress_type) and len(feat.shape) == 4
    times = int(compress_type[:-1])
    assert feat.shape[1] % times == 0
    feat = feat.reshape(
        (feat.shape[0], times, feat.shape[1] // times, feat.shape[2], feat.shape[3])
    )
    feat = feat.mean(dim=1)
    return feat
