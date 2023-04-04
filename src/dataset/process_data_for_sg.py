import json, random
from tqdm import tqdm
from pprint import pprint
from definitions.symbol_to_language import (
    obj_cls_symbol_to_language,
    action_to_language,
)
from definitions.symbol_to_language import subgoal_to_language
from definitions.symbol_to_language import state_to_predicate
from definitions.teach_tasks import GoalArguments, GoalConditions, GoalReceptacles
from transformers import BartTokenizer


pred_eos_idx = GoalConditions["EOS"].value
subj_none_idx = GoalArguments["NONE"].value
obj_none_idx = GoalReceptacles["NONE"].value


QUESTION_WORDS = {
    "Is",
    "Are",
    "How",
    "What",
    "Where",
    "Whether",
    "Shall",
    "Should",
    "Can",
    "Could",
    "Would",
    "May",
}


def process_one_game(processed_game, verbose=False):
    high_actions = processed_game["high_actions"]

    data_instances = []

    # first get all the possible edh instances within each game
    edh_start_end_idx = []
    edh_start_idx = 0
    for idx, high_action in enumerate(high_actions):
        if idx == len(high_actions) - 1:
            break

        next_high_action = high_actions[idx + 1]

        if high_action[2] == "dial" and next_high_action[2] != "dial":
            edh_start_idx = idx
        if (
            edh_start_idx != 0
            and high_action[2] != "dial"
            and next_high_action[2] == "dial"
        ):
            edh_end_idx = idx
            edh_start_end_idx.append((edh_start_idx, edh_end_idx))

    if verbose:
        print(processed_game["game_id"], processed_game["split"])
        for hidx, h in enumerate(high_actions):
            hidx, role, atype, aname, arg, arg_r, time, sg, event = h
            print("[%03d %s %s] %s %s" % (hidx, role, atype[0].upper(), aname, arg_r))
        print(edh_start_end_idx)

    if not edh_start_end_idx or processed_game["split"] == "test":
        return data_instances

    # iterate over all the edh instances
    for edh_start_idx, edh_end_idx in edh_start_end_idx:
        text_dialog = ""
        text_dialog_and_act = ""
        text_sg_done = ""
        idx_sg_done = []
        text_sg_todo_edh = ""
        idx_sg_todo_edh = []  # list of dict: {'action': '', 'arg1': '', 'arg2': ''}
        text_sg_todo_all = ""
        idx_sg_todo_all = []  # list of dict: {'action': '', 'arg1': '', 'arg2': ''}

        last_dialog_role, last_action = None, None
        for idx, high_action in enumerate(high_actions[: edh_start_idx + 1]):
            hidx, role, atype, aname, arg, arg_r, time, sg, event = high_action
            if atype == "dial":
                utter = arg_r.strip()
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
                    role_prefix = "Commander: " if role == "USR" else "Follower: "
                    text_dialog += role_prefix + utter

                if role == last_dialog_role and last_action[2] == "dial":
                    text_dialog_and_act += utter
                else:
                    role_prefix = "Commander: " if role == "USR" else "Follower: "
                    if last_action and last_action[2] != "dial":
                        role_prefix = ". " + role_prefix
                        text_dialog_and_act = text_dialog_and_act[:-1]
                    text_dialog_and_act += role_prefix + utter

                last_dialog_role = role

            else:
                if last_action and last_action[2] == "dial":
                    role_prefix = " Follower "
                    text_dialog_and_act = text_dialog_and_act.strip() + role_prefix
                else:
                    text_dialog_and_act = text_dialog_and_act.strip() + ", "
                if aname == "Pickup":
                    obj_arg = arg_r.split(" FROM ")[0]
                    obj_class, obj_idx = obj_arg.split("_")
                    act = action_to_language(aname)
                    obj_name = obj_cls_symbol_to_language(obj_class)
                    text_dialog_and_act += f"{act} {obj_name} {obj_idx}"
                elif aname in ["Place", "Pour"]:
                    obj_arg1, obj_arg2 = arg_r.split(" TO ")

                    obj_class1, obj_idx1 = obj_arg1.split("_")
                    obj_class2, obj_idx2 = obj_arg2.split("_")
                    obj_name1 = obj_cls_symbol_to_language(obj_class1)
                    obj_name2 = obj_cls_symbol_to_language(obj_class2)
                    act = action_to_language(aname).format(
                        f"{obj_name1} {obj_idx1}", f"{obj_name2} {obj_idx2}"
                    )
                    text_dialog_and_act += act
                else:
                    obj_arg = arg_r.split(" ")[0]
                    obj_class, obj_idx = obj_arg.split("_")
                    act = action_to_language(aname)
                    obj_name = obj_cls_symbol_to_language(obj_class)
                    text_dialog_and_act += f"{act} {obj_name} {obj_idx}"

            last_action = high_action

        text_dialog = text_dialog.strip()
        text_dialog_and_act = text_dialog_and_act.strip()
        tokenized_text_dialog = tokenizer(text_dialog, add_special_tokens=True)
        tokenized_text_dialog_and_act = tokenizer(
            text_dialog_and_act, add_special_tokens=True
        )

        # add subgoals that already completed into context (encoder input)
        text_sg_done += "Follower completed subgoals: "
        completed_subgoals_nl = []
        for idx, high_action in enumerate(high_actions[: edh_start_idx + 1]):
            hidx, role, atype, aname, arg, arg_r, time, sg, event = high_action
            sg = tuple(event[4]) if event[4] else None
            if sg:
                completed_subgoals_nl.append(subgoal_to_language(sg))

                subj, predicate, obj = sg
                predicate = state_to_predicate(sg)
                output_idx = {
                    "predicate": GoalConditions[predicate].value,
                    "subj": GoalArguments[subj].value,
                    "obj": GoalReceptacles[obj].value
                    if predicate == "parentReceptacles"
                    else obj_none_idx,
                }
                idx_sg_done.append(output_idx)
        idx_sg_done.append(
            {"predicate": pred_eos_idx, "subj": subj_none_idx, "obj": obj_none_idx}
        )

        text_sg_done += (
            "none"
            if not completed_subgoals_nl
            else "; ".join(completed_subgoals_nl) + ";"
        )
        # Note: add ';' for end of sentence prediction

        # add subgoals that are still to be completed as labels for prediction:
        # natural decoder input: natural language string
        # structural decoder output: index of actions and object arguments
        text_sg_todo_edh += "Future subgoals: "
        future_subgoals_edh_nl = []
        for idx, high_action in enumerate(
            high_actions[edh_start_idx + 1 : edh_end_idx + 1]
        ):
            hidx, role, atype, aname, arg, arg_r, time, sg, event = high_action
            sg = tuple(event[4]) if event[4] else None
            # print('[%03d %s %s] %s %s' % (hidx, role, atype[0].upper(), aname, arg_r))
            if sg:
                future_subgoals_edh_nl.append(subgoal_to_language(sg))
                subj, predicate, obj = sg
                predicate = state_to_predicate(sg)
                output_idx = {
                    "predicate": GoalConditions[predicate].value,
                    "subj": GoalArguments[subj].value,
                    "obj": GoalReceptacles[obj].value
                    if predicate == "parentReceptacles"
                    else obj_none_idx,
                }
                idx_sg_todo_edh.append(output_idx)

        idx_sg_todo_edh.append(
            {"predicate": pred_eos_idx, "subj": subj_none_idx, "obj": obj_none_idx}
        )
        text_sg_todo_edh += (
            "none"
            if not future_subgoals_edh_nl
            else "; ".join(future_subgoals_edh_nl) + ";"
        )

        text_sg_todo_all += "Future subgoals: "
        future_subgoals_all_nl = []
        for idx, high_action in enumerate(high_actions[edh_start_idx + 1 :]):
            hidx, role, atype, aname, arg, arg_r, time, sg, event = high_action
            sg = tuple(event[4]) if event[4] else None
            # print('[%03d %s %s] %s %s' % (hidx, role, atype[0].upper(), aname, arg_r))
            if sg:
                future_subgoals_all_nl.append(subgoal_to_language(sg))
                subj, predicate, obj = sg
                predicate = state_to_predicate(sg)
                output_idx = {
                    "predicate": GoalConditions[predicate].value,
                    "subj": GoalArguments[subj].value,
                    "obj": GoalReceptacles[obj].value
                    if predicate == "parentReceptacles"
                    else obj_none_idx,
                }
                idx_sg_todo_all.append(output_idx)
        idx_sg_todo_all.append(
            {"predicate": pred_eos_idx, "subj": subj_none_idx, "obj": obj_none_idx}
        )

        text_sg_todo_all += (
            "none"
            if not future_subgoals_all_nl
            else "; ".join(future_subgoals_all_nl) + ";"
        )
        tokenized_text_sg_todo_all = tokenizer(
            text_sg_todo_all, add_special_tokens=True
        )

        data_instance = {
            "game_id": processed_game["game_id"],
            "split": processed_game["split"],
            "edh_start_end_idx": [edh_start_idx, edh_end_idx],
            "text_dialog": text_dialog,
            "text_dialog_and_act": text_dialog_and_act,
            "text_sg_done": text_sg_done,
            "text_sg_todo_edh": text_sg_todo_edh,
            "text_sg_todo_all": text_sg_todo_all,
            # 'idx_sg_done': idx_sg_done,
            # 'idx_sg_todo_edh': idx_sg_todo_edh,
            # 'idx_sg_todo_all': idx_sg_todo_all,
            "idx_sg_done_pred": [i["predicate"] for i in idx_sg_done],
            "idx_sg_done_subj": [i["subj"] for i in idx_sg_done],
            "idx_sg_done_obj": [i["obj"] for i in idx_sg_done],
            "idx_sg_todo_edh_pred": [i["predicate"] for i in idx_sg_todo_edh],
            "idx_sg_todo_edh_subj": [i["subj"] for i in idx_sg_todo_edh],
            "idx_sg_todo_edh_obj": [i["obj"] for i in idx_sg_todo_edh],
            "idx_sg_todo_all_pred": [i["predicate"] for i in idx_sg_todo_all],
            "idx_sg_todo_all_subj": [i["subj"] for i in idx_sg_todo_all],
            "idx_sg_todo_all_obj": [i["obj"] for i in idx_sg_todo_all],
            "idx_sg_done_and_todo_edh_pred": [
                i["predicate"] for i in idx_sg_done + idx_sg_todo_edh
            ],
            "idx_sg_done_and_todo_edh_subj": [
                i["subj"] for i in idx_sg_done + idx_sg_todo_edh
            ],
            "idx_sg_done_and_todo_edh_obj": [
                i["obj"] for i in idx_sg_done + idx_sg_todo_edh
            ],
            "idx_sg_done_and_todo_all_pred": [
                i["predicate"] for i in idx_sg_done + idx_sg_todo_all
            ],
            "idx_sg_done_and_todo_all_subj": [
                i["subj"] for i in idx_sg_done + idx_sg_todo_all
            ],
            "idx_sg_done_and_todo_all_obj": [
                i["obj"] for i in idx_sg_done + idx_sg_todo_all
            ],
            "length_text_dialog": len(tokenized_text_dialog["input_ids"]),
            "length_text_dialog_and_act": len(
                tokenized_text_dialog_and_act["input_ids"]
            ),
            "length_text_sg_todo_all": len(tokenized_text_sg_todo_all["input_ids"]),
        }

        # print(len(high_actions), len(tokenized_text_dialog['input_ids']), len(tokenized_enc_inp_dial_and_act['input_ids']))
        if verbose:
            print(edh_start_idx, edh_end_idx)
            print(
                "text_dialog (len: %d)" % len(tokenized_text_dialog["input_ids"]),
                text_dialog,
            )
            print(
                "text_dialog_and_act (len: %d)"
                % len(tokenized_text_dialog_and_act["input_ids"]),
                text_dialog_and_act,
            )

            print("text_sg_done", text_sg_done)
            print("idx_sg_done")
            pprint(idx_sg_done)

            print("text_sg_todo_edh", text_sg_todo_edh)
            print("idx_sg_todo_edh")
            pprint(idx_sg_todo_edh)

            print(
                "text_sg_todo_all (len: %d)"
                % len(tokenized_text_sg_todo_all["input_ids"]),
                text_sg_todo_all,
            )
            print("idx_sg_todo_all")
            pprint(idx_sg_todo_all)

        data_instances.append(data_instance)

    return data_instances


if __name__ == "__main__":
    import os

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    processed_data_file = os.path.join(
        os.environ["DANLI_DATA_DIR"], "processed_20220610", "preprocessed_games.json"
    )
    with open(processed_data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    save_path = os.path.join(
        os.environ["DANLI_DATA_DIR"], "processed_20220610", "seq2seq_sg_pred_{}.json"
    )

    ctx_dial_length = []
    ctx_dial_and_act_length = []
    ctx_sg_todo_all_length = []

    train, valid_seen, valid_unseen = [], [], []
    for idx, (game_id, session) in tqdm(enumerate(data.items())):
        # if idx == 6:
        #     break
        split = session["split"]
        data_instances = process_one_game(session)
        ctx_dial_length.extend([i["length_text_dialog"] for i in data_instances])
        ctx_dial_and_act_length.extend(
            [i["length_text_dialog_and_act"] for i in data_instances]
        )
        ctx_sg_todo_all_length.extend(
            [i["length_text_sg_todo_all"] for i in data_instances]
        )

        if split == "train":
            train.extend(data_instances)
        elif split == "valid_seen":
            valid_seen.extend(data_instances)
        elif split == "valid_unseen":
            valid_unseen.extend(data_instances)

    print("train:", len(train))
    print("valid_seen:", len(valid_seen))
    print("valid_unseen:", len(valid_unseen))

    with open(save_path.format("train"), "w") as f:
        json.dump(train, f, indent=2)
    with open(save_path.format("valid_seen"), "w") as f:
        json.dump(valid_seen, f, indent=2)
    with open(save_path.format("valid_unseen"), "w") as f:
        json.dump(valid_unseen, f, indent=2)
    with open(save_path.format("debug"), "w") as f:
        json.dump(random.choices(train, k=30), f, indent=2)

    # pprint(sorted(ctx_dial_length, reverse=True)[:60])
    # pprint(sorted(ctx_dial_and_act_length,  reverse=True)[:60])
