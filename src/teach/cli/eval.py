#!/usr/bin/env python

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import re
import csv
import json
from tempfile import NamedTemporaryFile
import shutil

from argparse import ArgumentParser

from teach.eval.compute_metrics import (
    aggregate_metrics,
    create_new_traj_metrics,
    load_edh_traj_metrics,
    load_tfd_traj_metrics,
)
from teach.inference.inference_runner_base import InferenceBenchmarks
from teach.utils import save_json
from teach.logger import create_logger

logger = create_logger(__name__)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help='Base data directory containing subfolders "games" and "edh_instances',
    )
    arg_parser.add_argument(
        "--inference_output_dir",
        type=str,
        required=True,
        help="Directory containing output files from running inference on EDH instances",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        default="valid_seen",
        choices=[
            "train",
            "valid_seen",
            "valid_unseen",
            "test_seen",
            "test_unseen",
            "divided_val_seen",
            "divided_val_unseen",
            "divided_test_seen",
            "divided_test_unseen",
        ],
        help="One of train, valid_seen, valid_unseen, test_seen, test_unseen",
    )
    arg_parser.add_argument(
        "--benchmark",
        type=str,
        default="edh",
        help="TEACh benchmark to run inference for; Supported values: %s"
        % str([e.value for e in InferenceBenchmarks]),
    )
    arg_parser.add_argument(
        "--max_traj_steps",
        type=int,
        default=1000,
        help="Max predicted trajectory steps",
    )
    arg_parser.add_argument(
        "--max_api_fails", type=int, default=30, help="Max allowed API failures"
    )
    arg_parser.add_argument(
        "--metrics_file", type=str, required=True, help="File used to store metrics"
    )
    args = arg_parser.parse_args()

    if args.benchmark == InferenceBenchmarks.EDH:
        traj_eval_fn = load_edh_traj_metrics
        input_subdir = "edh_instances"
    elif args.benchmark == InferenceBenchmarks.TFD:
        traj_eval_fn = load_tfd_traj_metrics
        input_subdir = "tfd_instances"
    else:
        raise RuntimeError(
            "Invalid valid for --benchmark; must be one of %s "
            % str([e.value for e in InferenceBenchmarks])
        )

    if args.split in [
        "train",
        "valid_seen",
        "valid_unseen",
        "test_seen",
        "test_unseen",
    ]:
        input_files = set(
            os.listdir(os.path.join(args.data_dir, input_subdir, args.split))
        )
    else:
        with open(
            "src/teach/meta_data_files/divided_split/%s/%s.txt"
            % (input_subdir, args.split),
            "r",
        ) as f:
            input_files = set(f.read().splitlines())

    output_files = [
        os.path.join(args.inference_output_dir, f)
        for f in os.listdir(args.inference_output_dir)
        if re.sub("inference__", "", f) in input_files
    ]
    pred_action_files = [
        re.sub("inference__", "pred_actions__", f) for f in output_files
    ]

    input_files_missing_output = list(
        set(input_files).difference(
            [os.path.basename(f).split("__")[1] for f in output_files]
        )
    )
    logger.info(
        "Evaluating split %s requiring %d files" % (args.split, len(input_files))
    )
    logger.info(
        "Found output files for %d instances; treating remaining %d as failed..."
        % (len(output_files), len(input_files_missing_output))
    )

    traj_stats = dict()
    for idx, output_file in enumerate(output_files):
        pred_actions_file = pred_action_files[idx]
        if not os.path.isfile(pred_actions_file):
            logger.warning(
                "Skipping EDH instance %s with output file %s due to missing predicted actions file %s"
                % (
                    os.path.basename(output_file).split("__")[1],
                    output_file,
                    pred_actions_file,
                )
            )
            input_files_missing_output.append(
                os.path.basename(output_file).split("__")[1]
            )
        instance_id, traj_metrics = traj_eval_fn(output_file, pred_actions_file, args)
        traj_stats[instance_id] = traj_metrics

    for instance_file in input_files_missing_output:
        instance_id = re.sub(".json", "", os.path.basename(instance_file))
        game_id = instance_id.split(".")[0]
        traj_metrics = create_new_traj_metrics(instance_id, game_id)
        traj_metrics["error"] = 1
        traj_stats[instance_id] = traj_metrics

    results = aggregate_metrics(traj_stats, args)
    logger.info("-------------")
    logger.info(
        "SR: %d/%d = %.4f"
        % (
            results["success"]["num_successes"],
            results["success"]["num_evals"],
            results["success"]["success_rate"],
        )
    )
    logger.info(
        "GC: %d/%d = %.4f"
        % (
            results["goal_condition_success"]["completed_goal_conditions"],
            results["goal_condition_success"]["total_goal_conditions"],
            results["goal_condition_success"]["goal_condition_success_rate"],
        )
    )
    logger.info("PLW SR: %.4f" % (results["path_length_weighted_success_rate"]))
    logger.info(
        "PLW GC: %.4f" % (results["path_length_weighted_goal_condition_success_rate"])
    )
    logger.info("-------------")

    # collect results
    filename = "eval_results.json"
    exp_name = args.inference_output_dir.split("/")[-2]
    if not os.path.exists(filename):
        with open(filename, "w") as h:
            json.dump({exp_name: {args.split: results}}, h, indent=4)
    else:
        with open(filename, "r") as h:
            all_eval_results = json.load(h)
        if exp_name not in all_eval_results:
            all_eval_results[exp_name] = {args.split: results}
        else:
            all_eval_results[exp_name][args.split] = results
        with open(filename, "w") as h:
            json.dump(all_eval_results, h, indent=4)

    results["traj_stats"] = traj_stats
    with open(
        args.metrics_file + "_%s_%d" % (args.split, results["success"]["num_evals"]),
        "w",
    ) as h:
        json.dump(results, h, indent=4)


if __name__ == "__main__":
    main()
