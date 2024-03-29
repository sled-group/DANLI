#!/usr/bin/env python

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import glob
import json
import multiprocessing as mp
import os
from textwrap import indent
import torch
import random

from pprint import pprint
from argparse import ArgumentParser
from datetime import datetime

from teach.eval.compute_metrics import aggregate_metrics
from teach.inference.inference_runner_base import (
    InferenceRunnerConfig,
    InferenceBenchmarks,
)
from teach.inference.edh_inference_runner import EdhInferenceRunner
from teach.inference.tfd_inference_runner import TfdInferenceRunner
from teach.logger import create_logger
from teach.utils import dynamically_load_class, save_json

logger = create_logger(__name__)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help='Base data directory containing subfolders "games" and one of "edh_instances" or "tfd_instances"',
    )
    arg_parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Images directory for episode replay output",
    )
    arg_parser.add_argument(
        "--use_img_file",
        dest="use_img_file",
        action="store_true",
        help="synchronous save images with model api use the image file instead of streaming image",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store output files from running inference",
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
        "--num_processes", type=int, default=1, help="Number of processes to use"
    )
    arg_parser.add_argument(
        "--max_init_tries",
        type=int,
        default=5,
        help="Max attempts to correctly initialize an instance before declaring it as a failure",
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
        "--metrics_file",
        type=str,
        required=True,
        help="File used to store metrics",
    )
    arg_parser.add_argument(
        "--model_module",
        type=str,
        default="teach.inference.sample_model",
        help="Path of the python module to load the model class from.",
    )
    arg_parser.add_argument(
        "--model_class",
        type=str,
        default="SampleModel",
        help="Name of the TeachModel class to use during inference.",
    )
    arg_parser.add_argument(
        "--replay_timeout",
        type=int,
        default=500,
        help="The timeout for playing back the interactions in an episode.",
    )
    arg_parser.add_argument(
        "--benchmark",
        type=str,
        default="edh",
        help="TEACh benchmark to run inference for; Supported values: %s"
        % str([e.value for e in InferenceBenchmarks]),
    )
    arg_parser.add_argument(
        "--edh_instance_file",
        type=str,
        help="Run only on this EDH instance. Split must be set appropriately to find corresponding game file.",
    )
    arg_parser.add_argument(
        "--tfd_instance_file",
        type=str,
        help="Run only on this TfD instance. Split must be set appropriately to find corresponding game file.",
    )

    start_time = datetime.now()
    args, model_args = arg_parser.parse_known_args()
    args.debug = "--debug" in model_args

    # args.split = "valid_unseen" # TODO: remove this!

    if args.benchmark == InferenceBenchmarks.EDH:
        inference_runner_class = EdhInferenceRunner
    elif args.benchmark == InferenceBenchmarks.TFD:
        inference_runner_class = TfdInferenceRunner
    else:
        raise RuntimeError(
            "Invalid valid for --benchmark; must be one of %s "
            % str([e.value for e in InferenceBenchmarks])
        )

    if args.edh_instance_file:
        input_files = [args.edh_instance_file]
        args.benchmark = InferenceBenchmarks.EDH
    elif args.tfd_instance_file:
        input_files = [args.tfd_instance_file]
        args.benchmark = InferenceBenchmarks.TFD
    else:
        input_subdir = args.benchmark + "_instances"
        inference_output_files = glob.glob(
            os.path.join(args.output_dir, "inference__*.json")
        )
        finished_input_files = [
            os.path.join(os.path.basename(fn).split("__")[1])
            for fn in inference_output_files
        ]
        if args.split in [
            "train",
            "valid_seen",
            "valid_unseen",
            "test_seen",
            "test_unseen",
        ]:
            split = args.split
            input_files = [
                os.path.join(args.data_dir, input_subdir, args.split, f)
                for f in os.listdir(
                    os.path.join(args.data_dir, input_subdir, args.split)
                )
                if (f not in finished_input_files or args.debug)
            ]
        else:
            split = (
                "valid_seen"
                if args.split in ["divided_val_seen", "divided_test_seen"]
                else "valid_unseen"
            )
            with open(
                "src/teach/meta_data_files/divided_split/%s/%s.txt"
                % (input_subdir, args.split),
                "r",
            ) as f:
                input_file_names = set(f.read().splitlines())
            input_files = [
                os.path.join(args.data_dir, input_subdir, split, f)
                for f in input_file_names
                if (f not in finished_input_files or args.debug)
            ]
        if not input_files:
            print(
                f"all the edh instances have been ran for input_dir={os.path.join(args.data_dir, 'edh_instances', args.split)}"
            )
            exit(1)

    # pprint(edh_instance_files[:5])
    # random.shuffle(edh_instance_files)
    # random.shuffle(edh_instance_files)
    if args.debug:
        quick_start = []
        for f in input_files[:3]:
            quick_start.append(f)
        random.shuffle(quick_start)
        input_files = quick_start

    pprint(input_files[:5])
    print("Total files to run:", len(input_files))

    runner_config = InferenceRunnerConfig(
        data_dir=args.data_dir,
        split=split,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        metrics_file=args.metrics_file,
        num_processes=args.num_processes,
        max_init_tries=args.max_init_tries,
        max_traj_steps=args.max_traj_steps,
        max_api_fails=args.max_api_fails,
        model_class=dynamically_load_class(args.model_module, args.model_class),
        replay_timeout=args.replay_timeout,
        model_args=model_args,
        use_img_file=args.use_img_file,
    )

    runner = inference_runner_class(input_files, runner_config)
    print("start inference runner")
    metrics = runner.run()
    inference_end_time = datetime.now()
    logger.info("Time for inference: %s" % str(inference_end_time - start_time))

    results = aggregate_metrics(metrics, args)

    metric_str = "-------------\n"
    metric_str += "SPLIT: %s\n" % args.split
    metric_str += "SR: %d/%d = %.4f\n" % (
        results["success"]["num_successes"],
        results["success"]["num_evals"],
        results["success"]["success_rate"],
    )
    metric_str += "GC: %d/%d = %.4f\n" % (
        results["goal_condition_success"]["completed_goal_conditions"],
        results["goal_condition_success"]["total_goal_conditions"],
        results["goal_condition_success"]["goal_condition_success_rate"],
    )
    metric_str += "PLW SR: %.4f\n" % (results["path_length_weighted_success_rate"])
    metric_str += "PLW GC: %.4f\n" % (
        results["path_length_weighted_goal_condition_success_rate"]
    )
    metric_str += "-------------\n"
    with open(args.metrics_file + "_scores.txt", "a+") as h:
        h.write(metric_str)
    print(metric_str)

    results["traj_stats"] = metrics
    with open(
        args.metrics_file + "_%s_%d" % (args.split, results["success"]["num_evals"]),
        "w",
    ) as h:
        json.dump(results, h, indent=2)

    end_time = datetime.now()
    logger.info(
        "Total time for inference and evaluation: %s" % str(end_time - start_time)
    )


if __name__ == "__main__":
    # Using spawn method, parent process creates a new and independent child process,
    # which avoid sharing unnecessary resources.
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # mp.set_start_method("spawn", force=True)
    main()
