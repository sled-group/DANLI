import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--eval_name", default="test")
parser.add_argument("--data_dir", default=os.environ["DANLI_DATA_DIR"])
parser.add_argument("--images_dir", default="/tmp/teach-replay-images")
parser.add_argument("--output_base_dir", default=os.environ["DANLI_EVAL_DIR"])
parser.add_argument(
    "--depth_model_path",
    default=os.path.join(os.environ["DANLI_MODEL_DIR"], "depth_model.pth"),
)
parser.add_argument(
    "--panoptic_model_path",
    default=os.path.join(os.environ["DANLI_MODEL_DIR"], "panoptic_model.pth"),
)
parser.add_argument(
    "--panoptic_model_config",
    default=os.path.join(
        os.environ["DANLI_ROOT_DIR"], "src/model/model/m2former_swin_b.py"
    ),
)
parser.add_argument(
    "--state_estimator_path",
    default=os.path.join(os.environ["DANLI_MODEL_DIR"], "state_estimator.pth"),
)
parser.add_argument(
    "--subgoal_predictor_path",
    default=os.path.join(os.environ["DANLI_MODEL_DIR"], "subgoal_predictor"),
)
parser.add_argument("--subgoal_predictor_ckpt", default="ckpt.pth")
parser.add_argument(
    "--fastdownward_path",
    type=str,
    default=os.path.join(os.environ["FASTDOWNWARD_DIR"], "fast-downward.py"),
)
parser.add_argument(
    "--pddl_domain_file",
    type=str,
    default=os.path.join(
        os.environ["DANLI_ROOT_DIR"], "src/sledmap/plan/domains/teach/teach_domain.pddl"
    ),
)
parser.add_argument(
    "--gt_subgoals_file",
    type=str,
    default=os.path.join(
        os.environ["DANLI_DATA_DIR"], "processed_20220610/games_gt_subgoals.json"
    ),
)
parser.add_argument("--hlsm_use_gt_seg", action="store_true")
parser.add_argument("--hlsm_use_gt_obj_det", action="store_true")
parser.add_argument("--hlsm_use_gt_depth", action="store_true")
parser.add_argument("--use_gt_subgoals", action="store_true")
parser.add_argument("--agent_type", type=str, default="pddl", choices=["pddl", "fsm"])
parser.add_argument("--disable_replan", action="store_true")
parser.add_argument("--disable_scene_pruning", action="store_true")
parser.add_argument("--last_subgoal_only", action="store_true")

parser.add_argument(
    "--num_processes",
    type=int,
    default=1,
    help="Number of AI2THOR processes to run in parallel.",
)
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
parser.add_argument(
    "--split",
    default="valid_unseen",
    choices=["train", "valid_seen", "valid_unseen"],
)
parser.add_argument(
    "--benchmark",
    type=str,
    default="edh",
    choices=["edh", "tfd"],
    help="TEACh benchmark to run inference for; Supported values: edh, tfd",
)
parser.add_argument("--api", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save_meta_data", action="store_true")
parser.add_argument(
    "--pause_at",
    type=str,
    choices=["interaction", "each_step", "no_pause"],
    default="no_pause",
)
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

preds_dir = os.path.join(args.output_base_dir, args.eval_name, "predictions")
metrics_dir = os.path.join(args.output_base_dir, args.eval_name, "metrics")
metrics_file = os.path.join(metrics_dir, "metrics")
log_dir = os.path.join(args.output_base_dir, args.eval_name, "log")

pddl_save_dir = os.path.join(args.output_base_dir, args.eval_name, "pddl_problems")
if not os.path.exists(pddl_save_dir):
    os.makedirs(pddl_save_dir)
meta_save_dir = os.path.join(args.output_base_dir, args.eval_name, "meta")
if not os.path.exists(meta_save_dir):
    os.makedirs(meta_save_dir)

if not args.api:
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

splits = args.split.split(",")
for split in splits:
    subprocess.run(
        (
            "python src/teach/cli/inference.py "
            if args.debug
            else ("teach_api " if args.api else "teach_inference ")
        )
        + f"--eval_name {args.eval_name} "
        f"--data_dir {args.data_dir} "
        f"--images_dir {args.images_dir} "
        f"--split {split} "
        f"--benchmark {args.benchmark} "
        f"--model_module teach.inference.neural_symbolic_model "
        f"--model_class NeuralSymbolicModel "
        f"--depth_model_path {args.depth_model_path} "
        f"--panoptic_model_path {args.panoptic_model_path} "
        f"--panoptic_model_config {args.panoptic_model_config} "
        f"--state_estimator_path {args.state_estimator_path} "
        f"--subgoal_predictor_path {args.subgoal_predictor_path} "
        f"--subgoal_predictor_ckpt {args.subgoal_predictor_ckpt} "
        f"--fastdownward_path {args.fastdownward_path} "
        f"--pddl_problem_save_dir {pddl_save_dir} "
        f"--meta_save_dir {meta_save_dir} "
        f"--pddl_domain_file {args.pddl_domain_file} "
        f"--gt_subgoals_file {args.gt_subgoals_file} "
        f"--agent_type {args.agent_type} "
        + ("--disable_replan " if args.disable_replan else "")
        + ("--disable_scene_pruning " if args.disable_scene_pruning else "")
        + ("--last_subgoal_only " if args.last_subgoal_only else "")
        + (
            ""
            if args.api
            else f"--num_processes {args.num_processes} "
            f"--gpu_number {args.num_gpus} "
            f"--log_dir {log_dir} "
            f"--output_dir {preds_dir} "
            f"--metrics_file {metrics_file} "
            f"--pause_at {args.pause_at} "
        )
        + ("--plot " if args.plot else "")
        + ("--save_meta_data " if args.save_meta_data else "")
        + ("--hlsm_use_gt_seg " if args.hlsm_use_gt_seg else "")
        + ("--hlsm_use_gt_obj_det " if args.hlsm_use_gt_obj_det else "")
        + ("--hlsm_use_gt_depth " if args.hlsm_use_gt_depth else "")
        + ("--use_gt_subgoals " if args.use_gt_subgoals else "")
        + ("--debug " if args.debug else ""),
        shell=True,
    )
