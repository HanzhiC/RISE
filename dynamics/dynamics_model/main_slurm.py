#!/usr/bin/env python3
"""Create and submit a Slurm job for distributed training."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parent
    default_log_dir = repo_root / "slurm_logs"
    default_conda_sh = Path.home() / "anaconda3" / "etc" / "profile.d" / "conda.sh"

    parser = argparse.ArgumentParser(
        description="Submit a Slurm training job for this repo."
    )
    parser.add_argument(
        "--script",
        default="main.py",
        help="Training entry script relative to repo root. Default: main.py",
    )
    parser.add_argument(
        "--config",
        default="configs/ltx_model/finetune.yaml",
        help="Training config relative to repo root. Default: configs/ltx_model/finetune.yaml",
    )
    parser.add_argument("--job-name", default="dynamics-ft", help="Slurm job name.")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes.")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="GPUs per node.")
    parser.add_argument(
        "--cpus-per-task", type=int, default=4, help="CPUs per Slurm task."
    )
    parser.add_argument(
        "--mem-per-gpu-gb",
        type=int,
        default=96,
        help="Memory in GB to request per GPU. Total memory is computed automatically.",
    )
    parser.add_argument("--time", default="72:00:00", help="Slurm time limit.")
    parser.add_argument("--partition", default="DEADLINE", help="Slurm partition.")
    parser.add_argument(
        "--constraint",
        default="",
        help="Optional Slurm constraint string. Leave empty to disable.",
    )
    parser.add_argument(
        "--gres-vram",
        default="",
        help="Optional VRAM gres suffix, e.g. 48G. Leave empty to request plain gpu:N.",
    )
    parser.add_argument(
        "--comment", default="corl w anran", help="Optional Slurm comment."
    )
    parser.add_argument(
        "--log-dir",
        default=str(default_log_dir),
        help="Directory for generated .job/.out files.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Torch distributed master port.",
    )
    parser.add_argument(
        "--conda-env",
        default="rise",
        help="Conda env to activate inside the Slurm job. Default: current env.",
    )
    parser.add_argument(
        "--conda-sh",
        default=str(default_conda_sh),
        help="Path to conda.sh used for activation.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run after environment setup.",
    )
    parser.add_argument(
        "--module-cuda",
        default="cuda/13.1.0",
        help="Optional CUDA module to load, e.g. cuda/12.1.0.",
    )
    parser.add_argument(
        "--nccl-socket-ifname",
        default="",
        help="Optional NCCL socket interface name. Leave empty to unset it inside the job.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to the training script. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the Slurm job file but do not submit it.",
    )
    return parser.parse_args(), repo_root


def build_train_command(args, repo_root: Path) -> list[str]:
    script_path = repo_root / args.script
    config_path = repo_root / args.config

    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    train_cmd = [
        args.python,
        "-m",
        "torch.distributed.run",
        f"--nnodes={args.nodes}",
        f"--nproc_per_node={args.gpus_per_node}",
        "--node_rank=${SLURM_NODEID}",
        "--master_addr=${MASTER_ADDR}",
        f"--master_port={args.master_port}",
        str(script_path),
        "--config_file",
        str(config_path),
    ]

    train_cmd.extend(args.extra_arg)
    return train_cmd


def format_command(parts: list[str]):
    return " ".join(shlex.quote(part) for part in parts)


def build_srun_command(args, train_cmd: str) -> str:
    return (
        f"srun --nodes={args.nodes} "
        f"--ntasks={args.nodes} "
        "--ntasks-per-node=1 "
        f"{train_cmd}"
    )


def build_job_script(args, repo_root: Path, out_file: Path, err_file: Path):
    total_gpus = args.nodes * args.gpus_per_node
    total_mem_gb = args.mem_per_gpu_gb * total_gpus
    gres = f"gpu:{args.gpus_per_node}"
    if args.gres_vram:
        gres = f"{gres},VRAM:{args.gres_vram}"

    train_cmd = format_command(build_train_command(args, repo_root))
    srun_cmd = build_srun_command(args, train_cmd)
    conda_sh = Path(args.conda_sh).expanduser()

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --output={out_file}",
        f"#SBATCH --error={err_file}",
        f"#SBATCH --nodes={args.nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={total_mem_gb}G",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --partition={args.partition}",
    ]

    if args.constraint:
        lines.append(f"#SBATCH --constraint={args.constraint}")
    if args.comment:
        lines.append(f'#SBATCH --comment="{args.comment}"')

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            "pwd; hostname; date",
            "nvidia-smi",
            'export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"',
            'export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"',
            'export SLURM_CPU_BIND="${SLURM_CPU_BIND:-none}"',
            'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"',
            'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton-${SLURM_JOB_ID}}"',
            'mkdir -p "${TRITON_CACHE_DIR}"',
            'MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
            f'export MASTER_PORT="${{MASTER_PORT:-{args.master_port}}}"',
            "",
        ]
    )

    if args.nccl_socket_ifname:
        lines.append(
            f"export NCCL_SOCKET_IFNAME={shlex.quote(args.nccl_socket_ifname)}"
        )
    else:
        lines.append("unset NCCL_SOCKET_IFNAME")

    lines.append("")

    if args.module_cuda:
        lines.append(f"module load {shlex.quote(args.module_cuda)}")

    if args.conda_env:
        lines.extend(
            [
                f"source {shlex.quote(str(conda_sh))}",
                f"conda activate {shlex.quote(args.conda_env)}",
                "",
            ]
        )

    lines.extend(
        [
            'echo "MASTER_ADDR=${MASTER_ADDR}"',
            'echo "MASTER_PORT=${MASTER_PORT}"',
            f'echo "Training on {args.nodes} node(s), {args.gpus_per_node} GPU(s) per node"',
            f'echo "TRAIN_CMD: {srun_cmd}"',
            "",
            srun_cmd,
            "",
        ]
    )

    return "\n".join(lines)


def main():
    args, repo_root = parse_args()
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%b-%d-%H-%M-%S")
    job_file = log_dir / f"{timestamp}.job"
    out_file = log_dir / f"{timestamp}.out"
    err_file = log_dir / f"{timestamp}.err"

    train_cmd = format_command(build_train_command(args, repo_root))
    srun_cmd = build_srun_command(args, train_cmd)
    job_script = build_job_script(args, repo_root, out_file, err_file)
    job_file.write_text(job_script, encoding="utf-8")

    print(f"Slurm job file written to: {job_file}")
    print(f"Stdout log: {out_file}")
    print(f"Stderr log: {err_file}")
    print(f"Tail logs with: tail -f {out_file}")
    print(f"Training cmd: {srun_cmd}")

    if args.dry_run:
        print("Dry run enabled, not submitting job.")
        return

    subprocess.run(["sbatch", str(job_file)], check=True)


if __name__ == "__main__":
    main()
