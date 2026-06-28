from __future__ import annotations

import argparse
import sys
from pathlib import Path

from probing_mpe.experiments.run_matrix import (
    load_matrix_config, build_training_command, discover_final_checkpoint, 
    _matrix_output, _export_command, _diagnostics_command, _behavioral_command,
    _checkpointed_diagnostics_command, _run_or_print, _run_dir, run_command,
    CHECKPOINTED_CONFIG_ID, DEFAULT_BENCHMARL_ROOT
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single training job for the replication plan.")
    parser.add_argument("--env", required=True, help="Environment name (e.g. simple_spread_v3)")
    parser.add_argument("--config", required=True, help="Config ID (e.g. ippo_rnn)")
    parser.add_argument("--seed", type=int, required=True, help="Seed value")
    parser.add_argument("--wandb", default="online", help="W&B mode (online, offline, disabled)")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/results"), help="Base output directory")
    parser.add_argument("--config-root", type=Path, default=Path("configs/reduced_mpe"), help="Root path for configs")
    parser.add_argument("--benchmarl-root", type=Path, default=DEFAULT_BENCHMARL_ROOT, help="Path to BenchMARL repo")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    config_path = args.config_root / args.env / f"{args.config}.yaml"
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        return 1

    matrix_config = load_matrix_config(config_path)
    run_dir = _run_dir(args.output_dir, args.env, args.config, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    wandb_enabled = args.wandb != "disabled"
    
    # build_training_command explicitly validates that parameter sharing is disabled
    training_command = build_training_command(
        matrix_config=matrix_config,
        benchmarl_root=args.benchmarl_root,
        run_dir=run_dir,
        seed=args.seed,
        python_executable=sys.executable,
        wandb_enabled=wandb_enabled,
    )
    
    print(f"Starting training run for {args.env} / {args.config} / seed {args.seed}...")
    _run_or_print(training_command, args.benchmarl_root, args.dry_run, run_command)
    
    if args.dry_run:
        checkpoint_path = run_dir / "checkpoint_final" / "checkpoint.pt"
    else:
        checkpoint_path = discover_final_checkpoint(run_dir)
        
    output = _matrix_output(
        matrix_config=matrix_config,
        seed=args.seed,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
    )
    
    print("Exporting trajectory...")
    _run_or_print(_export_command(output, sys.executable), None, args.dry_run, run_command)
    
    print("Computing diagnostics...")
    _run_or_print(_diagnostics_command(output, sys.executable), None, args.dry_run, run_command)
    
    print("Computing behavioral metrics...")
    _run_or_print(_behavioral_command(output, sys.executable), None, args.dry_run, run_command)
    
    if args.config == CHECKPOINTED_CONFIG_ID:
        print("Running checkpointed diagnostics for MAPPO-RNN...")
        _run_or_print(_checkpointed_diagnostics_command(output, sys.executable), None, args.dry_run, run_command)
        
    print(f"Run single completed successfully. Results in {run_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
