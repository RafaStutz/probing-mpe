#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the random baseline pipeline for a given environment.")
    parser.add_argument("--env", required=True, help="Environment name (e.g., simple_spread_v3)")
    parser.add_argument("--seed", type=int, required=True, help="Seed value")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Base output directory")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser.parse_args()

def run_command(command: list[str], dry_run: bool) -> None:
    if dry_run:
        print(" ".join(command))
    else:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)

def main() -> int:
    args = parse_args()
    
    run_dir = args.output_dir / args.env / "random_baseline" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory_path = run_dir / "trajectory_eval_final.pkl"
    diagnostics_path = run_dir / "diagnostics_final.json"
    null_diagnostics_path = run_dir / "diagnostics_null_final.json"
    behavioral_metrics_path = run_dir / "behavioral_metrics_final.json"
    
    repo_root = Path(__file__).resolve().parents[1]
    
    print(f"Starting random baseline for {args.env} (Seed {args.seed})...")
    
    # 1. Generate random trajectory
    cmd_generate = [
        args.python_executable,
        "-m", "probing_mpe.random_baseline",
        "--env", args.env,
        "--output", str(trajectory_path),
        "--seed", str(args.seed)
    ]
    # Set PYTHONPATH to include src
    env_kwargs = {}
    if not args.dry_run:
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_root / 'src'}:{env.get('PYTHONPATH', '')}"
        env_kwargs = {"env": env}
        
    if args.dry_run:
        print(" ".join(cmd_generate))
    else:
        print(f"Running: {' '.join(cmd_generate)}")
        subprocess.run(cmd_generate, check=True, **env_kwargs)
        
    # 2. Compute diagnostics
    cmd_diag = [
        args.python_executable,
        str(repo_root / "scripts" / "compute_diagnostics_from_trajectory.py"),
        "--trajectory", str(trajectory_path),
        "--diagnostics-output", str(diagnostics_path),
        "--null-output", str(null_diagnostics_path)
    ]
    run_command(cmd_diag, args.dry_run)
    
    # 3. Compute behavioral metrics
    cmd_behav = [
        args.python_executable,
        str(repo_root / "scripts" / "compute_behavioral_metrics.py"),
        "--trajectory", str(trajectory_path),
        "--output", str(behavioral_metrics_path)
    ]
    run_command(cmd_behav, args.dry_run)
    
    print(f"\nRandom baseline completed successfully! Results in {run_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
