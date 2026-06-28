# Probing MPE

## Overview
This repository contains a reduced replication of the paper "Probing Dec-POMDP Reasoning in Cooperative MARL" on Multi-Agent Particle Environments (MPE). It evaluates coordination and information flow in cooperative MARL models using information-theoretic diagnostics, ensuring no parameter sharing among agents.

## Supported Environments
- simple_spread_v3
- simple_speaker_listener_v4

## Supported Algorithms
- IPPO (Feed-forward and RNN)
- MAPPO (Feed-forward and RNN)
- Random Baseline (Uniform action sampling)

## Project Structure
- `configs/`: Configuration files for BenchMARL and experiment matrices.
- `docs/`: Replication plans and capability reports.
- `scripts/`: Executable scripts to run single experiments, matrices, or baselines.
- `src/probing_mpe/`: Source code for generating trajectories, evaluating metrics, and computing diagnostics.

## Running Experiments

### Full Matrix Run
To run the full suite of experiments across all configurations:
```bash
python scripts/run_matrix.py
```

### Single Experiment Run
To run a specific configuration (e.g., IPPO RNN on Simple Spread):
```bash
python scripts/run_single.py --env simple_spread_v3 --config ippo_rnn --seed 0
```

### Random Baseline
To generate baseline metrics using an untrained, purely random policy:
```bash
python scripts/run_random_baseline.py --env simple_spread_v3 --seed 0
```