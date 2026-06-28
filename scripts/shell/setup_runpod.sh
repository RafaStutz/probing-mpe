#!/bin/bash
set -e

echo "Starting RunPod environment setup..."

# Default base directory for RunPod, but allow override
BASE_DIR="${RUNPOD_BASE_DIR:-/workspace}"
echo "Using base directory: $BASE_DIR"

# 1. Install system dependencies if needed (assuming Ubuntu base)
echo "Installing system dependencies..."
apt-get update && apt-get install -y git python3-pip python3-venv || echo "Skipping apt-get (requires sudo/root). Ensure git, python3-pip, and python3-venv are installed."

# 2. Set up virtual environment
VENV_DIR="$BASE_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 3. Clone BenchMARL or install a pinned BenchMARL release
BENCHMARL_DIR="$BASE_DIR/BenchMARL"
if [ ! -d "$BENCHMARL_DIR" ]; then
    echo "Cloning BenchMARL into $BENCHMARL_DIR..."
    git clone https://github.com/facebookresearch/BenchMARL.git "$BENCHMARL_DIR"
    cd "$BENCHMARL_DIR"
    # Checkout the specific commit from the capability report
    git checkout 65d649d80e0bdcbdbe2c5d6a3f02dbfed8f0bec1
    pip install -e .
    cd - > /dev/null
else
    echo "BenchMARL already cloned at $BENCHMARL_DIR."
fi

# 4. Clone probing-dec-pomdps
PROBING_DIR="$BASE_DIR/probing-dec-pomdps"
if [ ! -d "$PROBING_DIR" ]; then
    echo "Cloning probing-dec-pomdps into $PROBING_DIR..."
    git clone https://github.com/KaleabTessera/probing-dec-pomdps.git "$PROBING_DIR"
    cd "$PROBING_DIR"
    pip install -e .
    cd - > /dev/null
else
    echo "probing-dec-pomdps already cloned at $PROBING_DIR."
fi

# 5. Install pinned dependencies (TorchRL, PettingZoo/MPE, WandB)
echo "Installing pinned dependencies..."
pip install "torchrl>=0.10,<0.12"
pip install "pettingzoo[mpe]>=1.24.3"
pip install wandb

# 6. Verification tests
echo "Running verification tests..."
python3 -c "
import sys

print('Checking PettingZoo MPE environments...')
try:
    from pettingzoo.mpe import simple_spread_v3, simple_speaker_listener_v4
    print('[OK] Environments simple_spread_v3 and simple_speaker_listener_v4 can be imported.')
except Exception as e:
    print(f'[ERROR] Failed to import MPE environments: {e}')
    sys.exit(1)

print('Checking BenchMARL configs and parameter sharing...')
try:
    import benchmarl
    from benchmarl.algorithms import IppoConfig, MappoConfig
    
    # Verify configs are available
    ippo = IppoConfig(share_param_critic=False)
    mappo = MappoConfig(share_param_critic=False)
    print('[OK] IPPO and MAPPO configs are available and accept share_param_critic=False.')
    
    # Verify experiment config options
    from benchmarl.experiment import ExperimentConfig
    exp = ExperimentConfig(share_policy_params=False, prefer_continuous_actions=False)
    print('[OK] ExperimentConfig accepts share_policy_params=False and prefer_continuous_actions=False.')
    
except Exception as e:
    print(f'[ERROR] Failed BenchMARL verification: {e}')
    sys.exit(1)

print('[OK] All dependency and configuration verifications passed.')
"

# 7. Verify W&B login status
echo "Verifying W&B login status..."
wandb status || echo "[WARNING] W&B is offline or not logged in. Run 'wandb login' to authenticate if you want to sync to cloud."

echo "Setup complete! Environment is ready for training."
