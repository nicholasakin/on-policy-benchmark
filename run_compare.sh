#!/usr/bin/env bash
# Isolates the worker-count hypothesis using the reference MAPPO implementation itself,
# so our PPO code is entirely out of the comparison. Two conditions, both algorithms:
#   worker4   -> n_rollout_threads=4  (matches our ppo_baselines.py setup)
#   worker128 -> n_rollout_threads=128 (the paper's own default, per
#                onpolicy/scripts/train_mpe_scripts/train_mpe_spread.sh)
# n_rollout_threads is the only axis varied. Everything else below is pinned to the
# paper's own MPE Spread setup (Yu et al. 2103.01955, Table 16 + train_mpe_spread.sh):
#   num_agents=3 num_landmarks=3, episode_length=25, num_env_steps=20e6, ppo_epoch=10,
#   num_mini_batch=1, lr=critic_lr=7e-4, gain=0.01, Tanh activation (--use_ReLU flips
#   the store_false default OFF), hidden_size=64/layer_N=1/recurrent GRU (all config.py
#   defaults, untouched).
#
# This repo's --algorithm_name choices are rmappo/mappo/happo/hatrpo/mat/mat_dec --
# 'ippo' isn't a valid choice in this checked-out version (config.py's argparse
# `choices` rejects it) even though train_mpe.py's own branch still special-cases it:
#   elif algorithm_name == "ippo": use_centralized_V = False   (recurrence untouched)
# So IPPO here = --algorithm_name rmappo (keeps the paper's recurrent GRU network,
# same as MAPPO) + --use_centralized_V (flips centralized-critic OFF), which
# reproduces exactly what that dead branch would have set. Earlier versions of this
# script used --algorithm_name mappo for "ippo", which silently switches to a
# non-recurrent MLP -- NOT what the paper's IPPO baseline uses. Fixed here.
#
# --cuda is store_false (default True/GPU); earlier versions of this script passed it
# unconditionally, which FORCED CPU for every run. Fixed: omit it so GPU is used.
set -euo pipefail
cd "$(dirname "$0")/onpolicy/scripts/train"
source ../../../.venv/bin/activate
STEPS=${1:-20000000}
EPISODE_LENGTH=25
SEED=${2:-1}

run() {
  # exp encodes both worker count AND algo label -- algorithm_name is "rmappo" for
  # both conditions (see comment above), so without this the two runs land in the
  # same results/.../rmappo/<exp>/ dir as indistinguishable run1/run2 folders.
  local label=$1 workers=$2 workers_tag=$3 gpu=$4 extra=$5
  local exp="${workers_tag}_${label}"
  local episodes=$(( STEPS / EPISODE_LENGTH / workers ))
  local eval_interval=$(( episodes / 250 ))
  [ "$eval_interval" -lt 1 ] && eval_interval=1
  CUDA_VISIBLE_DEVICES="$gpu" python train_mpe.py --env_name MPE --algorithm_name rmappo --experiment_name "$exp" \
    --scenario_name simple_spread --num_agents 3 --num_landmarks 3 --seed "$SEED" \
    --n_training_threads 1 --n_rollout_threads "$workers" --num_mini_batch 1 \
    --episode_length "$EPISODE_LENGTH" --num_env_steps "$STEPS" --ppo_epoch 10 \
    --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
    --eval_interval "$eval_interval" --use_eval --n_eval_rollout_threads 20 \
    --use_wandb --user_name reproduction $extra \
    > "../../../results_compare/${exp}.log" 2>&1
}

# matched-worker-count pair: run together on separate GPUs
run mappo 4   worker4  0 "" &
run ippo  4   worker4  1 "--use_centralized_V" &
wait

# reference-default pair: run sequentially, 128 subprocess envs each is already a full machine
run mappo 128 worker128 0 ""
run ippo  128 worker128 0 "--use_centralized_V"
