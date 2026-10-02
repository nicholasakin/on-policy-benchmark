#!/usr/bin/env bash
# Paper-1 reproduction driver ("The Surprising Effectiveness of PPO in Cooperative,
# Multi-Agent Games", Yu et al., arxiv 2103.01955) on MPE, using this repo's own
# reference MAPPO/IPPO implementation and the paper's own defaults (mirrors
# onpolicy/scripts/train_mpe_scripts/train_mpe_spread.sh, made configurable).
#
# Gotchas baked into the flags below -- these bit us once, see onpolicy/config.py:
#   --use_ReLU   action='store_false', default True. Passing it SELECTS TANH, not ReLU.
#                This is what the paper's own script does (Table 16: activation=Tanh
#                for Spread) -- kept here on purpose, don't "fix" it away.
#   --cuda       action='store_false', default True (GPU). Passing it FORCES CPU.
#                Controlled here by USE_GPU; leave USE_GPU=1 (default) for GPU.
#   --use_wandb  action='store_false', default True (wandb). Passing it logs to
#                tensorboard instead. Kept on by default here so you don't need a
#                wandb login; set USE_WANDB=1 to actually use wandb.
#
# Every knob is overridable via environment variable, e.g.:
#   NUM_AGENTS=2 NUM_LANDMARKS=2 NUM_ENV_STEPS=2000000 SEED=2 ./run_repro.sh
#   ALGO=rmappo EXTRA_ARGS="--use_centralized_V" EXPERIMENT_NAME=ippo_check ./run_repro.sh
#   EXTRA_ARGS="--use_valuenorm" ./run_repro.sh
set -euo pipefail
cd "$(dirname "$0")/onpolicy/scripts/train"
source ../../../.venv/bin/activate

ALGO="${ALGO:-rmappo}"                             # rmappo | mappo | ippo | happo | hatrpo | mat | mat_dec
SCENARIO="${SCENARIO:-simple_spread}"
NUM_AGENTS="${NUM_AGENTS:-3}"
NUM_LANDMARKS="${NUM_LANDMARKS:-3}"
SEED="${SEED:-1}"

N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-128}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-20000000}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"
PPO_EPOCH="${PPO_EPOCH:-10}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
LR="${LR:-7e-4}"
CRITIC_LR="${CRITIC_LR:-7e-4}"

USE_EVAL="${USE_EVAL:-1}"
N_EVAL_ROLLOUT_THREADS="${N_EVAL_ROLLOUT_THREADS:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-paper_repro}"
USER_NAME="${USER_NAME:-reproduction}"
USE_GPU="${USE_GPU:-1}"                            # 1 = GPU (default), 0 = force CPU
USE_WANDB="${USE_WANDB:-0}"                         # 0 = tensorboard (default), 1 = wandb
EXTRA_ARGS="${EXTRA_ARGS:-}"                        # any extra passthrough flags, e.g. "--use_valuenorm"

CUDA_FLAG=""
[ "$USE_GPU" = "0" ] && CUDA_FLAG="--cuda"          # store_false: passing it flips GPU OFF

WANDB_FLAG=""
[ "$USE_WANDB" = "0" ] && WANDB_FLAG="--use_wandb"  # store_false: passing it flips wandb OFF (-> tensorboard)

EVAL_FLAGS=""
if [ "$USE_EVAL" = "1" ]; then
  EVAL_FLAGS="--use_eval --n_eval_rollout_threads $N_EVAL_ROLLOUT_THREADS --eval_interval $EVAL_INTERVAL"
fi

set -x
python train_mpe.py --env_name MPE --algorithm_name "$ALGO" --experiment_name "$EXPERIMENT_NAME" \
  --scenario_name "$SCENARIO" --num_agents "$NUM_AGENTS" --num_landmarks "$NUM_LANDMARKS" --seed "$SEED" \
  --n_training_threads 1 --n_rollout_threads "$N_ROLLOUT_THREADS" --num_mini_batch "$NUM_MINI_BATCH" \
  --episode_length "$EPISODE_LENGTH" --num_env_steps "$NUM_ENV_STEPS" \
  --ppo_epoch "$PPO_EPOCH" --use_ReLU --gain 0.01 --lr "$LR" --critic_lr "$CRITIC_LR" \
  $EVAL_FLAGS $WANDB_FLAG --user_name "$USER_NAME" $CUDA_FLAG $EXTRA_ARGS
