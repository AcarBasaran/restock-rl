#!/bin/bash
# Train DQN as a SLURM job array (D22: training only via SLURM). Each task trains one
# seed, saves a checkpoint every dqn.checkpoint_every steps, then scores every
# checkpoint greedily on the TUNING years (best_checkpoint.json).
#
# Submit all 5 seeds of the default config (array index = position in dqn.training_seeds):
#   sbatch scripts/train_dqn.slurm
# Tuning round 1 (D16): one array of 3 seeds per variant, e.g.
#   sbatch --array=0-2 --export=ALL,CONFIG=configs/tuning_round_1/v1.yaml,RUN_NAME=tuning1/v1 scripts/train_dqn.slurm
# Short timing job for one seed:
#   sbatch --array=0 --time=00:15:00 --export=ALL,TIMESTEPS=50000,RUN_NAME=timing scripts/train_dqn.slurm
#
# Account/partition: the CPU partitions (short/mid/long) only accept account "users",
# which this user does not have. The default course account may only use t4_ai,
# where jobs were estimated to wait ~8.5 h. Account "ai" on partition "ai" starts
# at once. No GPU is requested: a 2x256 MLP trains on CPU. See notes/open-questions.md.
#SBATCH --job-name=restock-dqn
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/%u/restock-rl/slurm/%x_%A_%a.out

set -euo pipefail

REPO="${REPO:-$HOME/restock-rl}"                  # code (read only during the job)
RUNS="${RUNS:-/scratch/$USER/restock-rl/runs}"    # outputs go to /scratch, as VALAR asks
RUN_NAME="${RUN_NAME:-dqn}"
TIMESTEPS="${TIMESTEPS:-}"                        # empty: use dqn.total_timesteps from the config
CONFIG="${CONFIG:-}"                              # empty: configs/default.yaml; relative paths are inside REPO
IDX="${SLURM_ARRAY_TASK_ID:-0}"
OUT="$RUNS/$RUN_NAME/seed_index_$IDX"

export PYTHONPATH="$REPO"
# git is a module on VALAR (not on every node's PATH); needed to record the commit.
set +u; module load git/2.9.5 >/dev/null 2>&1 || true; set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$OUT"
cd "$OUT"

echo "job ${SLURM_JOB_ID:-local} array ${SLURM_ARRAY_JOB_ID:-}-${IDX} on $(hostname) at $(date -Is)"
echo "repo commit: $(git -C "$REPO" rev-parse HEAD)"
if [[ -n "$CONFIG" && "$CONFIG" != /* ]]; then CONFIG="$REPO/$CONFIG"; fi
echo "config: ${CONFIG:-default}"
"$REPO/.venv/bin/python" -m src.train_dqn --seed-index "$IDX" --out-dir "$OUT" \
    ${CONFIG:+--config "$CONFIG"} ${TIMESTEPS:+--timesteps "$TIMESTEPS"}
echo "finished at $(date -Is)"
