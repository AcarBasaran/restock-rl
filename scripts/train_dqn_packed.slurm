#!/bin/bash
# Several DQN seeds in ONE SLURM job, as parallel processes with one CPU core and
# one thread each. Same training as scripts/train_dqn.slurm (same src.train_dqn,
# config, checkpointing and tuning-year scoring); it only uses fewer job slots
# (the ai QOS allows 8 running jobs per user).
#
# CODE is a checkout of the commit to run (e.g. a git worktree pinned at the
# commit the rest of a tuning round used), so run.json records that commit.
#   sbatch --cpus-per-task=3 --mem=6G \
#          --export=ALL,CODE=/scratch/$USER/restock-rl/code-bc48e5c,CONFIG=configs/tuning_round_1/v3.yaml,RUN_NAME=tuning1/v3,SEED_INDICES="0 1 2" \
#          scripts/train_dqn_packed.slurm
#
# Threads: src.train_dqn calls torch.set_num_threads(dqn.torch_threads) (2 in
# the config). To run 1 thread per process without changing the code or the
# config, each process starts through a small launcher that sets torch to 1
# thread and turns later set_num_threads calls into no-ops. Thread count does
# not change the algorithm; it can only change floating-point summation order.
#SBATCH --job-name=restock-packed
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/restock-rl/slurm/%x_%j.out

set -euo pipefail

CODE="${CODE:?set CODE to the code checkout to run}"
PY="${PY:-$HOME/restock-rl/.venv/bin/python}"
RUNS="${RUNS:-/scratch/$USER/restock-rl/runs}"
CONFIG="${CONFIG:?set CONFIG (path relative to CODE)}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
SEED_INDICES="${SEED_INDICES:-0 1 2}"

export PYTHONPATH="$CODE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set +u; module load git/2.9.5 >/dev/null 2>&1 || true; set -u

LAUNCH='import runpy, sys, torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_num_threads = lambda n: None  # keep 1 thread per process (config: dqn.torch_threads = 2)
sys.argv = ["src.train_dqn"] + sys.argv[1:]
try:
    runpy.run_module("src.train_dqn", run_name="__main__", alter_sys=True)
finally:
    print("torch threads in this process:", torch.get_num_threads(), flush=True)'

echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is), cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "code: $CODE at commit $(git -C "$CODE" rev-parse HEAD)"
echo "config: $CONFIG, run: $RUN_NAME, seed indices: $SEED_INDICES"

pids=()
for i in $SEED_INDICES; do
    OUT="$RUNS/$RUN_NAME/seed_index_$i"
    if [[ -e "$OUT" ]]; then echo "refusing: $OUT already exists" >&2; exit 1; fi
    mkdir -p "$OUT"
    (cd "$OUT" && exec "$PY" -c "$LAUNCH" --seed-index "$i" --out-dir "$OUT" --config "$CODE/$CONFIG" \
        > "$OUT/train.log" 2>&1) &
    pids+=($!)
    echo "seed index $i: pid $!, log $OUT/train.log"
done

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
for i in $SEED_INDICES; do
    echo "== seed index $i"; grep -E "^done:|^best checkpoint|^torch threads|Error" "$RUNS/$RUN_NAME/seed_index_$i/train.log" || true
done
echo "finished at $(date -Is), status $status"
exit $status
