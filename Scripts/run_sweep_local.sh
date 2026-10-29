#!/bin/bash
# Parameter sweep for Project/mixed_annulus.py or Project/colliding_front.py on the local machine.
# Double loop: an outer model parameter (F_RES for mixed_annulus, W_R for colliding_front) and the
# antibiotic concentration, optionally repeated over several random seeds. Each run goes through
# Scripts/run_sim_server.py. All runs of one sweep are grouped in <out-root>/data/<model>-sweep-<date>/
# together with a log file per run.
#
# Usage: Scripts/run_sweep_local.sh MODEL [options]
#   MODEL          mixed_annulus or colliding_front
#   -t SECONDS     max simulation time per run (default: 600)
#   -c N           max cells per run (default: 100000)
#   -k NAME        name of the outer loop parameter (default: F_RES / W_R depending on MODEL)
#   -v "LIST"      values of the outer loop parameter (default: see below)
#   -a "LIST"      antibiotic values (default: "2.5 5 10")
#   -s "LIST"      seeds, one run per seed (default: "1")
#   -p NAME=VALUE  extra model parameter, passed on as --param, can be repeated
#   -e STEPS       pickle every STEPS steps (default: 100)
#   -n             dry run: only print the commands
#
# Examples:
#   Scripts/run_sweep_local.sh mixed_annulus -t 60 -v "0.1 0.2" -a "5 10"
#   Scripts/run_sweep_local.sh colliding_front -v "15 30 60" -p HEIGHT=300

# Python from the cellmodeller-thesis environment, override with: PYTHON=/path/to/python Scripts/run_sweep_local.sh ...
PYTHON=${PYTHON:-$HOME/.local/share/mamba/envs/cellmodeller-thesis/bin/python}
OUT_ROOT=${OUT_ROOT:-$HOME/cellmodeller}

# Defaults
MAX_TIME=300
MAX_CELLS=200000
ANTIBIOTICS="10"
SEEDS="1"
PICKLE_STEPS=100
EXTRA_PARAMS=()
DRY_RUN=0

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

MODEL=$1
case "$MODEL" in
    mixed_annulus)   LOOP_PARAM=F_RES; LOOP_VALUES="0.05 0.1 0.2" ;;
    colliding_front) LOOP_PARAM=ANTIBIOTIC_DIFF;   LOOP_VALUES="1600 2000 2400" ;;
    *) usage ;;
esac
shift

while getopts "t:c:k:v:a:s:p:e:n" opt; do
    case "$opt" in
        t) MAX_TIME=$OPTARG ;;
        c) MAX_CELLS=$OPTARG ;;
        k) LOOP_PARAM=$OPTARG ;;
        v) LOOP_VALUES=$OPTARG ;;
        a) ANTIBIOTICS=$OPTARG ;;
        s) SEEDS=$OPTARG ;;
        p) EXTRA_PARAMS+=(--param "$OPTARG") ;;
        e) PICKLE_STEPS=$OPTARG ;;
        n) DRY_RUN=1 ;;
        *) usage ;;
    esac
done

# Run from the repository root, so the relative model and script paths work from anywhere
cd "$(dirname "$0")/.." || exit 1

if [ ! -x "$PYTHON" ]; then
    echo "Python not found at $PYTHON - set PYTHON=/path/to/python"
    exit 1
fi
if [ ! -d "$OUT_ROOT/data" ]; then
    echo "Output folder $OUT_ROOT/data does not exist - create it first."
    exit 1
fi

# One folder per sweep, the runs are created inside it by run_sim_server.py (via --name SWEEP/RUN)
SWEEP=$MODEL-sweep-$(date +%y-%m-%d-%H-%M-%S)
SWEEP_DIR=$OUT_ROOT/data/$SWEEP

n_runs=0
for v in $LOOP_VALUES; do for ab in $ANTIBIOTICS; do for seed in $SEEDS; do
    n_runs=$((n_runs + 1))
done; done; done
echo "Sweep $SWEEP: $n_runs runs of up to $MAX_TIME s ($LOOP_PARAM = $LOOP_VALUES; ANTIBIOTIC = $ANTIBIOTICS; SEED = $SEEDS)"
echo "Output: $SWEEP_DIR"
[ $DRY_RUN -eq 0 ] && mkdir -p "$SWEEP_DIR/logs"

i=0
failed=0
for v in $LOOP_VALUES; do
    for ab in $ANTIBIOTICS; do
        for seed in $SEEDS; do
            i=$((i + 1))
            run=${LOOP_PARAM}_${v}-ANTIBIOTIC_${ab}-SEED_${seed}
            cmd=("$PYTHON" Scripts/run_sim_server.py "Project/$MODEL.py"
                 --max-time "$MAX_TIME" --max-cells "$MAX_CELLS" --pickle-steps "$PICKLE_STEPS"
                 --out-root "$OUT_ROOT" --name "$SWEEP/$run"
                 --param "$LOOP_PARAM=$v" --param "ANTIBIOTIC=$ab" --param "SEED=$seed"
                 ${EXTRA_PARAMS[@]+"${EXTRA_PARAMS[@]}"})

            if [ $DRY_RUN -eq 1 ]; then
                echo "${cmd[*]}"
                continue
            fi

            echo "[$i/$n_runs] $(date +%H:%M:%S) $run"
            if "${cmd[@]}" > "$SWEEP_DIR/logs/$run.log" 2>&1; then
                echo "    $(tail -n 1 "$SWEEP_DIR/logs/$run.log")"
            else
                failed=$((failed + 1))
                echo "    FAILED, see logs/$run.log: $(tail -n 1 "$SWEEP_DIR/logs/$run.log")"
            fi
        done
    done
done

[ $DRY_RUN -eq 1 ] && exit 0
echo "Finished $n_runs runs, $failed failed. Output: $SWEEP_DIR"
[ $failed -eq 0 ]
