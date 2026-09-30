#!/bin/bash
# Sensitivity rerun: count-GBDT TEST stage with a wider future-covariate window.
# Why + how to read the result: docs/audits/2026-09-30/FUTWIN_RERUN.md
#
# Run on the LOGIN node from the repo root (~/thesis-refactor). It only copies
# a few small files and calls sbatch; every model run is a SLURM job.
#
#     bash scripts/slurm/sensitivity_futwin.sh --dry-run   # print what would happen
#     bash scripts/slurm/sensitivity_futwin.sh             # seed the stores + submit 19 jobs
#     bash scripts/slurm/sensitivity_futwin.sh --status    # queue + completed stages
#
# What it runs (seed 42, the publication run's tuned params and feature selections):
#   runs_sensitivity_futwin/lags_8_7/     18 jobs: 6 count GBDTs x global/activity/local,
#                                         +sensitivity.future_covariate_lags=[8,7]
#   runs_sensitivity_futwin/control_2_7/  1 job: catboost_tweedie@global with [2,7] through
#                                         the same switch; must reproduce runs_publication
#
# Environment overrides: SRC_STORE (default runs_publication), OUT_ROOT (default
# runs_sensitivity_futwin), PARTITION (default GPU).
set -euo pipefail

SRC_STORE="${SRC_STORE:-runs_publication}"
OUT_ROOT="${OUT_ROOT:-runs_sensitivity_futwin}"
PARTITION="${PARTITION:-GPU}"
LOGDIR="logs/slurm/sensitivity_futwin"
MODELS=(lightgbm_poisson lightgbm_tweedie xgboost_poisson xgboost_tweedie catboost_poisson catboost_tweedie)
PARADIGMS=(global activity local)
MODE="${1:-submit}"

die() { echo "sensitivity_futwin: $*" >&2; exit 1; }

[ -f scripts/slurm/job.sbatch ] || die "run me from the repo root (~/thesis-refactor)"

if [ "$MODE" = "--status" ]; then
  echo "== queue"
  squeue -u "$USER" -h -o "%.10i %.40j %.8T %.10M %R" | grep "sens-futwin" || echo "(nothing queued)"
  echo "== completed test stages"
  for store in "$OUT_ROOT/lags_8_7" "$OUT_ROOT/control_2_7"; do
    n=$(grep -l '"status": "complete"' "$store"/count/*/*/seed=42/state.json 2>/dev/null | wc -l)
    echo "$store: $n complete"
  done
  echo "== failures (last lines of logs that did not exit 0)"
  grep -L "STRIKECAST_EXIT 0" "$LOGDIR"/*.out 2>/dev/null | while read -r f; do
    echo "--- $f"; tail -n 5 "$f"; done || true
  exit 0
fi

# ---- preconditions ---------------------------------------------------------
grep -q "class SensitivityConfig" src/strikecast/config/schema.py \
  || die "this checkout has no sensitivity switch: git pull (branch refactor) first"
[ -x .venv/bin/python ] || die "no .venv: python3 scripts/slurm/submit_all.py --setup-only"
for m in "${MODELS[@]}"; do
  [ -f "$SRC_STORE/count/tuning/$m/best_params.json" ] \
    || die "$SRC_STORE/count/tuning/$m/best_params.json missing (wrong SRC_STORE?)"
done
ls "$SRC_STORE"/count/shared/feature_selection.*.json >/dev/null 2>&1 \
  || die "$SRC_STORE/count/shared has no feature selection"
if [ "$MODE" != "--dry-run" ] && [ -e "$OUT_ROOT/SUBMISSION.md" ]; then
  die "$OUT_ROOT/SUBMISSION.md exists: already submitted (use --status, or move $OUT_ROOT away)"
fi

run() { if [ "$MODE" = "--dry-run" ]; then echo "+ $*"; else "$@"; fi; }

# ---- seed both stores with the publication inputs (small: ~20 MB) ------------
for store in "$OUT_ROOT/lags_8_7" "$OUT_ROOT/control_2_7"; do
  run mkdir -p "$store/count/tuning"
  run cp -r "$SRC_STORE/count/shared" "$store/count/"
  for m in "${MODELS[@]}"; do
    run cp -r "$SRC_STORE/count/tuning/$m" "$store/count/tuning/"
  done
done
run mkdir -p "$LOGDIR"

# ---- submit ------------------------------------------------------------------
hours_for() {  # configs/cluster/time_table.json x ~2, rounded up
  case "$1/$2" in
    catboost_*/local) echo "06:00:00" ;;
    xgboost_*/local)  echo "05:00:00" ;;
    xgboost_*/*)      echo "03:00:00" ;;
    *)                echo "02:00:00" ;;
  esac
}

LINES=()
submit() {  # store lags model paradigm
  local store="$1" lags="$2" model="$3" par="$4"
  local name="sens-futwin-${lags/,/_}-${model}-${par}"
  local cmd=(sbatch --parsable --job-name="$name" --partition="$PARTITION"
             --cpus-per-task=16 --mem=48G --time="$(hours_for "$model" "$par")"
             --output="$LOGDIR/%x-%j.out"
             scripts/slurm/job.sbatch main -m strikecast.cli.main run
             experiment=count "model=$model" "paradigm=$par" stage=test seed=42
             "+sensitivity.future_covariate_lags=[$lags]" tracking=noop
             --store-root "$store" -v)
  if [ "$MODE" = "--dry-run" ]; then
    echo "+ ${cmd[*]}"
  else
    local id; id=$("${cmd[@]}")
    echo "$id  $name"
    LINES+=("| $id | $name | \`$store\` |")
  fi
}

for m in "${MODELS[@]}"; do
  for p in "${PARADIGMS[@]}"; do
    submit "$OUT_ROOT/lags_8_7" "8,7" "$m" "$p"
  done
done
submit "$OUT_ROOT/control_2_7" "2,7" catboost_tweedie global

[ "$MODE" = "--dry-run" ] && exit 0

{
  echo "# Sensitivity rerun: future-covariate window (submitted $(date -Is))"
  echo
  echo "- commit: \`$(git rev-parse HEAD)\`$(git diff --quiet || echo ' (DIRTY checkout)')"
  echo "- source store: \`$SRC_STORE\` (tuned params + feature selections copied from it)"
  echo "- plan and analysis: \`docs/audits/2026-09-30/FUTWIN_RERUN.md\`"
  echo
  echo "| job | name | store |"
  echo "|---|---|---|"
  printf '%s\n' "${LINES[@]}"
} > "$OUT_ROOT/SUBMISSION.md"
echo "wrote $OUT_ROOT/SUBMISSION.md"
