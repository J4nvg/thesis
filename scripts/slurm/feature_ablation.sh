#!/bin/bash
# Sensitivity experiment "feature ablation": selector value + feature-group ablation.
# Pilot: catboost_tweedie@global, DEFAULT CatBoost params (the one control variant
# `selected_tuned` uses the tuned params), TEST stage only.
# Design, variants, how to read the result: docs/feature_ablation/README.md
# The variant table (single source of truth): docs/feature_ablation/variants.tsv
#
# Run on the LOGIN node of the cluster from the repo root (~/reruns/thesis). It only
# copies a few small files and calls sbatch; every model run is a SLURM job.
#
#     bash scripts/slurm/feature_ablation.sh --list                  # print the parsed variant table
#     bash scripts/slurm/feature_ablation.sh --dry-run               # print mkdir/cp/sbatch lines, write nothing
#     bash scripts/slurm/feature_ablation.sh                         # seed the stores + submit every job
#     bash scripts/slurm/feature_ablation.sh --status                # queue + completed stages + failed logs
#     bash scripts/slurm/feature_ablation.sh --combo <name> <g1,g2> [--dry-run]  # round 2: one extra cumulative-path variant
#
# What it submits: for every variant V in the table, every RUNS item (model@paradigm)
# and every seed of V, one job
#     fabl-<V>-<model>-<paradigm>-s<seed>
# running `strikecast.cli.main run experiment=count ... stage=test` with the variant's
# Hydra overrides, into its own store $OUT_ROOT/V (seeded from $SRC_STORE: the shared
# feature selections, plus the tuned params for `tuned` variants). `defaults` variants
# get --allow-default-params. With the current table: 37 variants = 53 jobs.
#
# --combo <name> <g1,g2,...> submits ONE extra variant path_<name> with overrides
#     +sensitivity.feature_space.mode=groups +sensitivity.feature_space.groups=[g1,g2,...]
# (default params, seed 42 or $SEEDS, 06:00:00) and appends a line to SUBMISSION.md.
# Valid groups: strikes spatial conflict comms macro missile cyber weather calendar.
#
# SUBMISSION.md: a plain submit refuses if $OUT_ROOT/SUBMISSION.md exists (already
# submitted). With VARIANTS set (a subset) or with --combo it appends a new section
# instead, so you can submit more variants later without clobbering the record.
#
# Environment overrides:
#   SRC_STORE      store the inputs are copied from           (default runs_publication)
#   OUT_ROOT       parent of the per-variant stores            (default runs_feature_ablation)
#   PARTITION      SLURM partition                              (default GPU)
#   RUNS           space-separated model@paradigm items         (default catboost_tweedie@global)
#   SEEDS          comma-separated list REPLACING every variant's seeds column (default: use the table)
#   VARIANTS       space-separated subset of variant names      (default: all rows)
#   VARIANTS_FILE  the variant table                            (default docs/feature_ablation/variants.tsv)
#
# Logs: logs/slurm/feature_ablation/<jobname>-<jobid>.out (last line STRIKECAST_EXIT <code>).
set -euo pipefail

SRC_STORE="${SRC_STORE:-runs_publication}"
OUT_ROOT="${OUT_ROOT:-runs_feature_ablation}"
PARTITION="${PARTITION:-GPU}"
RUNS="${RUNS:-catboost_tweedie@global}"
SEEDS="${SEEDS:-}"
VARIANTS="${VARIANTS:-}"
VARIANTS_FILE="${VARIANTS_FILE:-docs/feature_ablation/variants.tsv}"
LOGDIR="logs/slurm/feature_ablation"
GROUPS_VALID="strikes spatial conflict comms macro missile cyber weather calendar"
MODE="${1:-submit}"

die() { echo "feature_ablation: $*" >&2; exit 1; }

[ -f scripts/slurm/job.sbatch ] || die "run me from the repo root (~/reruns/thesis)"
case "$MODE" in
  submit|--dry-run|--status|--combo|--list) ;;
  *) die "unknown mode '$MODE' (use --list, --dry-run, --status, --combo <name> <g1,g2,...>, or no argument)" ;;
esac

# ---- the variant table -------------------------------------------------------------
# Parallel arrays (bash 3.2 compatible): one entry per selected variant.
V_NAME=(); V_SEEDS=(); V_PARAMS=(); V_TIME=(); V_OVR=()

load_table() {
  [ -f "$VARIANTS_FILE" ] || die "$VARIANTS_FILE not found"
  local variant seeds params time family contrast overrides
  while IFS=$'\t' read -r variant seeds params time family contrast overrides || [ -n "$variant" ]; do
    case "$variant" in ''|'#'*|variant) continue ;; esac
    case "$params" in tuned|defaults) ;; *) die "$VARIANTS_FILE: variant '$variant': params must be tuned|defaults (got '$params')" ;; esac
    [ -n "$seeds" ] && [ -n "$time" ] && [ -n "$overrides" ] \
      || die "$VARIANTS_FILE: variant '$variant' has empty seeds/time/overrides column"
    if [ -n "$VARIANTS" ]; then
      case " $VARIANTS " in *" $variant "*) ;; *) continue ;; esac
    fi
    V_NAME+=("$variant")
    V_SEEDS+=("${SEEDS:-$seeds}")
    V_PARAMS+=("$params")
    V_TIME+=("$time")
    V_OVR+=("$overrides")
  done < "$VARIANTS_FILE"
  local v
  for v in $VARIANTS; do   # every requested name must exist
    local found=0 i
    for i in "${!V_NAME[@]}"; do [ "${V_NAME[$i]}" = "$v" ] && found=1; done
    [ "$found" = 1 ] || die "VARIANTS: '$v' is not a row of $VARIANTS_FILE"
  done
  [ "${#V_NAME[@]}" -gt 0 ] || die "no variants selected from $VARIANTS_FILE"
}

count_jobs() {  # prints the number of jobs for the loaded variants
  local n=0 i s nruns; local -a ss
  nruns=$(echo $RUNS | wc -w)
  for i in "${!V_NAME[@]}"; do
    IFS=, read -r -a ss <<< "${V_SEEDS[$i]}"
    n=$(( n + ${#ss[@]} * nruns ))
  done
  echo "$n"
}

# ---- --list (no checks beyond the table) -----------------------------------------------
if [ "$MODE" = "--list" ]; then
  load_table
  printf '%-24s %-8s %-9s %-9s %s\n' variant seeds params time overrides
  for i in "${!V_NAME[@]}"; do
    printf '%-24s %-8s %-9s %-9s %s\n' "${V_NAME[$i]}" "${V_SEEDS[$i]}" "${V_PARAMS[$i]}" "${V_TIME[$i]}" "${V_OVR[$i]}"
  done
  echo "${#V_NAME[@]} variants, $(count_jobs) jobs (RUNS: $RUNS)"
  exit 0
fi

# ---- --status ----------------------------------------------------------------------
if [ "$MODE" = "--status" ]; then
  load_table
  # round-2 path_* stores that are not rows of the table: report them too (1 seed unless SEEDS)
  if [ -z "$VARIANTS" ]; then
    for d in "$OUT_ROOT"/path_*; do
      [ -d "$d" ] || continue
      V_NAME+=("$(basename "$d")"); V_SEEDS+=("${SEEDS:-42}"); V_PARAMS+=(defaults); V_TIME+=("-"); V_OVR+=("(combo)")
    done
  fi
  nruns=$(echo $RUNS | wc -w)
  echo "== queue"
  squeue -u "$USER" -h -o "%.10i %.60j %.8T %.10M %R" | grep " fabl-" || echo "(nothing queued)"
  echo "== completed test stages (seed=*/state.json with status complete; expected = seeds x runs)"
  tot=0; exp=0
  for i in "${!V_NAME[@]}"; do
    store="$OUT_ROOT/${V_NAME[$i]}"
    IFS=, read -r -a ss <<< "${V_SEEDS[$i]}"
    e=$(( ${#ss[@]} * nruns ))
    n=0
    for f in "$store"/count/*/*/seed=*/state.json; do
      [ -f "$f" ] || continue
      case "$f" in */count/shared/*|*/count/tuning/*) continue ;; esac
      grep -q '"status": "complete"' "$f" && n=$((n + 1))
    done
    printf '%-26s %d / %d complete\n' "${V_NAME[$i]}" "$n" "$e"
    tot=$((tot + n)); exp=$((exp + e))
  done
  echo "total: $tot / $exp"
  echo "== logs without STRIKECAST_EXIT 0 (failed, OR STILL RUNNING / queued: check the queue above)"
  grep -L "STRIKECAST_EXIT 0" "$LOGDIR"/*.out 2>/dev/null | while read -r f; do
    echo "--- $f"; tail -n 5 "$f"; done || true
  exit 0
fi

# ---- --combo: build the one extra variant ------------------------------------------------
COMBO_LINE=""
if [ "$MODE" = "--combo" ]; then
  if [ "${4:-}" = "--dry-run" ]; then MODE="--dry-run"; set -- "$1" "$2" "$3"; fi
  [ "$#" -eq 3 ] || die "usage: --combo <name> <g1,g2,...> [--dry-run]"
  cname="$2"; cgroups="$3"
  case "$cname" in ''|*[!A-Za-z0-9_]*) die "combo name must match [A-Za-z0-9_]+ (got '$cname')" ;; esac
  IFS=, read -r -a gs <<< "$cgroups"
  [ "${#gs[@]}" -gt 0 ] || die "no groups given"
  for g in "${gs[@]}"; do
    case " $GROUPS_VALID " in *" $g "*) ;; *) die "unknown group '$g' (valid: $GROUPS_VALID)" ;; esac
  done
  V_NAME=("path_$cname"); V_SEEDS=("${SEEDS:-42}"); V_PARAMS=(defaults); V_TIME=("06:00:00")
  V_OVR=("+sensitivity.feature_space.mode=groups +sensitivity.feature_space.groups=[$cgroups]")
  COMBO_LINE="combo path_$cname = groups [$cgroups]"
else
  load_table
fi

# ---- preconditions -----------------------------------------------------------------
grep -q "class FeatureSpaceConfig" src/strikecast/config/schema.py \
  || die "this checkout has no feature-space switch: git pull (branch refactor) first"
if [ "$MODE" != "--dry-run" ]; then
  [ -x .venv/bin/python ] || die "no .venv: python3 scripts/slurm/submit_all.py --setup-only"
fi
ls "$SRC_STORE"/count/shared/feature_selection.*.json >/dev/null 2>&1 \
  || die "$SRC_STORE/count/shared has no feature selection (wrong SRC_STORE?)"
need_tuned=0
for i in "${!V_NAME[@]}"; do [ "${V_PARAMS[$i]}" = tuned ] && need_tuned=1; done
if [ "$need_tuned" = 1 ]; then
  for r in $RUNS; do
    [ -f "$SRC_STORE/count/tuning/${r%@*}/best_params.json" ] \
      || die "$SRC_STORE/count/tuning/${r%@*}/best_params.json missing (needed by a tuned variant; wrong SRC_STORE?)"
  done
fi
APPEND=0
if [ -n "$COMBO_LINE" ] || [ -n "$VARIANTS" ]; then APPEND=1; fi
if [ "$MODE" != "--dry-run" ] && [ "$APPEND" = 0 ] && [ -e "$OUT_ROOT/SUBMISSION.md" ]; then
  die "$OUT_ROOT/SUBMISSION.md exists: already submitted (use --status, set VARIANTS=\"...\" to add variants, or move $OUT_ROOT away)"
fi

run() { if [ "$MODE" = "--dry-run" ]; then echo "+ $*"; else "$@"; fi; }

# ---- seed each variant's store (idempotent) ------------------------------------------
seed_store() {  # store params
  local store="$1" params="$2" r m
  if [ -e "$store/count/shared" ]; then
    echo "store $store already seeded (count/shared exists): skipping"
    return 0
  fi
  run mkdir -p "$store/count"
  run cp -r "$SRC_STORE/count/shared" "$store/count/"
  if [ "$params" = tuned ]; then
    run mkdir -p "$store/count/tuning"
    for r in $RUNS; do
      m="${r%@*}"
      run cp -r "$SRC_STORE/count/tuning/$m" "$store/count/tuning/"
    done
  fi
}

# ---- submit ------------------------------------------------------------------------
LINES=()
NJOBS=0
submit() {  # variant store params time overrides seed model paradigm
  local variant="$1" store="$2" params="$3" tlim="$4" ovr="$5" seed="$6" model="$7" par="$8"
  local name="fabl-${variant}-${model}-${par}-s${seed}"
  local -a toks=()
  set -f; read -r -a toks <<< "$ovr"; set +f   # no globbing of [...] tokens
  local cmd=(sbatch --parsable --job-name="$name" --partition="$PARTITION"
             --cpus-per-task=16 --mem=48G --time="$tlim"
             --output="$LOGDIR/%x-%j.out"
             scripts/slurm/job.sbatch main -m strikecast.cli.main run
             experiment=count "model=$model" "paradigm=$par" stage=test "seed=$seed" tracking=noop
             "${toks[@]}")
  [ "$params" = defaults ] && cmd+=(--allow-default-params)
  cmd+=(--store-root "$store" -v)
  NJOBS=$((NJOBS + 1))
  if [ "$MODE" = "--dry-run" ]; then
    echo "+ ${cmd[*]}"
  else
    local id; id=$("${cmd[@]}")
    echo "$id  $name"
    LINES+=("| $id | $name | \`$store\` |")
  fi
}

run mkdir -p "$LOGDIR"
for i in "${!V_NAME[@]}"; do
  store="$OUT_ROOT/${V_NAME[$i]}"
  seed_store "$store" "${V_PARAMS[$i]}"
  IFS=, read -r -a ss <<< "${V_SEEDS[$i]}"
  for r in $RUNS; do
    for s in "${ss[@]}"; do
      submit "${V_NAME[$i]}" "$store" "${V_PARAMS[$i]}" "${V_TIME[$i]}" "${V_OVR[$i]}" "$s" "${r%@*}" "${r#*@}"
    done
  done
done

echo "${#V_NAME[@]} variants, $NJOBS jobs"
[ "$MODE" = "--dry-run" ] && exit 0

# ---- SUBMISSION.md -----------------------------------------------------------------
SUB="$OUT_ROOT/SUBMISSION.md"
mkdir -p "$OUT_ROOT"
COMMIT="$(git rev-parse HEAD)$(git diff --quiet || echo ' (DIRTY checkout)')"
if [ -e "$SUB" ]; then
  {
    echo
    echo "## Additional submission ($(date -Is))"
    echo
    echo "- commit: \`$COMMIT\`"
    [ -n "$COMBO_LINE" ] && echo "- $COMBO_LINE"
    echo "- variants: ${V_NAME[*]}"
    echo
    echo "| job | name | store |"
    echo "|---|---|---|"
    printf '%s\n' ${LINES[@]+"${LINES[@]}"}
  } >> "$SUB"
else
  {
    echo "# Feature ablation (submitted $(date -Is))"
    echo
    echo "- commit: \`$COMMIT\`"
    echo "- source store: \`$SRC_STORE\` (feature selections, and tuned params for tuned variants, copied from it)"
    echo "- runs: $RUNS; partition: $PARTITION"
    [ -n "$COMBO_LINE" ] && echo "- $COMBO_LINE"
    echo "- design and analysis: \`docs/feature_ablation/README.md\`"
    echo
    echo "| job | name | store |"
    echo "|---|---|---|"
    printf '%s\n' ${LINES[@]+"${LINES[@]}"}
  } > "$SUB"
fi
echo "wrote $SUB"
