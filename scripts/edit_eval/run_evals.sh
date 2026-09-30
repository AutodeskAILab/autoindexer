#!/usr/bin/env bash
# End-to-end edit eval for one checkpoint (teacher-forced, free-gen, tables).
set -euo pipefail

CKPT_NAME="${1:?usage: run_evals.sh <checkpoint-name> [data-dir] [--genonly] [--skip-noop] [--config KEY=VALUE ...] [-- run.py-args...]}"
shift

DATA_DIR="${DATA_DIR:-$HOME/data/cpt_datamix}"
DATA_DIR_SET=0
GENONLY=0
SKIP_NOOP=0
RUN_PY_ARGS=()
CONFIG_PAIRS=()

slugify_config_pair() {
  local pair="$1" key value
  key="${pair%%=*}"
  value="${pair#*=}"
  key="${key#config.}"
  key="${key//./-}"
  key="${key//_/-}"
  value="${value//./-}"
  value="${value//_/-}"
  value="${value//\//-}"
  printf '%s' "${key}-${value}"
}

config_suffix_from_pairs() {
  [[ $# -eq 0 ]] && return 0
  local -a slugs=()
  local pair slug joined
  for pair in "$@"; do
    slugs+=("$(slugify_config_pair "$pair")")
  done
  joined=$(printf '%s\n' "${slugs[@]}" | LC_ALL=C sort | paste -sd '_' -)
  printf '_%s' "$joined"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --genonly)
      GENONLY=1
      shift
      ;;
    --skip-noop)
      SKIP_NOOP=1
      RUN_PY_ARGS+=(--skip-noop)
      shift
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "error: --config requires KEY=VALUE" >&2; exit 1; }
      CONFIG_PAIRS+=("$2")
      RUN_PY_ARGS+=(--config "$2")
      shift 2
      ;;
    --)
      shift
      RUN_PY_ARGS+=("$@")
      break
      ;;
    *)
      if [[ "$DATA_DIR_SET" -eq 0 && "$1" != --* ]]; then
        DATA_DIR="$1"
        DATA_DIR_SET=1
        shift
      else
        RUN_PY_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

if [[ -n "${CONFIG_OVERRIDES:-}" ]]; then
  IFS=',' read -r -a _config_pairs <<< "$CONFIG_OVERRIDES"
  for pair in "${_config_pairs[@]}"; do
    pair="${pair#"${pair%%[![:space:]]*}"}"
    pair="${pair%"${pair##*[![:space:]]}"}"
    [[ -n "$pair" ]] || continue
    CONFIG_PAIRS+=("$pair")
    RUN_PY_ARGS+=(--config "$pair")
  done
fi

CONFIG_SUFFIX=""
if [[ ${#CONFIG_PAIRS[@]} -gt 0 ]]; then
  CONFIG_SUFFIX="$(config_suffix_from_pairs "${CONFIG_PAIRS[@]}")"
fi

S3_ROOT="${S3_ROOT:-}"
CKPT="${CKPT:-$S3_ROOT/$CKPT_NAME/final_checkpoint}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/edit_eval/${RUN_TS}_${CKPT_NAME}${CONFIG_SUFFIX}}"
CACHE_DIR="$REPO_ROOT/runs/edit_eval/cache"

WINDOW_LEN="${WINDOW_LEN:-512}"
TF_NUM_WINDOWS="${TF_NUM_WINDOWS:-100}"
FREE_GEN_NUM_WINDOWS="${FREE_GEN_NUM_WINDOWS:-100}"

SUITE_ARGS=(
  --suite
  --checkpoint "$CKPT"
  --out-dir "$OUT_DIR"
  --local-data-dir "$DATA_DIR"
  --num-val-samples-per-source 200
  --window-cache-dir "$CACHE_DIR"
  --window-len "$WINDOW_LEN"
  --mean-insert 5
  --mean-delete 5
  --tf-num-windows "$TF_NUM_WINDOWS"
  --free-gen-num-windows "$FREE_GEN_NUM_WINDOWS"
)
if [[ "$GENONLY" -eq 1 ]]; then
  SUITE_ARGS+=(--genonly)
fi
if [[ ${#RUN_PY_ARGS[@]} -gt 0 ]]; then
  SUITE_ARGS+=("${RUN_PY_ARGS[@]}")
fi

mkdir -p "$OUT_DIR"
echo "Output directory: $OUT_DIR"
if [[ ${#RUN_PY_ARGS[@]} -gt 0 ]]; then
  echo "Extra run.py args: ${RUN_PY_ARGS[*]}"
fi

if [[ "${SKIP_RUN:-0}" != "1" ]]; then
  echo "=== Running eval suite (single checkpoint load) ==="
  PYTHONPATH=src python "$REPO_ROOT/scripts/edit_eval/run.py" "${SUITE_ARGS[@]}"
fi

echo "=== Compiling tables and stats ==="
COMPILE_ARGS=("$OUT_DIR")
if [[ "$GENONLY" -eq 1 ]]; then
  COMPILE_ARGS+=(--genonly)
fi
if [[ "$SKIP_NOOP" -eq 1 ]]; then
  COMPILE_ARGS+=(--skip-noop)
fi
PYTHONPATH=src:scripts python "$REPO_ROOT/scripts/edit_eval/compile_results.py" "${COMPILE_ARGS[@]}"

echo
echo "Done. Results in $OUT_DIR:"
if [[ "$GENONLY" -eq 0 ]]; then
  echo "  TEACHER_FORCED.md   paper-format teacher-forced table"
  echo "  GALLERY.md          qualitative success/failure examples"
fi
echo "  FREE_GENERATION.md  paper-format free-gen table"
echo "  stats.json          raw numbers for recompiling tables"
