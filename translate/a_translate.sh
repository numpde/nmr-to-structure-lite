#!/bin/bash

# First argument is the path to the experiment folder, e.g., ./results/001_big/lambda/20250309-220000/
# Second argument is the checkpoint ID, e.g., 100000

# ROCm hack
export HSA_OVERRIDE_GFX_VERSION=11.0.0


function log_error_and_exit {
  echo "Error: $1"
  exit 1
}

THIS_PATH=$(dirname "$(realpath -s "$0")")
echo "Script location $THIS_PATH"

# Assert that $THIS_PATH/../results exists
RESULTS_PATH="${THIS_PATH}/../results"
RESULTS_PATH=$(realpath -s "${RESULTS_PATH}")
[ ! -d "${RESULTS_PATH}" ] && log_error_and_exit "Results path not found: ${RESULTS_PATH}"

# Read command line argument for the experiment name
EXPERIMENT_PATH=$(realpath "$1")
[ -z "${EXPERIMENT_PATH}" ] && log_error_and_exit "Experiment path is not set"

DATA_PATH="${EXPERIMENT_PATH}/data"
DATA_PATH=$(realpath -s "${DATA_PATH}")
[ ! -d "${DATA_PATH}" ] && log_error_and_exit "Data path not found: ${DATA_PATH}"

RUN_PATH="${EXPERIMENT_PATH}/run"
RUN_PATH=$(realpath -s "${RUN_PATH}")
[ ! -d "${RUN_PATH}" ] && log_error_and_exit "Run path not found: ${RUN_PATH}"

# Check that both are subfolders of RESULTS_PATH
[[ ! "${DATA_PATH}" =~ ^"${RESULTS_PATH}" ]] && log_error_and_exit "Data path not in ${RESULTS_PATH}: ${DATA_PATH}"
[[ ! "${RUN_PATH}" =~ ^"${RESULTS_PATH}" ]] && log_error_and_exit "Run path not in ${RESULTS_PATH}: ${RUN_PATH}"

EXPERIMENT_PATH_SUFFIX=${EXPERIMENT_PATH#${RESULTS_PATH}/}
echo "Experiment path suffix: ${EXPERIMENT_PATH_SUFFIX}"

# This script's filename (no extension):
SELF_NAME=$(basename "$0" .sh)

WORK_PATH="${THIS_PATH}/${SELF_NAME}/${EXPERIMENT_PATH_SUFFIX}"
mkdir -p "${WORK_PATH}"

WORK_DATA_PATH="${WORK_PATH}/data"
mkdir -p "${WORK_DATA_PATH}"

WORK_TRANSLATION_PATH="${WORK_PATH}/translation"
mkdir -p "${WORK_TRANSLATION_PATH}"

# Sample from the test set
N=1000 # Number of samples
WORK_SRC_TST="${WORK_DATA_PATH}/src-test_n$N.txt"
WORK_TGT_TST="${WORK_DATA_PATH}/tgt-test_n$N.txt"
paste "${DATA_PATH}/src-test.txt" "${DATA_PATH}/tgt-test.txt" | \
  shuf --random-source=<(yes 42) -n "$N" | \
  tee >(cut -f1 > "${WORK_SRC_TST}") >(cut -f2 > "${WORK_TGT_TST}") > /dev/null


[ ! -z "$2" ] && CHECKPOINT="_$2" || CHECKPOINT="_100000"
CHECKPOINT=$(find "${RUN_PATH}" -name "model_step_*.pt" | grep "${CHECKPOINT}.pt" | sort -V | tail -n 1)

# If no checkpoint is found, abort:
[ "$CHECKPOINT" ] || log_error_and_exit "No checkpoint found."

# Filename (no extension)
CHECKPOINT_NAME=$(basename "${CHECKPOINT}" .pt)

echo "Using checkpoint: ${CHECKPOINT_NAME}"

# Translate
NBEST=10
BEAM_SIZE=20
OUTPUT_FILE_NAME="$(basename ${WORK_TGT_TST})__${CHECKPOINT_NAME}__n_best=${NBEST}__beam_size=${BEAM_SIZE}.txt"
OUTPUT_FILE="${WORK_TRANSLATION_PATH}/${OUTPUT_FILE_NAME}"

LOG_FILE="${OUTPUT_FILE}.log"

if [ -f "${OUTPUT_FILE}" ] && [ -f "${LOG_FILE}" ]; then
  echo "Output file and log file already exist:"
  echo " - ${OUTPUT_FILE#${WORK_TRANSLATION_PATH}/}"
  echo " - ${LOG_FILE#${WORK_TRANSLATION_PATH}/}"
  exit 0
fi

date "+%Y-%m-%d %H:%M:%S" > "${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT#${RESULTS_PATH}/}" >> "${LOG_FILE}"
echo "Source: ${WORK_SRC_TST#${THIS_PATH}/}" >> "${LOG_FILE}"
echo "Output: ${OUTPUT_FILE#${THIS_PATH}/}" >> "${LOG_FILE}"
echo "n_best=${NBEST}" >> "${LOG_FILE}"
echo "beam_size=${BEAM_SIZE}" >> "${LOG_FILE}"
echo "== Start ==" >> "${LOG_FILE}"

echo "Log file: ${LOG_FILE#${WORK_TRANSLATION_PATH}/}"

onmt_translate 2>&1 \
  --model "${CHECKPOINT}" \
  --src "${WORK_SRC_TST}" \
  --output "${OUTPUT_FILE}" \
  --n_best $NBEST \
  --beam_size $BEAM_SIZE \
  --gpu -1 \
  --verbose | \
  grep -v "FutureWarning" | \
  tee -a "${LOG_FILE}"

#sed 's/ //g' 001_big_pre-test_n30.txt > 001_big_pre-test_n30.txt.no-space.txt
#sed 's/ //g' 001_big_tgt-test_n30.txt > 001_big_tgt-test_n30.txt.no-space.txt

# Note, beam_size=777 requires about 85GB of RAM
