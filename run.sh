#!/bin/bash

PRJ_NAME="nmr-to-structure-lite"

set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands and their arguments as they are executed


function log_error_and_exit {
  echo "Error: $1"
  exit 1
}

# Read command line argument for the experiment name
EXPERIMENT_NAME=$1
[ -z "${EXPERIMENT_NAME}" ] && log_error_and_exit "Experiment name is not set"


## PREPARATIONS

# Load settings for the experiment
source "./experiments/${EXPERIMENT_NAME}/envs.sh"

# Check that environment variables are set
for VAR in DATASET_URL; do
  [ -z "${!VAR}" ] && log_error_and_exit "Variable ${VAR} is not set"
done


# Depending on the host, select target path
# E.g., on host 'fw':
if [ "$(hostname)" == "fw" ]; then
  export HSA_OVERRIDE_GFX_VERSION=11.0.0

  CONDA="${HOME}/Programs/miniconda3"

  DATA_PATH="${HOME}/tmp"
  MACHINE="fw"

elif [[ "$(hostname)" == "lambda" ]]; then
  sudo apt install libarchive-tools

  CONDA="${HOME}/fs/miniconda"

  DATA_PATH="${HOME}/fs/tmp"
  MACHINE="lambda"
else
  log_error_and_exit "Unknown host: $(hostname)"
fi

DATA_PATH="${DATA_PATH}/${PRJ_NAME}/${EXPERIMENT_NAME}/${MACHINE}/$(date +'%Y%m%d-%H%M%S')"
mkdir -p "${DATA_PATH}"


## PART A: Setup the python environment

# We assume that a base environment is already set up
BASE_ENV="${PRJ_NAME}"

source "${CONDA}/bin/activate" "${BASE_ENV}"
conda activate "${BASE_ENV}"  # just in case

#EXP_ENV="${PRJ_NAME}_${EXPERIMENT_NAME}"
#
## Now copy the environment to the target path
#conda create --name ${EXP_ENV} --use-local --override-channels -c conda-forge
#conda activate ${EXP_ENV}
#
## Install/check the requirements
#pip install -r requirements.txt


## PART B: Download the dataset

DATASET_PATH="${DATA_PATH}/data"

# Check if there are any *.txt files in the top-level of DATASET_PATH
if [ -n "$(ls -A "${DATASET_PATH}"/*.txt 2>/dev/null)" ]; then
  echo "Dataset already exists in ${DATASET_PATH}"
else
  mkdir -p "${DATASET_PATH}"

  echo "Downloading the dataset..."
  wget -qO- "${DATASET_URL}" | bsdtar -xvf- -C "${DATASET_PATH}"
  echo "Dataset downloaded to ${DATASET_PATH}"

  # Move only *-*.txt files from subdirectories to the top-level directory
  find "${DATASET_PATH}" -mindepth 2 -type f -name "*-*.txt" -exec mv {} "${DATASET_PATH}/" \;

  # Remove empty subdirectories
  find "${DATASET_PATH}" -mindepth 1 -type d -empty -delete
fi


RUN_PATH="${DATA_PATH}/run"
mkdir -p ${RUN_PATH}

CONFIG_FILE="${DATA_PATH}/run/config.yaml"

## PART C: Complete the `transformer_template.yaml` file

cp ./experiments/${EXPERIMENT_NAME}/transformer_template.yaml "${CONFIG_FILE}"

sed -i "s|^[[:space:]]*save_data:.*|save_data: \"${RUN_PATH}\"|" "${CONFIG_FILE}"
sed -i "s|^[[:space:]]*src_vocab:.*|src_vocab: \"${RUN_PATH}/src_vocab.txt\"|" "${CONFIG_FILE}"
sed -i "s|^[[:space:]]*tgt_vocab:.*|tgt_vocab: \"${RUN_PATH}/tgt_vocab.txt\"|" "${CONFIG_FILE}"
sed -i "/corpus_1:/,/valid:/ s|^\([[:space:]]*\)path_src:[[:space:]]*{}.*|\1path_src: \"${DATASET_PATH}/src-train.txt\"|" "${CONFIG_FILE}"
sed -i "/corpus_1:/,/valid:/ s|^\([[:space:]]*\)path_tgt:[[:space:]]*{}.*|\1path_tgt: \"${DATASET_PATH}/tgt-train.txt\"|" "${CONFIG_FILE}"
sed -i "/valid:/,/^$/ s|^\([[:space:]]*\)path_src:[[:space:]]*{}.*|\1path_src: \"${DATASET_PATH}/src-val.txt\"|" "${CONFIG_FILE}"
sed -i "/valid:/,/^$/ s|^\([[:space:]]*\)path_tgt:[[:space:]]*{}.*|\1path_tgt: \"${DATASET_PATH}/tgt-val.txt\"|" "${CONFIG_FILE}"
sed -i "s|^[[:space:]]*tensorboard_log_dir:.*|tensorboard_log_dir: \"${RUN_PATH}/tensorboard\"|" "${CONFIG_FILE}"
sed -i "s|^[[:space:]]*save_model:.*|save_model: \"${RUN_PATH}/model\"|" "${CONFIG_FILE}"

# Check that no "{}" are left in the file
grep -q "{}" "${CONFIG_FILE}" && log_error_and_exit "Unresolved placeholders in ${CONFIG_FILE}"

#rm -f "${RUN_PATH}/*.log" "${RUN_PATH}/*.err"


## PART D: Build the vocabulary

if [ -f "${RUN_PATH}/src_vocab.txt" ] && [ -f "${RUN_PATH}/tgt_vocab.txt" ]; then
  echo "Vocabulary files already exist in ${RUN_PATH}"
else
  onmt_build_vocab -config "${CONFIG_FILE}" -n_sample 10000
fi


## PART E: Train the model

#onmt_train -config "${CONFIG_FILE}" 2>&1

onmt_train -config "${CONFIG_FILE}" 2>&1 | \
  grep -Ev "Weighted corpora loaded so far|corpus_1:" | \
  grep -Ev "FutureWarning|def (forward|backward)" | \
  tee "${RUN_PATH}/train.log"
