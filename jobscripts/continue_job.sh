#!/bin/bash

# Check if an experiment name was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 DATA EXP_ID"
    exit 1
fi

# The first argument is the dataset
DATA="$1"
EXP_ID="$2"

# Source your environment variables
source ./env_vars.sh

# Path to the job script template
JOB_SCRIPT_TEMPLATE="./jobscript_template.sh"

# Temporary job script file that will be populated with environment variables and the experiment name
TEMP_JOB_SCRIPT="./jobscript_populated.sh"

RUN_EXPERIMENTS_ARGS="--device cuda --dataset ${DATA} --cexp ${EXP_ID} --wandb"

# Replace placeholders in the template with actual environment variable values and the experiment name
sed -e "s|\${VENV_PATH}|$VENV_PATH|g" \
    -e "s|\${WORKING_DIR}|$WORKING_DIR|g" \
    -e "s|\${EMAIL}|$EMAIL|g" \
    -e "s|\${QUEUE}|$QUEUE|g" \
    -e "s|\${WALLTIME}|$WALLTIME|g" \
    -e "s|\${GPU_MODE}|$GPU_MODE|g" \
    -e "s|\${NUM_CORES}|$NUM_CORES|g" \
    -e "s|\${MEM_GB}|$MEM_GB|g" \
    -e "s|\${EXPERIMENT_NAME}|$DATA|g" \
    -e "s|\${RUN_EXPERIMENTS_ARGS}|$RUN_EXPERIMENTS_ARGS|g" \
    "$JOB_SCRIPT_TEMPLATE" > "$TEMP_JOB_SCRIPT"

# Submit the job
bsub < "$TEMP_JOB_SCRIPT"

# Optionally, remove the temporary job script after submission
rm "$TEMP_JOB_SCRIPT"
