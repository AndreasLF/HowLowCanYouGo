#!/bin/bash

# Check if an experiment name was provided
if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
    echo "Usage: $0 EXPERIMENT_NAME [LOSS_TYPE MODEL_TYPE]"
    exit 1
fi

# The first argument is the experiment name
EXPERIMENT_NAME="$1"

# Source your environment variables
source ./env_vars.sh

# Path to the job script template
JOB_SCRIPT_TEMPLATE="./jobscript_template1.sh"

# Temporary job script file that will be populated with environment variables and the experiment name
TEMP_JOB_SCRIPT="./jobscript_populated.sh"

# Optionally provide two additional arguments for the loss_type and the model_type
if [ "$#" -eq 3 ]; then
    # Check if they are valid loss and model types, loss_type can be one of "logistic", "hinge", "poisson"
    # and model_type can be one of "L2", "PCA"

    if [ "$2" != "logistic" ] && [ "$2" != "hinge" ] && [ "$2" != "poisson" ]; then
        echo "Invalid loss type. Must be one of 'logistic', 'hinge', 'poisson'"
        exit 1
    fi

    if [ "$3" != "L2" ] && [ "$3" != "PCA" ]; then
        echo "Invalid model type. Must be one of 'L2', 'PCA'"
        exit 1
    fi

    LOSS_TYPE="$2"
    MODEL_TYPE="$3"

    # Assign the loss and model types to variables
    RUN_EXPERIMENTS_ARGS="--device cuda --experiment ${EXPERIMENT_NAME} --loss ${LOSS_TYPE} --model ${MODEL_TYPE}"
else 
    RUN_EXPERIMENTS_ARGS="--device cuda --experiment ${EXPERIMENT_NAME}"
fi

# Replace placeholders in the template with actual environment variable values and the experiment name
sed -e "s|\${VENV_PATH}|$VENV_PATH|g" \
    -e "s|\${WORKING_DIR}|$WORKING_DIR|g" \
    -e "s|\${EMAIL}|$EMAIL|g" \
    -e "s|\${QUEUE}|$QUEUE|g" \
    -e "s|\${WALLTIME}|$WALLTIME|g" \
    -e "s|\${GPU_MODE}|$GPU_MODE|g" \
    -e "s|\${NUM_CORES}|$NUM_CORES|g" \
    -e "s|\${MEM_GB}|$MEM_GB|g" \
    -e "s|\${EXPERIMENT_NAME}|$EXPERIMENT_NAME|g" \
    -e "s|\${RUN_EXPERIMENTS_ARGS}|$RUN_EXPERIMENTS_ARGS|g" \
    "$JOB_SCRIPT_TEMPLATE" > "$TEMP_JOB_SCRIPT"

# Submit the job
bsub < "$TEMP_JOB_SCRIPT"

# Optionally, remove the temporary job script after submission
rm "$TEMP_JOB_SCRIPT"
