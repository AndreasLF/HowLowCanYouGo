#!/bin/bash

# Check if an experiment name was provided
if [ "$#" -ne 2 ] && [ "$#" -ne 4 ] && [ "$#" -ne 7 ]; then
    echo "Usage: $0 DATA EXP_ID [MIN_RANK MAX_RANK [PHASE1_EPOCHS PHASE2_EPOCHS PHASE3_EPOCHS]]"
    exit 1
fi

# The first argument is the dataset
DATA="$1"
EXP_ID="$2"

# Source your environment variables
source ./env_vars.sh

# Path to the job script template
JOB_SCRIPT_TEMPLATE="./jobscript_template_resubmit.sh"

# Temporary job script file that will be populated with environment variables and the experiment name
TEMP_JOB_SCRIPT="./jobscript_populated_resubmit.sh"

# Optionally provide two additional arguments for the loss_type and the model_type
if [ "$#" -eq 3 ]; then
    # Check if they are valid loss and model types, loss_type can be one of "logistic", "hinge", "poisson"
    # and model_type can be one of "L2", "PCA"
    MIN_RANK="$2"
    MAX_RANK="$3"

    # Assign the loss and model types to variables
    RUN_EXPERIMENTS_ARGS="--device cuda --dataset ${DATA} --min ${MIN_RANK} --max ${MAX_RANK} --cexp ${EXP_ID} --wandb"
elif [ "$#" -eq 6 ]; then
    MIN_RANK="$2"
    MAX_RANK="$3"
    PHASE1_EPOCHS="$4"
    PHASE2_EPOCHS="$5"
    PHASE3_EPOCHS="$6"

    RUN_EXPERIMENTS_ARGS="--device cuda --dataset ${DATA} --min ${MIN_RANK} --max ${MAX_RANK} --cexp ${EXP_ID}  --phase1 ${PHASE1_EPOCHS} --phase2 ${PHASE2_EPOCHS} --phase3 ${PHASE3_EPOCHS} --wandb"
else
    RUN_EXPERIMENTS_ARGS="--device cuda --dataset ${DATA} --cexp ${EXP_ID}  --wandb"
fi

# Path for the resubmit times tracker. How many times have we resubmitted the script?
RESUBMIT_TIMES_TRACKER=$WORKING_DIR/jobscripts/resubmit_times_tracker_$EXP_ID.txt

# Initialize the resubmit times tracker if it doesn't exist
if [ ! -f "$RESUBMIT_TIMES_TRACKER" ]; then
    echo "0" > "$RESUBMIT_TIMES_TRACKER"
    if [ $? -eq 0 ]; then
        echo "Successfully created ${RESUBMIT_TIMES_TRACKER}"b
    else
        echo "Failed to create ${RESUBMIT_TIMES_TRACKER}"
    fi
else
    echo "Resubmit times tracker file already exists at ${RESUBMIT_TIMES_TRACKER}"
fi

# Verify the resubmit times tracker file
if [ -f "$RESUBMIT_TIMES_TRACKER" ]; then
    echo "Resubmit times tracker file successfully created or exists: $(cat $RESUBMIT_TIMES_TRACKER)"
else
    echo "Error: Resubmit times tracker file could not be created."
    exit 1
fi


# Extract hours and minutes from WALLTIME
WALLTIME_HH=${WALLTIME%:*}
WALLTIME_MM=${WALLTIME#*:}
# Convert walltime (hh:mm) to seconds and subtract 3 minutes
WALLTIME_SECONDS=$((WALLTIME_HH * 3600 + WALLTIME_MM * 60 - 3 * 60))

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
    -e "s|\${RESUBMIT_TIMES}|$RESUBMIT_TIMES|g" \
    -e "s|\${SUBMIT_SCRIPT_PATH}|$SUBMIT_SCRIPT_PATH|g" \
    -e "s|\${RESUBMIT_TIMES_TRACKER}|$RESUBMIT_TIMES_TRACKER|g" \
    -e "s|\${WALLTIME_SECONDS}|$WALLTIME_SECONDS|g" \
    "$JOB_SCRIPT_TEMPLATE" > "$TEMP_JOB_SCRIPT"

# Submit the job
bsub < "$TEMP_JOB_SCRIPT"

# Optionally, remove the temporary job script after submission
rm "$TEMP_JOB_SCRIPT"
