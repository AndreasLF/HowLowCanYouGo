#!/bin/sh
### General options
### –- specify queue --
#BSUB -q ${QUEUE}
### -- set the job Name --
#BSUB -J GraphEmbeddings_Run1_${EXPERIMENT_NAME}
### -- ask for number of cores (default: 1) --
#BSUB -n ${NUM_CORES}
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu ${GPU_MODE}
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W ${WALLTIME}
# request X GB of system-memory
#BSUB -R "rusage[mem=${MEM_GB}GB]"

### -- set the email address --
#BSUB $EMAIL

##BSUB -u ${EMAIL}
### -- send notification at start --
#BSUB -B ${EMAIL}
### -- send notification at completion--
#BSUB -N ${EMAIL}
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o GraphEmbeddings_Run1_${EXPERIMENT_NAME}%J.out
#BSUB -e GraphEmbeddings_Run1_${EXPERIMENT_NAME}%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load python3/3.11.7
module load cuda/11.6

# Activate the virtual environment
source ${VENV_PATH}
# Change to the working directory
cd ${WORKING_DIR}
make run_experiments ARGS="--device cuda --experiment ${EXPERIMENT_NAME}" # This is the line that runs the experiment, experiment name is passed as an argument