#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J GraphEmbeddings_run1
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 8:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=16GB]"

### -- set the email address --
#BSUB <INSERT_EMAIL>

##BSUB -u <INSERT_EMAIL>
### -- send notification at start --
#BSUB -B <INSERT_EMAIL>
### -- send notification at completion--
#BSUB -N <INSERT_EMAIL>
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o GraphEmbeddings_run1%J.out
#BSUB -e GraphEmbeddings_run1%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
# Load the cuda module
module load python3/3.11.7
module load cuda/11.6

source <path to venv>
cd <project folder>
make run_experiments ARGS="--device cuda"