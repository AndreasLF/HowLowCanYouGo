# Running experiments on HPC cluster
We are using the HPC cluster at the Techincal University of Denmark to run our experiments. More info can be found [here](https://www.hpc.dtu.dk/).

## Accessing the cluster
To access the cluster you need to have a DTU account and be on the DTU network or VPN. SSH into the cluster using the following command:

```sh
ssh login.gbar.dtu.dk
```


## Jobscripts
For reproducibility purpososes we have provided a jobscript template along with a submit_job script. The experiment config files can all be found in the [configs folder](https://github.com/AndreasLF/GraphEmbeddings/tree/main/configs).

To be able to run the jobscripts you need to create an `env_vars.sh` and place it in the [jobscrips folder](https://github.com/AndreasLF/GraphEmbeddings/tree/main/jobscripts). Below is an example of the contents of an `env_vars.sh` file:

```sh
export WORKING_DIR=<working_directory>
export VENV_PATH="$WORKING_DIR/venv/bin/activate"
export EMAIL=<email>
export QUEUE="gpua100"
export NUM_CORES=4
export GPU_MODE="num=1:mode=exclusive_process"
export WALLTIME="12:00"
export MEM_GB=16
```

## Submitting a job

### Jobscript 1 (Find lowest rank representation of graph)
To submit a job, run the following command in the terminal:

```sh
./submit_job1.sh EXPERIMENT_NAME
```

Where `EXPERIMENT_NAME` is the name of the experiment you want to run as specified in the config file.

This will populate the `jobscript_template1.sh` with the parameters defined in `env_vars.sh` along with the provided `EXPERIMENT_NAME` and submit the job to the cluster.
