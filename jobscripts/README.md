# Running experiments on HPC cluster
We are using the HPC cluster at the Techincal University of Denmark to run our experiments. More info can be found [here](https://www.hpc.dtu.dk/).

## Accessing the cluster
To access the cluster you need to have a DTU account and be on the DTU network or VPN. SSH into the cluster using the following command:

```sh
ssh login.gbar.dtu.dk
```


## Jobscripts
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
./submit_job1.sh DATA [MIN_RANK MAX_RANK]
```

Where `DATA` is the name of the dataset, e.g. Cora. `[MIN_RANK MAX_RANK]` are optional arguments that can be used to specify the minimum rank and maximum rank

This will populate the `jobscript_template1.sh` with the parameters defined in `env_vars.sh` along with the provided `DATA` and submit the job to the cluster.
