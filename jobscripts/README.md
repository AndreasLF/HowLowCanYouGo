# Running experiments on HPC cluster
We are using the HPC cluster at the Techincal University of Denmark to run our experiments. More info can be found [here](https://www.hpc.dtu.dk/).

## Accessing the cluster
To access the cluster you need to have a DTU account and be on the DTU network or VPN. SSH into the cluster using the following command:

```sh
ssh login.gbar.dtu.dk
```


## Jobscripts
For reproducibility purpososes we have provided all the jobscripts used to run the experiments throughout the whole project. 

To be able to run the jobscripts you need to create and env_vars.sh and place it in the [jobscrips folder](https://github.com/AndreasLF/GraphEmbeddings/tree/main/jobscripts). The env_vars.sh file should contain the following:

```sh
export WORKING_DIR=<working_directory>
export VENV_PATH="$WORKING_DIR/venv/bin/activate"
export EMAIL=<email>
```



## Submitting a job
To submit a job, run the following command (replace `jobscript.sh` with the name of the jobscript you want to run):

```sh
bsub < jobscript.sh
```
