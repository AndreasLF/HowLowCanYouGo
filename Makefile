.PHONY: create_environment requirements dev_requirements clean data build_documentation serve_documentation

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = graph_embeddings
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_venv:
	PYTHON_INTERPRETER -m venv venv

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

datasets:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data/make_datasets.py

run_experiments:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/run_experiments.py $(ARGS)

DEVICE = cuda
RANK = 64
LR = 0.25
EPOCHS = 10_000
# DATASET = Planetoid/Cora
DATASET = SNAPDataset/Wiki-Vote
# BATCHING_TYPE = casecontrol
# BATCH_SIZE_PERCENTAGE = 0.1 # batch = full adj
BATCHING_TYPE = random
BATCH_SIZE_PERCENTAGE = 1.0 # batch = full adj
RECONS_CHECK = frob
TRAIN_RANDOM = 	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py \
		--rank $(RANK) \
		--lr $(LR) --num-epochs $(EPOCHS) \
		--model-init random \
		--dataset $(DATASET) \
		--batching-type $(BATCHING_TYPE) \
		--batchsize-percentage $(BATCH_SIZE_PERCENTAGE) \
		--recons-check $(RECONS_CHECK) \
		--device $(DEVICE)

train-ll2:
	$(TRAIN_RANDOM) \
		--model-type L2 --loss-type logistic \
		--save-ckpt results/ll2.pt
train-lpca:
	$(TRAIN_RANDOM) \
		--model-type PCA --loss-type logistic \
		--save-ckpt results/lpca.pt
train-leig:
	$(TRAIN_RANDOM) \
		--model-type LatentEigen --loss-type logistic \
		--save-ckpt results/lpca.pt
train-lhyp:
	$(TRAIN_RANDOM) \
		--model-type Hyperbolic --loss-type logistic \
		--save-ckpt results/lhyp.pt

train-hhyp:
	$(TRAIN_RANDOM) \
		--model-type Hyperbolic --loss-type hinge \
		--save-ckpt results/hhyp.pt
train-hl2:
	$(TRAIN_RANDOM) \
		--model-type L2 --loss-type hinge \
		--save-ckpt results/hl2.pt
train-hpca:
	$(TRAIN_RANDOM) \
		--model-type PCA --loss-type hinge \
		--save-ckpt results/hpca.pt

train-simple-l2:
	$(TRAIN_RANDOM) \
		--model-type L2 --loss-type simple \
		--save-ckpt results/sl2.pt
train-pl2:
	$(TRAIN_RANDOM) \
		--model-type L2 --loss-type poisson \
		--save-ckpt results/pl2.pt
train-ppca:
	$(TRAIN_RANDOM) \
		--model-type PCA --loss-type poisson \
		--save-ckpt results/ppca.pt
train-phyp:
	$(TRAIN_RANDOM) \
		--model-type Hyperbolic --loss-type poisson \
		--save-ckpt results/phyp.pt


PLOT_BETA = True

example-1:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/examples.py \
		single \
		--example 1 --l2-rank 1 \
		--N 50 --num-blocks 10 --plot-beta-radius $(PLOT_BETA)
example-2:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/examples.py \
		single \
		--example 2 --l2-rank 2 \
		--N 50 --num-blocks 10 --plot-beta-radius $(PLOT_BETA)
example-ch2020-easy:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/examples.py \
		single \
		--example ch2020 --l2-rank 2 \
		--N 15 --num-blocks 5 --plot-beta-radius $(PLOT_BETA)
example-ch2020-hard:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/examples.py \
		multiple \
		--example ch2020 --l2-rank 2 \
		--N 51 --num-blocks 17 --plot-beta-radius $(PLOT_BETA)



get_stats:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/make_stats.py --print-latex

profile:
	$(PYTHON_INTERPRETER) -m cProfile -o profile.prof $(PROJECT_NAME)/train.py \
	--model-type L2 --loss-type logistic \
	--rank $(RANK) \
	--lr $(LR) --num-epochs $(EPOCHS) \
	--model-init random \
	--device $(DEVICE)


#################################################################################
# Plotting                                                                      #
#################################################################################

COLD_START_RANK = 40

train-cold-start: 
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train_coldstart.py \
		--rank $(COLD_START_RANK) \
		--experiment Pubmed1 \
		--model-type L2 \
		--loss-type logistic\
		--loglevel 3 \
		--device $(DEVICE)

plot-hot-n-cold:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/plotting/plot_hot_n_cold.py


plot-batching:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/plotting/batching_plots.py

plot-frob-errors:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/plotting/plot_frob_errors.py

hbdm-plot:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/plotting/hbdm_plots.py $(ARGS)