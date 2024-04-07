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

DEVICE = cpu
RANK = 8
LR = 1.0
EPOCHS = 10_000
DATASET = Planetoid/PubMed
BATCH_SIZE_PERCENT = 1.0 # 1.0 means full batch
TRAIN_RANDOM = 	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py \
		--rank $(RANK) \
		--lr $(LR) --num-epochs $(EPOCHS) \
		--model-init random \
		--dataset $(DATASET) \
		--batch-size-percent $(BATCH_SIZE_PERCENT) \
		--device $(DEVICE)

train-ll2:
	$(TRAIN_RANDOM) \
		--model-type L2 --loss-type logistic \
		--save-ckpt results/ll2.pt
train-lpca:
	$(TRAIN_RANDOM) \
		--model-type PCA --loss-type logistic \
		--save-ckpt results/lpca.pt

train-hl2:
	$(TRAIN_RANDOM) \
		--model-type L2 --loss-type hinge \
		--save-ckpt results/hl2.pt
train-hpca:
	$(TRAIN_RANDOM) \
		--model-type PCA --loss-type hinge \
		--save-ckpt results/hpca.pt

train-pl2:
	$(TRAIN_RANDOM) \
		--model-type L2 --loss-type poisson \
		--save-ckpt results/pl2.pt
train-ppca:
	$(TRAIN_RANDOM) \
		--model-type PCA --loss-type poisson \
		--save-ckpt results/ppca.pt

get_stats:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/make_stats.py --print-latex