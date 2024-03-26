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

RANK = 8
LR = 1.0
EPOCHS = 10_000

train-ll2:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py \
		--model-type L2 --loss-type logistic \
		--rank $(RANK) \
		--lr $(LR) --num-epochs $(EPOCHS) \
		--save-ckpt results/ll2.pt \
		--device cuda \
		--model-init random

train-lpca:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py \
		--model-type PCA --loss-type logistic \
		--rank $(RANK) \
		--lr $(LR) --num-epochs $(EPOCHS) \
		--save-ckpt results/lpca.pt \
		--device cuda \
		--model-init random

train-hl2:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py \
		--model-type L2 --loss-type hinge \
		--rank $(RANK) \
		--lr $(LR) --num-epochs $(EPOCHS) \
		--save-ckpt results/hl2.pt \
		--device cuda \
		--model-init random

train-hpca:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py \
		--model-type PCA --loss-type hinge \
		--rank $(RANK) \
		--lr $(LR) --num-epochs $(EPOCHS) \
		--save-ckpt results/hpca.pt \
		--device cuda \
		--model-init random

get_stats:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/make_stats.py --print-latex