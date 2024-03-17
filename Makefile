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

pretrain_svd_l2:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py --model-type L2 --rank 32 --train-mode pretrain --optim-type adam --lr 1e-1 --num-epochs 3_000 --save-ckpt results/svd-init-l2.pt --device cuda
pretrain_svd_lpca:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py --model-type LPCA --rank 32 --train-mode pretrain --optim-type adam --lr 1e-2 --num-epochs 3_000 --save-ckpt results/svd-init-lpca.pt --device cuda

init_from_svd_l2:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py --model-type L2 --rank 32 --train-mode reconstruct-from-svd --optim-type adam --lr 1e-1 --num-epochs 3_000 --load-ckpt results/svd-init-l2.pt --save-ckpt results/pretrained-test-l2.pt --device cuda
init_from_svd_lpca:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py --model-type LPCA --rank 32 --train-mode reconstruct-from-svd --optim-type adam --lr 1e-2 --num-epochs 3_000 --load-ckpt results/svd-init-lpca.pt --save-ckpt results/pretrained-test-lpca.pt --device cuda


MDS_RANK = 8
MDS_LR = 1e-1

init_from_mds_l2:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py --model-type L2 --rank $(MDS_RANK) \
		--optim-type adam --lr $(MDS_LR) --num-epochs 3_000 \
		--save-ckpt results/mds-test-l2.pt --device cuda \
		--model-init mds
init_from_rand_l2:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py --model-type L2 --rank $(MDS_RANK) \
		--optim-type adam --lr $(MDS_LR) --num-epochs 3_000 \
		--save-ckpt results/mds-test-l2.pt --device cuda \
		--model-init random