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

train:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train.py