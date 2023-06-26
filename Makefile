.PHONY: virtualenv install build clean data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL   		= /bin/bash
PYTHON 			= python
PROJECT_DIR 	= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME 	= $(shell basename $(CURDIR))

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create virtualenv.
## Activate with the command:
## source venv/bin/activate
virtualenv:
	virtualenv -p $(PYTHON) venv

## Install Python Dependencies.
## Make sure you activate the virtualenv first!
install:
	$(PYTHON) -m pip install -r requirements.txt


## Create settings file where absolute paths to save data and results will be specified
## Create files and directories that are ignored by git but required for the project
## Create symlinks to external data and results storage into this project
build:
	touch src/user_settings.py
ifeq ($(SAVE_DIR),)
	@mkdir -vp $(PROJECT_DIR)/data/{raw,samples,processed,datasets}
	@mkdir -vp $(PROJECT_DIR)/results/{checkpoints,logs/{tensorboard,base_logs},models}
else
	@mkdir -vp /$(SAVE_DIR)/$(PROJECT_NAME)/data/{raw,samples,processed,datasets}
	@mkdir -vp /$(SAVE_DIR)/$(PROJECT_NAME)/results/{checkpoints,logs/{tensorboard,base_logs},models}
	@ln -vs /$(SAVE_DIR)/$(PROJECT_NAME)/data $(PROJECT_DIR)/data
	@ln -vs /$(SAVE_DIR)/$(PROJECT_NAME)/results $(PROJECT_DIR)/results
endif

data:
	wget http://files.grouplens.org/datasets/movielens/ml-25m.zip -P data/raw/
	unzip data/raw/ml-25m.zip -d data/raw/ && rm data/raw/ml-25m.zip


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################
.PHONY: help
.DEFAULT_GOAL := help

help:
	@echo "Usage:"
	@echo "    virtualenv [PYTHON='']"
	@echo "    install [PYTHON='']"
	@echo "    build [SAVE_DIR=''] [PROJECT_NAME='']"
	@echo "    clean"

