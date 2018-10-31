.PHONY: help create_environment requirements train predict
.DEFAULT_GOAL := help


###############################################################
# GLOBALS                                                     #
###############################################################
# PY_ENV can be "conda" or "pip"
PY_ENV ?= conda
PIP_EXEC ?= pip3
CONDA_EXEC ?= conda
CONDA_PY_VERSION := "3.7"
NO_OF_TEST_FILES := $(words $(wildcard tests/test_*.py))
NO_OF_REPORT_FILES := $(words $(wildcard reports/))
NO_OF_REPORT_FILES := $(words $(filter-out reports/.gitkeep, $(SRC_FILES)))

###############################################################
# COMMANDS                                                    #
###############################################################

create-environment: ## create a python environment
	@echo ">>> removing old environment if exists"
	@rm -rf ./env

    ifeq (conda, $(PY_ENV))
		@echo ">>> setting up conda environment"
		@echo ">>> installing python version $(PYTHON_VERSION)"
		@$(CONDA_EXEC) create -y --prefix env python=$(PYTHON_VERSION)
		@echo ">>> conda environment created, activate with: source activate ./env"
    endif

requirements: ## install requirements and mlmonkey
	@echo ">>> installing requirements and mlmonkey"
	. activate ./env; \
	pip install -r requirements.txt; \
	pip install mlmonkey/.

generate-dataset: ## run new ETL pipeline
	@echo ">>> generating dataset"
	. activate ./env; \
	python -m src.etl.generate_dataset $(ARGS)

lint: ## lint the code using flake8
	@. activate ./env; \
	flake8 src/

train: ## train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
	@echo ">>> training model"
	. activate ./env; \
	python -m src.model.train $(ARGS)

prediction: ## predict new values, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" prediction
	@echo ">>> generating new predictions/estimates"
	. activate ./env; \
	python -m src.model.predict $(ARGS)

deploy-endpoint: ## start flask server, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" deploy-endpoint
	@echo ">>> starting flask"
	. activate ./env; \
	python -m src.helpers.deploy_endpoint $(ARGS)

count-test-files: ## count the number of present test files
    ifeq (0, $(NO_OF_TEST_FILES))
		$(error >>> No tests found)
    else
	@echo ">>> OK, $(NO_OF_TEST_FILES) pytest file found"
    endif

count-report-files: ## count the number of present report files
    ifeq (0, $(NO_OF_REPORT_FILES))
		$(warning >>> No report files found)
    else
	@echo ">>> OK, $(NO_OF_REPORT_FILES) report files found"
    endif

pytest: ## run pytest tests
	. activate ./env; \
	pip install .; \
	pytest tests

init: create-environment requirements ## create environment & install requirements.txt

init-train: init generate-dataset train ## create environment & train the model

test: init generate-dataset train prediction lint pytest count-test-files count-report-files ## run extensive tests

## build: ## package the project
##    @echo ">>> building package"
##    python setup.py sdist bdist_wheel

help: ## show help on available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
