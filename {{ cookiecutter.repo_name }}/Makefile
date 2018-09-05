.PHONY: help create_environment requirements train predict
.DEFAULT_GOAL := help


###############################################################
# GLOBALS                                                     #
###############################################################

CONDA_PATH=[OPTIONAL]
#"{{ cookiecutter.conda_path }}"
PYTHON_VERSION=3.6
#"{{ cookiecutter.python_version }}"
REPO_NAME="test_repo"
#"{{ cookiecutter.repo_name g}}"

###############################################################
# COMMANDS                                                    #
###############################################################

init: create_environment requirements ## create environment & install requirements.txt

create_environment: ## create a conda environment
	@echo ">>> creating conda environment"
    ifeq (OPTIONAL,$(findstring OPTIONAL,$(CONDA_PATH)))
		$(eval CONDA_PATH=`which conda`)
    endif
	@echo ">>> using conda at $(CONDA_PATH)"
	@echo ">>> installing python version $(PYTHON_VERSION)"
	@$(CONDA_PATH) create -y -n $(REPO_NAME) python=$(PYTHON_VERSION)
	@echo ">>> conda environment created, activate with: source activate $(REPO_NAME)"

requirements: ## install requirements specified in "requirements.txt"
	@echo ">>> installing requirements.txt"
	source activate $(REPO_NAME); \
	pip install -r requirements.txt

train: ## train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
	@echo ">>> generating new predictions/estimates"
	source activate $(REPO_NAME); \
	python -m src.model.train $(ARGS)

prediction: ## predict new values, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
	@echo ">>> training model"
	source activate $(REPO_NAME); \
	python -m src.model.predict $(ARGS)

generate_dataset: ## run new ETL pipeline
	@echo ">>> generating dataset"
	@source activate $(REPO_NAME); \
	python -m src.etl.generate_dataset $(ARGS)

lint: ## lint the code using flake8
	@source activate $(REPO_NAME); \
	flake8 src/

help: ## show help on available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
