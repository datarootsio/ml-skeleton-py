.PHONY: help default clean lint requirements replacement_test help
.DEFAULT_GOAL := help


###############################################################
# GLOBALS                                                     #
###############################################################

###############################################################
# COMMANDS                                                    #
###############################################################

init: create_environment requirements ## create environment & install requirements.txt

default_project: ## initialize a cookiecutter project using the default config
	@echo ">>> generating template using default config"
	cookiecutter . --default-config --overwrite-if-exists --no-input
	ls

clean: ## clean default cookiecutter template (beautifulml)
	@rm -rf ./beautifulml/
	-conda env remove --name beautifulml -y

lint: ## lint all python source code
	@flake8

requirements: ## install requirements needed for testing
	@pip install -r requirements.txt

replacement_test: ## check for correct variable interpolation
	python tests/test_templating.py

run_examples: ## run all examples in the project
	make --directory beautifulml/ init
	make --directory beautifulml/ generate_dataset
	make --directory beautifulml/ train
	make --directory beautifulml/ prediction

test: default_project replacement_test run_examples clean lint ## run all tests

help: ## show help on available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

