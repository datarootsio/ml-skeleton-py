.PHONY: help create_environment requirements train predict
.DEFAULT_GOAL := help


###############################################################
# GLOBALS                                                     #
###############################################################
NO_OF_TEST_FILES := $(words $(wildcard tests/test_*.py))
NO_OF_REPORT_FILES := $(words $(wildcard reports/))
NO_OF_REPORT_FILES := $(words $(filter-out reports/.gitkeep, $(SRC_FILES)))
DATASET := data/transformed/creditcard.csv

###############################################################
# COMMANDS                                                    #
###############################################################
install: ## install dependencies
	pip install -e ".[test, serve]"

clean: ## clean artifacts
	@echo ">>> cleaning files"
	rm ./data/predictions/* ./data/transformed/* ./models/*.joblib || true

generate-dataset: $(DATASET)

$(DATASET):
	@echo ">>> generating dataset"
	python ./scripts/generate_dataset.py $(ARGS)

train: $(DATASET) ## train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
	@echo ">>> training model"
	python ./scripts/train.py $(ARGS)

serve: ## serve trained model with a REST API using dploy-kickstart
	@echo ">>> serving the trained model"
	kickstart serve -e ml_skeleton_py/model/predict.py -l .

run-pipeline: install clean generate-dataset train serve  ## install dependencies -> clean artifacts -> generate dataset -> train -> serve

lint: ## flake8 linting and black code style
	@echo ">>> black files"
	black scripts ml_skeleton_py tests
	@echo ">>> linting files"
	flake8 scripts ml_skeleton_py tests

coverage: ## create coverage report
	@echo ">>> running coverage pytest"
	pytest --cov=./ --cov-report=xml

test: ## run unit tests in the current virtual environment
	@echo ">>> running unit tests with the existing environment"
	pytest

test-docker: ## run unit tests in docker environment
	@echo ">>> running unit tests in an isolated docker environment"
	docker-compose up test

help: ## show help on available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


