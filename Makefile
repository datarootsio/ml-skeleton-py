.PHONY: help create_environment requirements train predict
.DEFAULT_GOAL := help


###############################################################
# GLOBALS                                                     #
###############################################################
NO_OF_TEST_FILES := $(words $(wildcard tests/test_*.py))
NO_OF_REPORT_FILES := $(words $(wildcard reports/))
NO_OF_REPORT_FILES := $(words $(filter-out reports/.gitkeep, $(SRC_FILES)))

###############################################################
# COMMANDS                                                    #
###############################################################

generate-dataset: ## run new ETL pipeline
	@echo ">>> generating dataset"
	python ./scripts/generate_dataset.py $(ARGS)

train: ## train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
	@echo ">>> training model"
	python ./scripts/train.py $(ARGS)

prediction: ## predict new values, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" prediction
	@echo ">>> generating new predictions/estimates"
	python ./scripts/predict.py $(ARGS)

init-train: generate-dataset train ## generate dataset & train the model

clean:
	@echo ">>> cleaning files"
	rm ./data/predictions/* ./data/transformed/* ./models/*.p

linting:
	@echo ">>> black files"
	black scripts ml_skeleton_py tests
	@echo ">>> linting files"
	flake8 scripts ml_skeleton_py tests

test-package:
	@echo ">>> running coverage pytest"
	coverage run -m pytest ./tests/
	coverage report -m --include=./tests/*

test: generate-dataset train prediction test-package ## run extensive tests

help: ## show help on available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
