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

api: ## start flask server, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" deploy-endpoint
	@echo ">>> starting flask"
	python ./scripts/api.py $(ARGS)

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
	coverage report -m

tox: ## run tox tests
	tox

test: generate-dataset train prediction tox count-test-files count-report-files ## run extensive tests

spark-zip: ## build the dependency zip file to submit with a spark job
	mkdir -p ./dist/libs
	cp -a ./src/. ./dist/libs/src/
	pip install -r requirements-spark.txt -t ./dist/libs/
	cd ./dist/libs/; zip  -r ../spark-libs.zip *
	rm -rf ./dist/libs


help: ## show help on available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
