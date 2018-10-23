# Python ML project skeleton

**THIS HAS NOT BEEN VALIDATED YET**

This is an opinionated project skeleton for a Python based machine learning projects.  
This skeleton is to be used as the default project start for any Python ML project (unless arguments support otherwise).  
While the project is heavily opinionated, opinions are welcomed to be discussed: feel free to open an issue or comment.

This project is built upon the best practices discussed in our [methodology](https://gitlab.com/dataroots/public/methodology) repo.

## Scope

There are several things we cover using this skeleton, including:
1. predefined project structure
2. generation of metadata used to ease model reproducibility and report generation
3. check if the project includes necessary reports 
4. check if the project is appropriately tested (unittest coverage)
5. assist creation of deployable solution, e.g. using Flask, Docker etc.  
... (this list will be extended)

## Overview project structure

We want to ensure each project follows (roughly) the same structure, according to best practices:

```
├── data
│   ├── predictions
│   ├── raw
│   ├── staging
│   └── transformed
├── models
│   └── metadata
├── notebooks
├── reports
├── src
│   ├── etl
│   ├── helpers
│   └── model
└── tests

```

### data/

Location for data in various shapes.
Note that the `raw` data should always be considered **immutable**.  
For larger projects, where it's infeasible to have project specific datasets within the project structure,  
make sure to update the configuration and connectors to reflect as much.

### models/

Location for saving serialized models. Make sure that the serialized version is of a reasonable size  
(e.g., do not include training data in the model object). For every modelm metadata will be stored in `models/metadata`.  
Metadata is very important as it should support model reproducibility and report generation.  
You can find more details on metadata description in following section.

### notebooks/

Location to save notebooks used for data and model exploration.

### reports/

Location for reports and files necessary to reproduce report generation.  
Note that non-Python tools (e.g. RMarkdown) can be used as well.  
At least explorative and delivery report are expected.  
Note that `make test` will not error if reports are not included, but will instead issue a warning.

### src/

This directory should contain the logic for the model (training & prediction), ETL, helpers and potential apps.  
All source code relevant to a packaged and deployable delivery should be contained in this folder.

Two scripts must exist in this directory:
1. `train.py` - trains the model, saves it and generates metadata
2. `predict.py` - calculates predictions for new data

### tests/

This directory contains the unittests by which you test your helper functions and coded logic.

## Model metadata

Here is list of metadata properties you should have:
1. `model_location`: Path to saved serialized model.
2. `model_type`: Name of the Python class for the model instance. Eg: sklearn.linear_model.base.LinearRegression
3. `model_description`: Textual description of the model (free-form).
4. `model_identifier`: Unique ID of the model. Calculated as hash of concatenated git commit number and timestamp.
5. `sklearn_object`: If the model is trained using sklearn, provide model object. Othervise, provide None.  
In case we provided sklearn model, hyperparameters are extracted automatically using sklearn method and stored in `model_hyperparameters` (see below).  
In case we don't use sklearn model, provide values for `model_hyperparameters` in some other way, if possible.
6. `input_data_location`: Path to data used for model training.
7. `input_data_identifier`: In case we want to reproduce model, it is important to have version of the data used for training, if possible.  
Data identifier should be any value that can help us to get right version of the data.  
For example, that can include id of the last row in dataset, if that dataset is always updated by appending new rows.
8. `feature_names`: Names of features used for model training. Don't need to be the same as names of columns in original input data.
9. `testing_strategy`: Description of the strategy used for testing/evaluating model, i.e. calculating `scores`.  
It should be detailed so the scores can be reproduced. For example, include information if you used cross validation and/or holdout set and what setup  
(number of folds, stratification etc).
10. `scores`: Metrics for model assessment. They should be formatted as map containing one or more metrics:  
`metric name: scores map`, where scores map can contain multiple key-value pairs of type: `strategy name: score value`.  
There are constraints on what can be valid metric name, and what can be strategy name.  
Valid metric names: Those listed in `sklearn.metrics.SCORERS.keys()` +   
`['log_loss', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error']`  
In addition, if you want to use some alternative metric, you can include it but you must set its name to start with 'custom'.  
Valid names for testing strategy: `['cross_val', 'hold_out']`, or other strategy which name must start with 'custom'.
11. `git_commit`: Number of git commit associated with the code version used when model was trained.
12. `timestamp`: Time when the model was saved.
13. `model_hyperparameters`: Map (key-value pairs) containing hyperparameters used when model was fitted. 
 
We don't keep information about feature engineering in metadata, but reather in the reports.

## Setup the environment

Setup these environment variables:  
PY_ENV (conda or pip)  
PIP_EXEC (e.g. pip3)  
CONDA_EXEC (e.g. conda)  

Note:  
These variables are set in Makefile (where we use only conda currently), but this will be changed so we can support both conda and pip.

TODO: 
- add docker-compose instructions (and how to setup pycharm to use this container)

## Makefile and test example

Try out the `make` commands on the example iris dataset model (see `make help`).

```sh
help                           show help on available commands
create-environment             create new virtual environment
requirements                   install requirements specified in "requirements.txt"
generate_dataset               run new ETL pipeline, to generate dataset from raw data
lint                           lint the code using flake8
train                          train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
prediction                     predict new values, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
count_report_files             count the number of present report files
count_test_files               count the number of present test files
pytest                         run tox/pytest tests
init                           create environment & install requirements.txt
test                           run extensive test
```

Note the dependency: `generate_dataset` > `train` > `prediction`.


## Best practices for development

- Make sure that `make test` runs properly.  
This includes running `make lint`, to check your code format.
- Commit often, perfect later.
- Integrate `make test` with your CI pipeline.
- Capture `stdout` when deployed.

## Project configuration

TODO:
Add configuration variables to the `.env` file. Note, check to see if it's OK
to include this in the git repository. No credentials should be committed.