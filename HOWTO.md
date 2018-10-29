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

Here is list of arguments we provide to generate metadata:
1. `model_location`: Path to saved serialized model.
2. `model_description`: Textual description of the model (free-form).
3. `model_object`: Model object. Used only for automatic extraction of additional metadata (e.g model type, size etc).
4. `data_location`: Path to data used for model training.
5. `data_identifier`: In case we want to reproduce model, it is important to have version of the data used for training, if possible.  
Data identifier should be any value that can help us to get right version of the data.  
For example, that can include id of the last row in dataset, if that dataset is always updated by appending new rows.
6. `features_object`: Object (dataframe) representing train data (only features, no labels).  
Used only for automatic extraction of additional metadata (e.g. feature names).
7. `testing_strategy`: Description of the strategy used for testing/evaluating model, i.e. calculating `scores`.  
It should be detailed so the scores can be reproduced. For example, include information if you used cross validation and/or holdout set  
and what setup (number of folds, stratification etc).
8. `scores`: Model evaluation results. These scores should be formatted as map containing one or more metrics:  
`metric name: scores map`, where scores map can contain multiple key-value pairs of type: `strategy name: score value`.  
There are constraints on what can be valid metric name, and what can be strategy name.  
Valid metric names: Those listed in `sklearn.metrics.SCORERS.keys()` +   
`['log_loss', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error']`  
In addition, if you want to use some alternative metric, you can include it but you must set its name to start with 'custom'.  
Valid names for testing strategy: `['cross_val', 'hold_out']`, or other strategy which name must start with 'custom'.
9. `model_hyperparameters`: Map (key-value pairs) containing hyperparameters used when model was fitted. If not provided,  
and if the `model object` is of sklearn type, these hyper-parameters are extracted automatically using sklearn method.
10. `extra_metadata`: Additional metadata user can provide. These metadata can be for example:
- data type (e.g. csv), which can not be easily extracted automatically since we can have many formats, including distributed datasets, 
- training time on given samples - which might depend whether it's about training on whole dataset, cv,  
which hardware was used etc. Some of these info might be automatically extracted. For now, free-text description might be enough.


Based on these 10 provided arguments, some additional metadata is automatically extracted:
1. `model_identifier`: Unique ID of the model. Calculated as hash of concatenated git commit number and timestamp.
2. `model_type`: Name of the Python class for the model instance. Eg: sklearn.linear_model.base.LinearRegression.
3. `model_size`: Size of model object (when loaded in RAM) in bytes.
4. `num_data_rows`: Number of rows (observations) in input data.
5. `num_data_features`: Number of features in input data.
6. `feature_names`: Names of features used for model training.  
Don't need to be the same as names of columns in original (raw) input data.
7. `data_size`: size of `features_object`, in bytes.
8. `git_commit`: Number of git commit associated with the code version used when model was trained.
9. `timestamp`: Time when metadata was generated.
 
We don't keep information about feature engineering in metadata, but rather in the reports.


## Setup the environment

Setup these environment variables:  
PY_ENV (conda or pip)  
PIP_EXEC (e.g. pip3)  
CONDA_EXEC (e.g. conda)  

Note:  
These variables are set in Makefile (where we use only conda currently), but this will be changed so we can support both conda and pip.


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
deploy-endpoint                start Flask endpoint for calculating predictions
count_report_files             count the number of present report files
count_test_files               count the number of present test files
pytest                         run tox/pytest tests
init                           create environment & install requirements.txt
init-train:                    create environment & train the model
test                           run extensive test
```

Note the dependency: `generate_dataset` > `train` > `prediction`.


## Creating API endpoint

Calling `make deploy-endpoint` will start Flask endpoint, which will calculate predictions for new data,  
using up-to-date model. `deploy-endpoint` can accept three parameters (model path, host and port).   
Default configuration is to use host 0.0.0.0 and port 5000.

Detailed description of valid requests/responses is given in `swagger_specification.json`,  
within the root of the project.


## Dockerization

Currently you can find two docker files within the project root.  
1. `Dockerfile.jupyter` builds an image for running notebooks.  
2. `Dockerfile.api` builds an image for starting API endpoint. When building image,  
initial model will be trained and included in image definition. You can build image using following command:  
`docker build -t your_tag -f Dockerfile.api .`, and run as `docker run -d -p 5000:5000 your_tag`.  
After this, requests are accepted on localhost, port 5000.  

Finally, you can start both services using `docker-compose`:  
for example `docker-compose up jupyter` and `docker-compose up api`.  
TODO: 
- add docker-compose instructions (and how to setup pycharm to use this container)


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