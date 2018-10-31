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

After training the model, metadata are being generated, using helper methods  from `mlmonkey` package.  
For details on how metadata are generated, refer to documentation of that package.


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
For detailed description of valid requests/responses, refer to `mlmonkey` package documentation. 


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