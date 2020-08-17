# Python ML project skeleton

**THIS HAS NOT BEEN VALIDATED YET**

This is an opinionated project skeleton for a Python based machine learning projects.  
This skeleton is to be used as the default project start for any Python ML project (unless arguments support otherwise).  
While the project is heavily opinionated, opinions are welcomed to be discussed: feel free to open an issue or comment.

This project is built upon the best practices discussed in our [methodology](https://gitlab.com/dataroots/public/methodology) repo.

## How to test the package

1. Clone the repo

```bash
git clone git@github.com:datarootsio/ml-skeleton-py.git
cd ml-skeleton-py
git checkout api-ci-pipeline
```

2. Install local environment and install requirements.txt

```bash
virtualenv local
source local/bin/activate
pip install -r requirements.txt
```

3. Run basic comands

```bash
make init-train
make ARGS="--model_name lr.p --input_df X_test.p --output_df predictions.csv" prediction
```

4. Run some tests

```bash
coverage run -m pytest ./tests
coverage report -m
coverage html
```

## Scope

There are several things we cover using this skeleton, including:
1. predefined project structure (directories and scripts)
2. example scripts for model training, predictions, model explanation, and modeling report
3. generation of metadata used to ease model reproducibility and report generation
4. assist creation of deployable solution, e.g. using Flask, Docker etc.
5. check if the project includes necessary report files and tests (unit test coverage)
... (this list will be extended)

## Overview project structure

We want to ensure each project follows (roughly) the same structure, according to best practices:

```
.
├── data
│   ├── predictions
│   ├── raw
│   ├── staging
│   └── transformed
├── docker
├── models
│   └── metadata
├── notebooks
├── openapi
├── reports
├── scripts
├── src
│   ├── app
│   ├── etl
│   ├── helpers
│   └── model
├── tests
```

### data/

Location for data in various shapes. Directories for storing data:  
`raw` - contains original dataset, which should always be considered **immutable**.    
`staging` - data obtained after preprocessing - cleaning, merging, filtering etc.  
`transformed` - data ready for modeling (dataset containing features and label).  
`predictions` - for storing predictions calculated using the model.   

For larger projects, where it's infeasible to have project specific datasets within the project structure,  
make sure to update the configuration and connectors to reflect as much.

### models/

Location for saving serialized models. Make sure that the serialized version is of a reasonable size  
(e.g., do not include training data in the model object). For every modelm metadata will be stored in `models/metadata`.  
Metadata is very important as it should support model reproducibility and report generation.  
You can find more details on metadata description in following section.

### notebooks/

Location to save notebooks used for data and model exploration.  
Reporting notebooks are here placed by default, with default content.  
They can be exported to `reports` as HTML/PDF etc.

### openapi/

OpenAPI specs used to deploy the prediction endpoint. From the root of the project you can run `python scripts/api/py`
to deploy the prediction endpoint. Likewise you can run it using `docker-compose up api`. 
Note that `connexxion` will open a SwaggerUI at `/ui` where you can inspect the endpoint specs. 

### reports/

Location for reports and files necessary to reproduce report generation. Any format of reports can be used.  
We provide report templates as notebooks. Notebooks are chosen because users are familiar with them and  
interactivity is supported (HTML). At least explorative and delivery report are expected.  
Note that `make test` will not error if reports are not included, but will instead issue a warning.

### scripts/

Python scripts that expose functionality from `src`. The idea is that source can be freely changed, and those scripts  
should preferably stay the same, or not changed much. They can be run manually, from `Makefile`,  
but also represent example scripts that can be passed to Spark job, (e.g. with spark-submit, in case we use Spark).

### ml_skeleton_py/

This directory should contain the logic for the model (training & prediction), ETL, helpers and potential apps.  
All source code relevant to a packaged and deployable delivery should be contained in this folder.

Two scripts must exist in this directory:
1. `model/train.py` - trains the model, saves it and generates metadata
2. `model/predict.py` - calculates predictions for new data

### tests/

This directory contains the unittests by which you test your helper functions and coded logic.


## Running the project

Preferably, you can use make commands (from `Makefile`) or directly run scripts from `scripts`.  
Refer to section below for the descriptions of make commands. Before running it, consider creating  
a virtual environment.  

First install `mlmonkey` and then dependencies listed in `requirements.txt`.    
You can `pip install` `mlmonkey` directly from here: https://gitlab.com/dataroots-public/mlmonkey.git.


## Makefile and test example

Try out the `make` commands on the example iris dataset model (see `make help`).
You need to install packages listed in requirements.txt file before running any commands that execute code.

```sh
api                            start flask server, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" deploy-endpoint
count-report-files             count the number of present report files
count-test-files               count the number of present test files
generate-dataset               run new ETL pipeline
help                           show help on available commands
init-train                     generate dataset & train the model
prediction                     predict new values, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" prediction
spark-zip                      build the dependency zip file to submit with a spark job
test                           run extensive tests
tox                            run tox tests
train                          train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
```

Note the dependency: `generate_dataset` > `train` > `prediction`.


## Example scripts

In the `src` directory, we provide basic examples of scripts for generating features dataset,  
training, calculating predictions, and model explanations. After running the `train.py` script, model and metadata  
will be saved in the `models` directory.  

## Model metadata

After training the model, metadata are being generated, using helper methods  from `mlmonkey` package.  
For details on how metadata are generated, refer to documentation of that package.

## Deploying the API

Calling `make api` will start a Flask based API (implemented in `connexion`) which calculates predictions for new data.  
The API is defined in `scripts/api.py` and simply implements the specification in `openapi/prediction-api.yaml`.

## Report example

Within `report` directory, you can find example of modeling report, in notebook format.  
The report contains sections that should preferably exist in the report, as well as examples  
of textual content and visualizations. 


## Docker

Currently you can find the following docker files:  
1. `Dockerfile.jupyter` builds an image for running notebooks.  
2. `Dockerfile.api` builds an image for starting an API endpoint.
3. `Dockerfile.test` builds an image to run all tests in (`make test`).

Finally, you can start all services using `docker-compose`:  
for example `docker-compose up jupyter`, `docker-compose up api` or `docker-compose up test`.  

Do you need a notebook for development? Just run `docker-compose up jupyter`. It will launch a Jupyter Notebook 
with access to you local development files.

## Best practices for development

- Make sure that `make test` and/or `docker-compose up test` runs properly.  
- In need for a Notebook? Use the docker image: `docker-compose up jupyter`.
- Commit often, perfect later.
- Integrate `make test` with your CI pipeline.
- Capture `stdout` when deployed.

## Project configuration

Environment variables for the project can be specified in `.env` file,
in project root. These variables will be read by dotenv package.  
For example, you can set variables defined in `src/settings.py`, such as
`MODEL_DIR = /your/path/to/the/model/`.  

Logging can be adjusted in source init script (output location, verbosity level etc).      
Verbosity is read from environment variable `LOG_LEVEL`, and use `WARNING` if such variable is not defined.  

## Spark

See the Makefile for some logic to build a Spark dep file (example TBD.

## Prerequisites

If you are about to use graphviz in your project (example is given in template modeling report),  
you should install graphviz software in your system (not just the python package).  
On Linux you can use: `sudo apt-get install graphviz`, for Mac `brew install graphviz`.  
Graphs can be specified in code using Python API, but also specified in separate (.gv) file.