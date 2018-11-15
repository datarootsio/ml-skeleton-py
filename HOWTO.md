# Python ML project skeleton

**THIS HAS NOT BEEN VALIDATED YET**

This is an opinionated project skeleton for a Python based machine learning projects.  
This skeleton is to be used as the default project start for any Python ML project (unless arguments support otherwise).  
While the project is heavily opinionated, opinions are welcomed to be discussed: feel free to open an issue or comment.

This project is built upon the best practices discussed in our [methodology](https://gitlab.com/dataroots/public/methodology) repo.


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
├── data
│   ├── predictions
│   ├── raw
│   ├── staging
│   └── transformed
├── models
│   └── metadata
├── notebooks
├── reports
├── scripts
├── src
│   ├── app
│   ├── etl
│   ├── helpers
│   └── model
└── tests

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

### reports/

Location for reports and files necessary to reproduce report generation. Any format of reports can be used.  
We provide report templates as notebooks. Notebooks are chosen because users are familiar with them and  
interactivity is supported (HTML). At least explorative and delivery report are expected.  
Note that `make test` will not error if reports are not included, but will instead issue a warning.

### scripts/

Python scripts that expose functionality from `src`. The idea is that source can be freely changed, and those scripts  
should preferably stay the same, or not changed much. They can be run manually, from `Makefile`,  
but also represent example scripts that can be passed to Spark job, (e.g. with spark-submit, in case we use Spark).

### src/

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
a virtual environment. Project example code should be compatible with both Python 2.7 and 3.7.  
Next, we need to install `mlmonkey` package. If it is included as a submodule in this project,  
it can be done by `pip install mlmonkey/.`


## Makefile and test example

Try out the `make` commands on the example iris dataset model (see `make help`).
You need to install packages listed in requirements.txt file before running any commands that execute code.

```sh
help                           show help on available commands
generate_dataset               run new ETL pipeline, to generate dataset from raw data
train                          train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
prediction                     predict new values, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" prediction
deploy-endpoint                start Flask endpoint for calculating predictions
count-report-files             count the number of present report files
count-test-files               count the number of present test files
init-train                     generate dataset & train the model
tox                            run tox, that includes running unit tests (pytest) and linting (flake8)
test                           run extensive test
```

Note the dependency: `generate_dataset` > `train` > `prediction`.


## Example scripts

In source directory, we provide basic examples of scripts for generating features dataset,  
training, calculating predictions, and model explanations. After running train script, model and metadata  
will be saved in `models` directory.  Note that for model, supported file extensions are `joblib` and `pickle`,  
as a requirement from `mlmonkey` package, which later loads the model for purpose of exposing the model via Flask API.   
Predictions are saved in `predictions` directory, prediction details are included in log.  


## Model metadata

After training the model, metadata are being generated, using helper methods  from `mlmonkey` package.  
For details on how metadata are generated, refer to documentation of that package.


## Creating API endpoint

Calling `make deploy-endpoint` will start Flask API which calculates predictions for new data.  
`deploy-endpoint` can accept four parameters (model path, metadata path, host and port).   
Default configuration is to use host 0.0.0.0 and port 5000.
For detailed description of valid requests/responses, refer to `mlmonkey` package documentation.  
Example how to call API using curl:
```
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"features":[[4.8, 1.8, 2.4], [7.4, 2.5, 3.1]]}' \
  http://localhost:5000/predict
```

## Report example

Within `report` directory, you can find example of modeling report, in notebook format.  
The report contains sections that should preferably exist in the report, as well as examples  
of textual content and visualizations. 


## Dockerization

Currently you can find two docker files within the project root.  
1. `Dockerfile.jupyter` builds an image for running notebooks.  
2. `Dockerfile.api` builds an image for starting API endpoint. When building image,  
initial model will be trained and included in image definition. You can build image using following command:  
`docker build -t your_tag -f Dockerfile.api .`, and run as `docker run -d -p 5000:5000 your_tag`.  
After this, requests are accepted on localhost, port 5000.  

Finally, you can start both services using `docker-compose`:  
for example `docker-compose up jupyter` and `docker-compose up api`.  


## Best practices for development

- Make sure that `make test` runs properly.  
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

## Prerequisites

If you are about to use graphviz in your project (example is given in template modeling report),  
you should install graphviz software in your system (not just the python package).  
On Linux you can use: `sudo apt-get install graphviz`, for Mac `brew install graphviz`.  
Graphs can be specified in code using Python API, but also specified in separate (.gv) file.