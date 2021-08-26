# Python ML project skeleton

This is an opinionated project skeleton for a Python based machine learning projects.  
This skeleton is to be used as the default project start for any Python ML project (unless arguments support otherwise).  
While the project is heavily opinionated, opinions are welcomed to be discussed: feel free to open an issue or comment.

## Installing the package

1. Clone the repo

    ```bash
    git clone git@github.com:datarootsio/ml-skeleton-py.git
    cd ml-skeleton-py
    ```

2. Install dependencies using [pip](https://pip.pypa.io/en/stable/installing/). The following command
will install the dependencies from `setup.py`. In the backend it will run `pip install -e ".[test, serve]"`. Note that installing dependencies with `-e` 
editable mode is needed to properly run unit tests. `[test, serve]` is optional. `test` refers to
unit test dependencies and `serve` refers to deployment dependencies.

    ```bash
    make install
    ```

## Running the project

Preferably, you can use make commands (from `Makefile`) or directly run scripts from `scripts`.  
Refer to section below for the descriptions of make commands. Before running it, consider creating  
a virtual environment.  

**Makefile and test example**

Try out the `make` commands on the example `creditcard.csv` dataset model (see `make help`).

```
clean                          clean artifacts
coverage                       create coverage report
generate-dataset               run ETL pipeline
help                           show help on available commands
lint                           flake8 linting and black code style
run-pipeline                   clean artifacts -> generate dataset -> train -> serve
serve                          serve trained model with a REST API using dploy-kickstart
test-docker                    run unit tests in docker environment
test                           run unit tests in the current virtual environment
train                          train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
```

Note the dependency: `generate-dataset` > `train` > `serve`.

## Docker

Currently, you can find the following docker files:  
1. `jupyter.Dockerfile` builds an image for running notebooks.  
2. `test.Dockerfile` builds an image to run all tests in (`make test-docker`).
3. `serve.Dockerfile` build an image to serve the trained model via a REST api.
To ease the serving it uses open source `dploy-kickstart` module. To find more info
about `dploy-kickstart` click [here](https://github.com/dploy-ai/dploy-kickstart/).

Finally, you can start all services using `docker-compose`:  
for example `docker-compose up jupyter` or `docker-compose up serve`.  

Do you need a notebook for development? Just run `docker-compose up jupyter`. It will launch a Jupyter Notebook 
with access to your local development files.

## Deploying the API

Calling `make serve` will start a Flask based API using `dploy-kickstart`
wrapper. 

In `ml_skeleton_py/model/predict.py` file, there is `# @dploy endpoint predict`
annotation above the `predict` method. 

From `# @dploy endpoint predict` annotation, we are telling `dploy-kickstart` 
that the url that we need to do the post request is `http://localhost:8080/predict`.
As another example, if the annotation would be `# @dploy endpoint score` then the url
would change to `http://localhost:8080/score`.  

Going back to our case, the posted data to `http://localhost:8080/predict` url will be
the argument of the exposed method which is `def predict(body)`. 

As a concrete example;

After calling `make serve`, we can do our predictions with the following curl command.
In this case, `def predict(body)` method will be triggered and the value of the `--data`
will be the argument of `def predict(body)` function, i.e. `body`.

```sh
 curl --request POST \
  --url http://localhost:8080/predict \
  --header 'content-type: application/json' \
  --data '{"model_f_name": "lr.joblib",
           "features": [28692.0,-29.200328590574397,16.1557014298057,-30.013712485724803,6.47673117996833,-21.2258096535165,-4.90299739658728,
                        -19.791248405247,19.168327389730102,-3.6172417860425496,-7.87012194292549,4.06625507293473,-5.66149242261771,1.2929501445424199,
                        -5.07984568135779,-0.126522740416921,-5.24447151974264,-11.274972585125198,-4.67843652929376,0.650807370688892,1.7158618242835801,1.8093709332883998,
                        -2.1758152034214198,-1.3651041075509,0.174286359566544,2.10386807204715,-0.20994399913056697,1.27868097084218,0.37239271433854104,
                        99.99]
           }'
```

To test the health of the deployed model, you can make a get request as shown below;

```sh
    curl --request GET \
      --url http://localhost:8080/healthz
```



## Project Structure Overview 
The project structure tree is shown below. This structure is designed
in a way to easily develop ML projects. Feedback / PRs are always welcome
about the structure.

```
.
├── .github             # Github actions CI pipeline
|
├── data                
│   ├── predictions     # predictions data, calculated using the model
│   ├── raw             # immutable original data
│   ├── staging         # data obtained after preprocessing, i.e. cleaning, merging, filtering etc.
│   └── transformed     # data ready for modeling (dataset containing features and label)
|
├── docker              # Store all dockerfiles
|
├── ml_skeleton_py      # Logic of the model
│   ├── etl             # Logic for cleaning the data and preparing train / test set 
│   └── model           # Logic for ML model including CV, parameter tuning, model evaluation
|
├── models              # Store serialized fitted models
|
├── notebooks           # Store prototype or exploration related .ipynb notebooks
|
├── reports             # Store textual or visualisation content, i.e. pdf, latex, .doc, .txt 
|
├── scripts             # Call ml_skeleton_py module from here e.g. cli for training
|
└── tests               # Unit tests
```

## Best practices for development

- Make sure that `docker-compose up test` runs properly.  
- In need for a Notebook? Use the docker image: `docker-compose up jupyter`.
- Commit often, perfect later.
- Integrate `make test` with your CI pipeline.
- Capture `stdout` when deployed.
