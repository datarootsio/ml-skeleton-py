FROM python:3.7
RUN apt-get update && apt-get install -y python-dev libffi-dev build-essential
WORKDIR /app
COPY . /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip install ".[serve]"

# Serve your ml skeleton locally with a REST API using open source dploy-kickstart
# visit https://github.com/dploy-ai/dploy-kickstart for more info
ENTRYPOINT ["make", "run-pipeline"]
