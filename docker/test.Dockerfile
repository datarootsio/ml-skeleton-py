FROM python:3.7
RUN apt-get update && apt-get install -y python-dev libffi-dev build-essential
WORKDIR /app
COPY . /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD} 
RUN pip3 install poetry
# docker is already a virtual env, no need to create a new one
RUN poetry config virtualenvs.create false 
# install poetry minus development packages
RUN poetry install --no-dev
ENV LOG_LEVEL INFO
ENTRYPOINT ["make", "test-package"]
