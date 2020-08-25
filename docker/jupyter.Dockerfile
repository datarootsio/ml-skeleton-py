FROM python:3.7
RUN apt-get update && apt-get install -y python-dev libffi-dev build-essential
WORKDIR /workspace
COPY ./pyproject.toml /workspace
RUN pip3 install poetry
# docker is already a virtual env, no need to create a new one
RUN poetry config virtualenvs.create false 
# install poetry minus development packages
RUN poetry install --no-dev
RUN poetry add jupyter
EXPOSE 8888
ENTRYPOINT ["poetry", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
