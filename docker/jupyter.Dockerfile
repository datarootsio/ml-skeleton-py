FROM python:3.7
RUN apt-get update && apt-get install -y python-dev libffi-dev build-essential
WORKDIR /workspace
COPY ./setup.py /workspace
RUN pip install .
RUN pip install jupyter
EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
