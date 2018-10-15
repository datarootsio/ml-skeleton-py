FROM continuumio/miniconda3
RUN conda create -n env python=3.7
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# better to install requirements before because of docker caching
ADD . /code
WORKDIR /code
