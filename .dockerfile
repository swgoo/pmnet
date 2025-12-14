FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
RUN apt-get update && \
    apt-get install -y git &&\
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /temp/requirements.txt
RUN pip install --upgrade-strategy only-if-needed -r /temp/requirements.txt
RUN rm /temp/requirements.txt