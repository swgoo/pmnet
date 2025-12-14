FROM huggingface/transformers-pytorch-gpu:latest
WORKDIR /temp
COPY requirements.txt .
RUN pip install --upgrade-strategy only-if-needed -r requirements.txt
RUN apt-get update && \
    apt-get install -y git

WORKDIR /workspace
RUN rm -rf /temp/*
