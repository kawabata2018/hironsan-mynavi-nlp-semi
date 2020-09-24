FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /tmp

COPY requirements.txt /tmp
RUN pip --no-cache-dir install -r requirements.txt

WORKDIR /work
