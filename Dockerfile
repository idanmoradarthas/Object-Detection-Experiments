FROM jupyter/scipy-notebook:1386e2046833

# Install system packages
USER root
RUN sudo apt-get update
RUN sudo apt-get install --no-install-recommends -y \
 apt-utils\
 git \
 python-pil \
 python-lxml \
 protobuf-compiler \
 python-tk \
 vim \
 wget \
 unzip


# Install core packages
RUN pip install -U pip \
 numpy==1.16.2 \
 matplotlib \
 pandas \
 Cython \
 scipy \
 tensorflow==1.14.0 \
 contextlib2 \
 jupyter_contrib_nbextensions \
 googledrivedownloader \
 opencv-python

# Add dataframe display widget
RUN jupyter contrib nbextension install --sys-prefix

# Install tensorflow models object detection
RUN GIT_SSL_NO_VERIFY=true git clone https://github.com/tensorflow/models

# COCO API installation
RUN GIT_SSL_NO_VERIFY=true git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && make \
 && cp -r pycocotools models/research/
RUN sudo rm -r cocoapi/

RUN cd models/research && protoc object_detection/protos/*.proto --python_out=.

RUN cd models/research && python setup.py build && python setup.py install

# Set TF object detection available
ENV PYTHONPATH "$PYTHONPATH:/home/jovyan/models/research:/home/jovyan/models/research/slim"

EXPOSE 8888

RUN mkdir experiments
COPY experiments experiments/

CMD ["jupyter", "notebook"]