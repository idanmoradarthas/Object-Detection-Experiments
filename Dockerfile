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
 git+https://github.com/idanmoradarthas/DataScienceUtils.git \
 numpy==1.16.2 \
 matplotlib==3.0.3 \
 pandas \
 Cython \
 scipy \
 tensorflow==1.14.0 \
 contextlib2 \
 jupyter_contrib_nbextensions \
 googledrivedownloader \
 opencv-python \
 jupytext \
 requests

# Add dataframe display widget
RUN jupyter contrib nbextension install --sys-prefix

# Install tensorflow models object detection
RUN GIT_SSL_NO_VERIFY=true git clone https://github.com/tensorflow/models

# COCO API installation
RUN GIT_SSL_NO_VERIFY=true git clone https://github.com/cocodataset/cocoapi.git
RUN cd cocoapi/PythonAPI && make
RUN cp -r cocoapi/PythonAPI/pycocotools models/research/
RUN sudo rm -r cocoapi/

RUN cd models/research && protoc object_detection/protos/*.proto --python_out=.

RUN cd models/research && python setup.py build && python setup.py install

EXPOSE 8888

RUN mkdir notebooks
RUN mkdir downloaded_models
RUN mkdir images
RUN mkdir scripts
COPY notebooks notebooks/
COPY downloaded_models downloaded_models/
COPY images images/
COPY scripts scripts/

CMD ["jupyter", "lab", "--allow-root"]