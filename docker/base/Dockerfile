FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Tom Breuel <tmbdev@gmail.com>

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN apt-get update -y
RUN apt-get dist-upgrade -y

RUN apt-get -y install locales
RUN locale-gen en_US.UTF-8
RUN dpkg-reconfigure locales

RUN apt-get -y install sudo lsb-release
RUN apt-get -y install build-essential git software-properties-common
RUN apt-get -y install libhdf5-dev
RUN apt-get -y install vim-nox

RUN apt-get -y install nvidia-settings
RUN apt-get -y install tightvncserver
RUN apt-get -y install rxvt-unicode
RUN apt-get -y install blackbox
RUN apt-get -y install qiv
RUN apt-get -y install lynx

RUN apt-get -y install python python2.7-dev libpython2.7-dev
RUN apt-get -y install python-pip
RUN apt-get -y install python-tk
RUN apt-get -y install python-dev
RUN apt-get -y install python-cffi

# RUN apt-get -y install firefox
# RUN apt-get -y install imagemagick
# RUN apt-get -y install aria2
# RUN apt-get -y install net-tools
# RUN apt-get -y install socat netrw netcat
# RUN apt-get -y install mercurial git
# RUN apt-get -y install libxml2-dev libxslt-dev

RUN pip install --upgrade pip
RUN pip install setuptools
RUN pip install lxml
RUN pip install editdistance
RUN pip install msgpack-python
RUN pip install pyzmq
RUN pip install numpy
RUN pip install scipy
RUN pip install matplotlib
RUN pip install h5py

RUN pip install ipython[all]
RUN pip install https://s3.amazonaws.com/pytorch/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
RUN pip install torchvision

RUN pip install reportlab
RUN pip install psutil
RUN pip install simplejson

# RUN pip install pandas
# RUN pip install jupyter
# RUN pip install minio
# RUN pip install simplejson
# RUN pip install scikit-learn
# RUN pip install scikit-image
# RUN pip install sklearn-extras
# RUN pip install pycuda
# RUN pip install scikit-cuda
# RUN pip install pyfft
# RUN pip install dask[complete]
# RUN pip install reportlab
# RUN pip install Pillow
# RUN pip install olefile
