FROM tensorflow/tensorflow:latest-gpu-py3

ENV DARKFLOW_HOME=/qps-ai/darkflow
ENV IA_DATA=/qps-ai/data

MAINTAINER Vadim Delendik (vdelendik@qaprosoft.com)

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    ffmpeg \
    build-essential \
    cmake git pkg-config libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev \
        libpng-dev libtiff-dev libjasper-dev \
    python3-numpy \
    python3-pip \
    tesseract-ocr \
    nano

RUN mkdir OpenCV && cd OpenCV

RUN apt-get update && apt-get install -y \
  build-essential \
  checkinstall \
  cmake \
  pkg-config \
  yasm \
  libtiff5-dev \
  libjpeg-dev \
  libjasper-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libdc1394-22-dev \
 # libxine-dev \
  libgstreamer0.10-dev \
  libgstreamer-plugins-base0.10-dev \
  libv4l-dev \
  python-dev \
  python-numpy \
  python-pip \
  libtbb-dev \
  libeigen3-dev \
  libqt4-dev \
  libgtk2.0-dev \
  # Doesn't work libfaac-dev \
libgtk2.0-dev \
#  zlib1g-dev \
#  libavcodec-dev \
  unzip \
  libhdf5-dev \
  wget \
  sudo

RUN cd /opt && \
  wget https://github.com/opencv/opencv/archive/4.0.1.zip -O opencv-4.0.1.zip -nv && \
  unzip opencv-4.0.1.zip && \
  cd opencv-4.0.1 && \
  rm -rf build && \
  mkdir build && \
  cd build && \
  cmake -D CUDA_ARCH_BIN=3.2 \
    -D CUDA_ARCH_PTX=3.2 \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D WITH_V4L=ON \
    -D BUILD_TIFF=ON \
    -D WITH_QT=ON \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
 #  -D USE_GStreamer=ON \
    -D WITH_OPENGL=ON .. && \
  make -j4 && \
  make install && \
  echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/opencv.conf && \
  ldconfig
RUN cp /opt/opencv-4.0.1/build/lib/cv2.so /usr/lib/python2.7/dist-packages/cv2.so

WORKDIR /qps-ai/

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py

RUN \
    pip3 install --no-cache-dir Cython pytesseract scikit-image joblib

# install aws cli
RUN \
    pip install awscli --upgrade --user

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_BUCKET_URL=s3://ai.qaprosoft.com/
ARG AWS_REGION=us-east-1
ARG AWS_OUTPUT=json

# copy AI models and artifacts
# TODO: verify if it is possible to use aws without /root/.local/bin
RUN \
    export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    && export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    && export AWS_DEFAULT_OUTPUT=${AWS_OUTPUT} \
    && export AWS_DEFAULT_REGION=${AWS_REGION} \
    && mkdir -p ./data \
    && mkdir -p ./darkflow \
    && /root/.local/bin/aws s3 cp ${AWS_BUCKET_URL} ./data --recursive

COPY . /qps-ai/darkflow/

RUN \
    cd darkflow && \
    ls -l && \
    python3 setup.py build_ext --inplace
    #pip3 install .

#RUN \
#    cd /qps-ai/darkflow && \
#    ./scripts/deploy.sh

CMD nvidia-smi -q
RUN python3 -c "import Cython; print(Cython.__version__)"
RUN python3 -c "import cv2; print(cv2.__version__)"
RUN echo "cd /qps-ai/darkflow/scripts"
RUN echo "Usage: ./recognize.sh MODEL_NAME IMAGE_HOME OUTPUT_TYPE [img, json, xml]"
CMD ["bash"]
